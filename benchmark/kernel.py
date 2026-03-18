import argparse
import math
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from codec_gpu import build_prefix_tree, tree_attn
from flash_attn.flash_attn_interface import flash_attn_with_kvcache


@dataclass
class Node:
    node_id: str
    level: int
    tokens: int = 0
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)


@dataclass
class Block:
    block_id: str
    node_id: str
    level: int
    start_idx: int  # 节点内 token 起始下标
    length: int  # 实际块长（<= block_size）
    is_shared: bool
    leaf_desc_count: int


# ==============================
# 树构建
# ==============================


def build_full_kary_tree(k: int, depth: int) -> Dict[str, Node]:
    assert depth >= 1
    nid = 0

    def new_id():
        nonlocal nid
        nid += 1
        return f"n{nid}"

    root = new_id()
    nodes = {root: Node(root, 0)}
    q = deque([root])
    while q:
        cur = q.popleft()
        if nodes[cur].level < depth - 1:
            for _ in range(k):
                cid = new_id()
                nodes[cid] = Node(cid, nodes[cur].level + 1, parent=cur)
                nodes[cur].children.append(cid)
                q.append(cid)
    return nodes


def build_degenerate_tree(depth: int) -> Dict[str, Node]:
    assert depth >= 1
    nid = 0

    def new_id():
        nonlocal nid
        nid += 1
        return f"n{nid}"

    root = new_id()
    nodes = {root: Node(root, 0)}
    cur = root
    for lvl in range(1, depth):
        cid = new_id()
        nodes[cid] = Node(cid, lvl, parent=cur)
        nodes[cur].children.append(cid)
        cur = cid
    return nodes


def enumerate_leaves(nodes: Dict[str, Node]) -> List[str]:
    return [i for i, n in nodes.items() if not n.children]


def path_root_to(nid: str, nodes: Dict[str, Node]) -> List[str]:
    path = []
    cur = nid
    while cur is not None:
        path.append(cur)
        cur = nodes[cur].parent
    return list(reversed(path))


def compute_leaf_desc_counts(nodes: Dict[str, Node]) -> Dict[str, int]:
    by_level = defaultdict(list)
    max_lvl = 0
    for nid, n in nodes.items():
        by_level[n.level].append(nid)
        max_lvl = max(max_lvl, n.level)
    leaf_counts = {}
    for lvl in range(max_lvl, -1, -1):
        for nid in by_level[lvl]:
            n = nodes[nid]
            if not n.children:
                leaf_counts[nid] = 1
            else:
                leaf_counts[nid] = sum(leaf_counts[c] for c in n.children)
    return leaf_counts


# ==============================
# Token 分配 & Block 构建
# ==============================


def assign_tokens_by_level(nodes: Dict[str, Node], shared_tokens_per_internal_level: Dict[int, int], leaf_tokens: int) -> None:
    for nid, n in nodes.items():
        if n.children:
            n.tokens = int(shared_tokens_per_internal_level.get(n.level, 0))
        else:
            n.tokens = int(leaf_tokens)


def make_blocks(nodes: Dict[str, Node], block_size: int) -> Tuple[List[Block], Dict[str, List[str]]]:
    leaf_counts = compute_leaf_desc_counts(nodes)
    blocks: List[Block] = []
    node2bids: Dict[str, List[str]] = {}
    bid = 0

    def new_bid():
        nonlocal bid
        bid += 1
        return f"b{bid}"

    levels = sorted({n.level for n in nodes.values()})
    for L in levels:
        ids = sorted([nid for nid, n in nodes.items() if n.level == L], key=lambda x: int(x[1:]))
        for nid in ids:
            seg = nodes[nid].tokens
            node2bids[nid] = []
            if seg <= 0:
                continue
            num_blocks = math.ceil(seg / block_size)
            for i in range(num_blocks):
                s = i * block_size
                ln = min(block_size, seg - s)
                b = new_bid()
                blocks.append(
                    Block(
                        block_id=b,
                        node_id=nid,
                        level=L,
                        start_idx=s,
                        length=ln,
                        is_shared=(leaf_counts[nid] >= 2),
                        leaf_desc_count=leaf_counts[nid],
                    )
                )
                node2bids[nid].append(b)
    return blocks, node2bids


def build_request_block_ids(nodes: Dict[str, Node], node2bids: Dict[str, List[str]]) -> Dict[str, List[str]]:
    reqs = {}
    leaves = sorted(enumerate_leaves(nodes), key=lambda x: int(x[1:]))
    for i, leaf in enumerate(leaves, start=1):
        seq = []
        for nid in path_root_to(leaf, nodes):
            seq.extend(node2bids.get(nid, []))
        reqs[f"req{i}"] = seq
    return reqs


def request_token_lens(nodes: Dict[str, Node]) -> List[int]:
    lens = []
    leaves = sorted(enumerate_leaves(nodes), key=lambda x: int(x[1:]))
    for leaf in leaves:
        total = 0
        for nid in path_root_to(leaf, nodes):
            total += nodes[nid].tokens
        lens.append(total)
    return lens


# ==============================
# FlashAttn/Paged-Attention 风格张量
# ==============================


def build_flashattn_block_table(req_to_block_ids: Dict[str, List[str]], blocks: List[Block], block_size: int, pad_val: int = -1):
    """
    返回:
      block_table (int32)        : [num_seqs, max_num_blocks], -1 padding
      context_lens (int32)       : [num_seqs]  —— 需要你传给内核的真实 token 长度
      num_blocks_per_seq (int32) : [num_seqs]
      block_size (int)           : 统一块大小
      blocks_meta (list of dict) : 全局块元数据（顺序对应 block_table 中的物理块索引）
      block_id_to_index (dict)   : bX → 0..N-1
    """
    # 1) 全局块索引（0-based）
    block_id_to_index = {blk.block_id: idx for idx, blk in enumerate(blocks)}
    blocks_meta = [asdict(blk) for blk in blocks]

    # 2) 统计各序列块数与最大块数
    req_ids = sorted(req_to_block_ids.keys(), key=lambda x: int(x[3:]))
    num_seqs = len(req_ids)
    num_blocks_list = [len(req_to_block_ids[r]) for r in req_ids]
    max_blocks = max(num_blocks_list) if num_blocks_list else 0

    # 3) 构造 block_table
    bt = torch.full((num_seqs, max_blocks), pad_val, dtype=torch.int32)
    for i, rid in enumerate(req_ids):
        idxs = [block_id_to_index[b] for b in req_to_block_ids[rid]]
        if idxs:
            bt[i, : len(idxs)] = torch.tensor(idxs, dtype=torch.int32)

    # 4) num_blocks_per_seq
    num_blocks_per_seq = torch.tensor(num_blocks_list, dtype=torch.int32)

    return bt, num_blocks_per_seq, block_size, blocks_meta, block_id_to_index


def build_context_lens_tensor(nodes: Dict[str, Node]) -> torch.Tensor:
    lens = request_token_lens(nodes)
    return torch.tensor(lens, dtype=torch.int32)


# ==============================
# 场景封装
# ==============================


def solve_shared_tokens_for_ratio(T: int, B: int, r: float) -> int:
    # r = S / (B*T - (B-1)S)  =>  S = (r * B * T) / (1 + r*(B-1))
    S = (r * B * T) / (1 + r * (B - 1))
    return max(0, int(round(S)))


def scenario_vary_sequence_length(non_shared_list, root_shared=4096, block_size=16):
    out = {}
    for L in non_shared_list:
        nodes = build_full_kary_tree(2, 4)
        assign_tokens_by_level(nodes, {0: root_shared}, leaf_tokens=L)
        blocks, node2bids = make_blocks(nodes, block_size)
        req_map = build_request_block_ids(nodes, node2bids)
        block_table, num_blocks_per_seq, block_size, blocks_meta, bid2idx = build_flashattn_block_table(req_map, blocks, block_size)
        context_lens = build_context_lens_tensor(nodes)
        out[L] = dict(
            block_table=block_table,
            context_lens=context_lens,
            num_blocks_per_seq=num_blocks_per_seq,
            block_size=block_size,
            blocks_meta=blocks_meta,
            block_id_to_index=bid2idx,
        )
    return out


def scenario_vary_batch_size(batch_sizes, root_shared=120_000, leaf_non_shared=2048, block_size=16):
    out = {}
    for B in batch_sizes:
        nodes = build_full_kary_tree(B, 2)  # 深度=2，k=B => B 个请求
        assign_tokens_by_level(nodes, {0: root_shared}, leaf_tokens=leaf_non_shared)
        blocks, node2bids = make_blocks(nodes, block_size)
        req_map = build_request_block_ids(nodes, node2bids)
        block_table, num_blocks_per_seq, block_size, blocks_meta, bid2idx = build_flashattn_block_table(req_map, blocks, block_size)
        context_lens = build_context_lens_tensor(nodes)
        out[B] = dict(
            block_table=block_table,
            context_lens=context_lens,
            num_blocks_per_seq=num_blocks_per_seq,
            block_size=block_size,
            blocks_meta=blocks_meta,
            block_id_to_index=bid2idx,
        )
    return out


def scenario_vary_depth(depths, k=2, internal_shared=1024, leaf_non_shared=1024, block_size=16):
    out = {}
    for d in depths:
        nodes = build_full_kary_tree(k, d)
        shared_map = {lvl: internal_shared for lvl in range(0, d - 1)}
        assign_tokens_by_level(nodes, shared_map, leaf_tokens=leaf_non_shared)
        blocks, node2bids = make_blocks(nodes, block_size)
        req_map = build_request_block_ids(nodes, node2bids)
        block_table, num_blocks_per_seq, block_size, blocks_meta, bid2idx = build_flashattn_block_table(req_map, blocks, block_size)
        context_lens = build_context_lens_tensor(nodes)
        out[d] = dict(
            block_table=block_table,
            context_lens=context_lens,
            num_blocks_per_seq=num_blocks_per_seq,
            block_size=block_size,
            blocks_meta=blocks_meta,
            block_id_to_index=bid2idx,
        )
    return out


def scenario_vary_ratio(ratios, total_len=120_000, batch_size=16, block_size=16):
    out = {}
    for r in ratios:
        S = solve_shared_tokens_for_ratio(total_len, batch_size, r)
        leaf_non_shared = max(0, total_len - S)
        nodes = build_full_kary_tree(batch_size, 2)
        assign_tokens_by_level(nodes, {0: S}, leaf_tokens=leaf_non_shared)
        blocks, node2bids = make_blocks(nodes, block_size)
        req_map = build_request_block_ids(nodes, node2bids)
        block_table, num_blocks_per_seq, block_size, blocks_meta, bid2idx = build_flashattn_block_table(req_map, blocks, block_size)
        context_lens = build_context_lens_tensor(nodes)
        out[f"{r:.4f}"] = dict(
            block_table=block_table,
            context_lens=context_lens,
            num_blocks_per_seq=num_blocks_per_seq,
            block_size=block_size,
            blocks_meta=blocks_meta,
            block_id_to_index=bid2idx,
        )
    return out


def scenario_vary_shape(k_list, depth=3, internal_shared=1024, leaf_non_shared=1024, block_size=16, include_dt=True):
    out = {}
    for k in k_list:
        nodes = build_full_kary_tree(k, depth)
        shared_map = {lvl: internal_shared for lvl in range(0, depth - 1)}
        assign_tokens_by_level(nodes, shared_map, leaf_tokens=leaf_non_shared)
        blocks, node2bids = make_blocks(nodes, block_size)
        req_map = build_request_block_ids(nodes, node2bids)
        block_table, num_blocks_per_seq, block_size, blocks_meta, bid2idx = build_flashattn_block_table(req_map, blocks, block_size)
        context_lens = build_context_lens_tensor(nodes)
        out[f"{k}T"] = dict(
            block_table=block_table,
            context_lens=context_lens,
            num_blocks_per_seq=num_blocks_per_seq,
            block_size=block_size,
            blocks_meta=blocks_meta,
            block_id_to_index=bid2idx,
        )
    if include_dt:
        nodes = build_degenerate_tree(depth)
        shared_map = {lvl: internal_shared for lvl in range(0, depth - 1)}
        assign_tokens_by_level(nodes, shared_map, leaf_tokens=leaf_non_shared)
        blocks, node2bids = make_blocks(nodes, block_size)
        req_map = build_request_block_ids(nodes, node2bids)
        block_table, num_blocks_per_seq, block_size, blocks_meta, bid2idx = build_flashattn_block_table(req_map, blocks, block_size)
        context_lens = build_context_lens_tensor(nodes)
        out["DT"] = dict(
            block_table=block_table,
            context_lens=context_lens,
            num_blocks_per_seq=num_blocks_per_seq,
            block_size=block_size,
            blocks_meta=blocks_meta,
            block_id_to_index=bid2idx,
        )
    return out


# ==============================
# CLI & Demo
# ==============================


def build_parser():
    p = argparse.ArgumentParser("Prefix-tree → FlashAttention BlockTable (torch)")
    p.add_argument("--scenario", required=True, choices=["vary-seq", "vary-batch", "vary-depth", "vary-ratio", "vary-shape"])
    p.add_argument("--block-size", type=int, default=1024)

    # vary-seq
    p.add_argument("--seq-list", type=str, default="128,256,512,1024,2048")
    p.add_argument("--root-shared", type=int, default=4096)

    # vary-batch
    p.add_argument("--batches", type=str, default="4,8,16,32,64")
    p.add_argument("--root-shared-batch", type=int, default=2048)
    p.add_argument("--leaf-non-shared", type=int, default=2048)

    # vary-depth
    p.add_argument("--depths", type=str, default="2,3,4,5,6")
    p.add_argument("--k-arity", type=int, default=2)
    p.add_argument("--internal-shared", type=int, default=1024)
    p.add_argument("--leaf-non-shared-depth", type=int, default=1024)

    # vary-ratio
    p.add_argument("--ratios", type=str, default="0.05,0.1,0.2,0.5,0.8")
    p.add_argument("--total-len", type=int, default=120000)
    p.add_argument("--batch-size", type=int, default=16)

    # vary-shape
    p.add_argument("--k-list", type=str, default="2,3,4,5")
    p.add_argument("--shape-depth", type=int, default=3)
    p.add_argument("--shape-internal-shared", type=int, default=1024)
    p.add_argument("--shape-leaf-non-shared", type=int, default=1024)
    p.add_argument("--include-DT", default=True)

    return p


def main():
    args = build_parser().parse_args()
    bs = args.block_size

    if args.scenario == "vary-seq":
        seqs = [int(x) for x in args.seq_list.split(",") if x.strip()]
        pack = scenario_vary_sequence_length(seqs, root_shared=args.root_shared, block_size=bs)

    elif args.scenario == "vary-batch":
        batches = [int(x) for x in args.batches.split(",") if x.strip()]
        pack = scenario_vary_batch_size(batches, root_shared=args.root_shared_batch, leaf_non_shared=args.leaf_non_shared, block_size=bs)

    elif args.scenario == "vary-depth":
        depths = [int(x) for x in args.depths.split(",") if x.strip()]
        pack = scenario_vary_depth(depths, k=args.k_arity, internal_shared=args.internal_shared, leaf_non_shared=args.leaf_non_shared_depth, block_size=bs)

    elif args.scenario == "vary-ratio":
        ratios = [float(x) for x in args.ratios.split(",") if x.strip()]
        pack = scenario_vary_ratio(ratios, total_len=args.total_len, batch_size=args.batch_size, block_size=bs)

    elif args.scenario == "vary-shape":
        k_list = [int(x) for x in args.k_list.split(",") if x.strip()]
        pack = scenario_vary_shape(k_list, depth=args.shape_depth, internal_shared=args.shape_internal_shared, leaf_non_shared=args.shape_leaf_non_shared, block_size=bs, include_dt=args.include_DT)

    for k, d in pack.items():
        block_table = d["block_table"]
        block_table = block_table.detach().cpu().tolist()
        block_table.sort()
        block_table = torch.tensor(block_table, dtype=torch.int32, device="cuda")
        seqlens_k = d["context_lens"].cuda()
        tot_blocks = sum(d["num_blocks_per_seq"])
        print(f"k = {k}")

        batch_size = block_table.shape[0]
        query = torch.rand((batch_size, 32, 128), dtype=torch.float16, device="cuda")
        key = torch.rand((tot_blocks, d["block_size"], 8, 128), dtype=torch.float16, device="cuda")
        value = torch.rand((tot_blocks, d["block_size"], 8, 128), dtype=torch.float16, device="cuda")
        pages, nodes, edges, leaves, max_splits = build_prefix_tree(seqlens_k, block_table, d["block_size"], 16)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        run = 4

        for _ in range(4):
            flash_attn_with_kvcache(query.unsqueeze(1), key, value, cache_seqlens=seqlens_k, block_table=block_table)
            tree_attn(query, key, value, pages, nodes, edges, leaves, max_splits)
        torch.cuda.nvtx.range_push("flash")
        start_event.record()
        for _ in range(run):
            flash_attn_with_kvcache(query.unsqueeze(1), key, value, cache_seqlens=seqlens_k, block_table=block_table)
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)
        print(f"Flash: {elapsed_time_ms / run}")
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("tree")
        start_event.record()
        for _ in range(run):
            tree_attn(query, key, value, pages, nodes, edges, leaves, max_splits)
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)
        print(f"CoDec: {elapsed_time_ms / run}")
        torch.cuda.nvtx.range_pop()


if __name__ == "__main__":
    main()
