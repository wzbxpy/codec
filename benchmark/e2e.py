import argparse
import hashlib
import json
import math
import os
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from random import randint, randrange, seed
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset
from transformers import AutoTokenizer

from codec.config import SYSTEM_PROMPT


@dataclass
class Node:
    node_id: str
    level: int
    tokens_len: int = 0  # 该节点持有的 token 段长度
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)


def build_full_kary_tree(k: int, depth: int) -> Dict[str, Node]:
    """
    构建满 k 叉树，深度 depth（根层=0，叶层=depth-1）
    返回 {node_id: Node}
    """
    assert depth >= 1
    nid_counter = 0

    def new_id():
        nonlocal nid_counter
        nid_counter += 1
        return f"n{nid_counter}"

    root_id = new_id()
    nodes: Dict[str, Node] = {root_id: Node(node_id=root_id, level=0)}
    q = deque([root_id])

    while q:
        nid = q.popleft()
        node = nodes[nid]
        if node.level < depth - 1:
            for _ in range(k):
                cid = new_id()
                nodes[cid] = Node(node_id=cid, level=node.level + 1, parent=nid)
                nodes[nid].children.append(cid)
                q.append(cid)
    return nodes


def build_degenerate_tree(depth: int) -> Dict[str, Node]:
    """
    构建退化树（只有“左孩子”的链），总深度=depth
    """
    assert depth >= 1
    nid_counter = 0

    def new_id():
        nonlocal nid_counter
        nid_counter += 1
        return f"n{nid_counter}"

    root_id = new_id()
    nodes: Dict[str, Node] = {root_id: Node(node_id=root_id, level=0)}
    cur = root_id
    for lvl in range(1, depth):
        cid = new_id()
        nodes[cid] = Node(node_id=cid, level=lvl, parent=cur)
        nodes[cur].children.append(cid)
        cur = cid
    return nodes


def enumerate_leaves(nodes: Dict[str, Node]) -> List[str]:
    return [nid for nid, n in nodes.items() if not n.children]


def path_root_to(nid: str, nodes: Dict[str, Node]) -> List[str]:
    path = []
    cur = nid
    while cur is not None:
        path.append(cur)
        cur = nodes[cur].parent
    return list(reversed(path))


def assign_tokens_by_level(nodes: Dict[str, Node], shared_tokens_per_internal_level: Dict[int, int], leaf_tokens: int) -> None:
    """
    为节点分配 token 段长度：
    - 内部节点（level 0..depth-2）：shared_tokens_per_internal_level[level]，无则 0
    - 叶子节点：leaf_tokens
    """
    for nid, node in nodes.items():
        if node.children:
            node.tokens_len = int(shared_tokens_per_internal_level.get(node.level, 0))
        else:
            node.tokens_len = int(leaf_tokens)


def _node_seed_from_id(node_id: str) -> int:
    """用 md5(node_id) 取前 8 hex 作为可复现种子"""
    h = hashlib.md5(node_id.encode("utf-8")).hexdigest()[:8]
    return int(h, 16)


def gen_token_ids_for_node(node_id: str, length: int, token_min: int = 3, token_max: int = 100) -> List[int]:
    """
    生成某个节点的 token 段（可复现 & 对所有后代序列共享）。
    - token id 均来自 [token_min, token_max]（含端点）
    """
    if length <= 0:
        return []
    rng = random.Random(_node_seed_from_id(node_id))
    vocab = list(range(token_min, token_max + 1))
    # 使用 choices 保证 O(length)，允许重复
    return rng.choices(vocab, k=length)


def build_sequences_from_tree(nodes: Dict[str, Node]) -> List[List[int]]:
    """
    根据树结构 + 各节点 tokens_len，生成每个叶子的完整序列（根→叶的所有节点拼接）。
    """
    # 为所有节点一次性生成 tokens，保证共享段在所有后代一致
    node_tokens: Dict[str, List[int]] = {nid: gen_token_ids_for_node(nid, n.tokens_len) for nid, n in nodes.items()}

    sequences: List[List[int]] = []
    leaves = sorted(enumerate_leaves(nodes), key=lambda x: int(x[1:]))
    for leaf in leaves:
        seq: List[int] = []
        for nid in path_root_to(leaf, nodes):
            seq.extend(node_tokens[nid])
        sequences.append(seq)
    return sequences


def solve_shared_tokens_for_ratio(total_request_len: int, batch_size: int, ratio: float) -> int:
    """
    共享比例 r 定义：shared_tokens / total_tokens_in_tree
    在二层树（根共享 S，叶私有 T-S）里：
        total_tokens_in_tree = S + B*(T - S) = B*T - (B-1)S
        r = S / (B*T - (B-1)S)
      => S = (r * B * T) / (1 + r * (B - 1))
    """
    B, T, r = batch_size, total_request_len, ratio
    S = (r * B * T) / (1 + r * (B - 1))
    return max(0, int(round(S)))


def scenario_vary_sequence_length(non_shared_lengths: List[int], root_shared: int = 4096) -> Dict[int, List[List[int]]]:
    """
    深度=2，二叉树；改变叶子非共享长度
    返回 {leaf_non_shared_len: [seq1, seq2, ...]}
    """
    out: Dict[int, List[List[int]]] = {}
    for L in non_shared_lengths:
        nodes = build_full_kary_tree(k=2, depth=4)
        assign_tokens_by_level(nodes, shared_tokens_per_internal_level={0: root_shared}, leaf_tokens=L)
        out[L] = build_sequences_from_tree(nodes)
    return out


def scenario_vary_batch_size(batch_sizes: List[int], root_shared: int = 2048, leaf_non_shared: int = 2048) -> Dict[int, List[List[int]]]:
    """
    深度=2；根共享固定，改变请求数（k=B）
    返回 {B: [seq1..seqB]}
    """
    out: Dict[int, List[List[int]]] = {}
    for B in batch_sizes:
        nodes = build_full_kary_tree(k=B, depth=2)
        assign_tokens_by_level(nodes, shared_tokens_per_internal_level={0: root_shared}, leaf_tokens=leaf_non_shared)
        out[B] = build_sequences_from_tree(nodes)
    return out


def scenario_vary_depth(depths: List[int], k: int = 2, shared_tokens_each_internal: int = 1024, leaf_non_shared: int = 1024) -> Dict[int, List[List[int]]]:
    """
    满 k 叉树，改变深度（2..6），内部各层共享长度相同
    返回 {depth: [seq...]}
    """
    out: Dict[int, List[List[int]]] = {}
    for d in depths:
        nodes = build_full_kary_tree(k=k, depth=d)
        shared_map = {lvl: shared_tokens_each_internal for lvl in range(0, d - 1)}
        assign_tokens_by_level(nodes, shared_map, leaf_non_shared)
        out[d] = build_sequences_from_tree(nodes)
    return out


def scenario_vary_shared_ratio(ratios: List[float], total_request_len: int = 65536, batch_size: int = 16) -> Dict[str, List[List[int]]]:
    """
    二层树；每请求总长度固定 T，改变 r 推导根共享 S，再令叶非共享为 T-S
    返回 {f"{r:.4f}": [seq...]}
    """
    out: Dict[str, List[List[int]]] = {}
    for r in ratios:
        S = solve_shared_tokens_for_ratio(total_request_len, batch_size, r)
        leaf_non_shared = max(0, total_request_len - S)
        nodes = build_full_kary_tree(k=batch_size, depth=2)
        assign_tokens_by_level(nodes, {0: S}, leaf_non_shared)
        out[f"{r:.4f}"] = build_sequences_from_tree(nodes)
    return out


def scenario_vary_tree_shape(k_list: List[int], depth: int = 3, shared_tokens_each_internal: int = 1024, leaf_non_shared: int = 1024, include_degenerate: bool = True) -> Dict[str, List[List[int]]]:
    """
    2T/3T/4T/5T + 退化树（DT）
    返回 {"2T": [seq...], "3T": [...], ..., "DT": [...]}
    """
    out: Dict[str, List[List[int]]] = {}
    for k in k_list:
        nodes = build_full_kary_tree(k=k, depth=depth)
        shared_map = {lvl: shared_tokens_each_internal for lvl in range(0, depth - 1)}
        assign_tokens_by_level(nodes, shared_map, leaf_non_shared)
        out[f"{k}T"] = build_sequences_from_tree(nodes)
    if include_degenerate:
        nodes = build_degenerate_tree(depth=depth)
        shared_map = {lvl: shared_tokens_each_internal for lvl in range(0, depth - 1)}
        assign_tokens_by_level(nodes, shared_map, leaf_non_shared)
        out["DT"] = build_sequences_from_tree(nodes)
    return out


def build_parser():
    p = argparse.ArgumentParser("Prefix-tree → seq token-id lists (IDs in [3,100])")
    p.add_argument("--index", required=True, choices=["0", "1", "2", "3", "4"])
    p.add_argument("--scenario", required=True, choices=["vary-seq", "vary-batch", "vary-depth", "vary-ratio", "vary-shape"])
    p.add_argument("--write-json", type=str, default="", help="将结果写入指定 JSON 文件（避免终端打印巨量序列）")

    # vary-seq
    p.add_argument("--seq-list", type=str, default="128,256,512,1024,2048")
    p.add_argument("--root-shared", type=int, default=4096)

    # vary-batch
    p.add_argument("--batches", type=str, default="2,4,8,16,32")
    p.add_argument("--root-shared-batch", type=int, default=2048)
    p.add_argument("--leaf-non-shared", type=int, default=2048)

    # vary-depth
    p.add_argument("--depths", type=str, default="2,3,4,5,6")
    p.add_argument("--k-arity", type=int, default=2)
    p.add_argument("--internal-shared", type=int, default=1024)
    p.add_argument("--leaf-non-shared-depth", type=int, default=1024)

    # vary-ratio
    p.add_argument("--ratios", type=str, default="0.05,0.1,0.2,0.5,0.8")
    p.add_argument("--total-len", type=int, default=65536)
    p.add_argument("--batch-size", type=int, default=16)

    # vary-shape
    p.add_argument("--k-list", type=str, default="2,3,4,5")
    p.add_argument("--shape-depth", type=int, default=3)
    p.add_argument("--shape-internal-shared", type=int, default=1024)
    p.add_argument("--shape-leaf-non-shared", type=int, default=1024)
    p.add_argument("--include-DT", default=True)

    return p


def main():
    backend = os.getenv("LLM_BACKEND")

    if backend == "vllm":
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        from vllm import LLM, SamplingParams
    elif backend == "codec":
        from codec import LLM, SamplingParams

    seed(0)

    ds_name = os.getenv("DATASET")
    path = os.path.expanduser("~/huggingface/Qwen3-4B-fp16/")
    llm = LLM(path, enforce_eager=True, max_model_len=131072)
    if ds_name:
        ds = load_dataset("bigai-nlco/LooGLE", ds_name)
        texts = [f"{SYSTEM_PROMPT} {c.strip()} {q.strip()}" for c, q in zip(ds["test"]["context"], ds["test"]["question"])]
        start = randrange(0, len(texts) - 128 + 1)
        texts = texts[start : start + 128]
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        enc = tokenizer(texts, add_special_tokens=False, truncation=True, padding=False, return_attention_mask=False, return_token_type_ids=False)
        prompt_token_ids = enc["input_ids"]
    else:
        args = build_parser().parse_args()

        if args.scenario == "vary-seq":
            seqs = [int(x) for x in args.seq_list.split(",") if x.strip()]
            result = scenario_vary_sequence_length(seqs, root_shared=args.root_shared)

        elif args.scenario == "vary-batch":
            batches = [int(x) for x in args.batches.split(",") if x.strip()]
            result = scenario_vary_batch_size(batches, root_shared=args.root_shared_batch, leaf_non_shared=args.leaf_non_shared)

        elif args.scenario == "vary-depth":
            depths = [int(x) for x in args.depths.split(",") if x.strip()]
            result = scenario_vary_depth(depths, k=args.k_arity, shared_tokens_each_internal=args.internal_shared, leaf_non_shared=args.leaf_non_shared_depth)

        elif args.scenario == "vary-ratio":
            ratios = [float(x) for x in args.ratios.split(",") if x.strip()]
            result = scenario_vary_shared_ratio(ratios, total_request_len=args.total_len, batch_size=args.batch_size)

        elif args.scenario == "vary-shape":
            k_list = [int(x) for x in args.k_list.split(",") if x.strip()]
            result = scenario_vary_tree_shape(k_list, depth=args.shape_depth, shared_tokens_each_internal=args.shape_internal_shared, leaf_non_shared=args.shape_leaf_non_shared, include_degenerate=args.include_DT)
        else:
            raise ValueError("unknown scenario")

        _, prompt_token_ids = list(result.items())[int(args.index)]

    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True) for _ in range(len(prompt_token_ids))]
    if backend == "vllm":
        prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    llm.generate(prompt_token_ids, sampling_params, use_tqdm=True)


if __name__ == "__main__":
    main()
