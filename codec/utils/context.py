from dataclasses import dataclass
import torch


@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    pages: torch.Tensor | None = None
    nodes: torch.Tensor | None = None
    edges: tuple[torch.Tensor] = tuple()
    leaves: torch.Tensor | None = None
    max_splits: int = 0

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)

def set_codec(pages, nodes, edges, leaves, max_splits):
    global _CONTEXT
    _CONTEXT.pages = pages
    _CONTEXT.nodes = nodes
    _CONTEXT.edges = edges
    _CONTEXT.leaves = leaves
    _CONTEXT.max_splits = max_splits

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
