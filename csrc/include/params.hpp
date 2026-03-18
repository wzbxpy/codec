#pragma once

#include <cstdint>

#include <cute/tensor.hpp>

using namespace cute;

struct Stage1Params {
    using index_t = int64_t;

    // The Q matrix
    void* __restrict__ q_ptr;

    // The KV matrices
    void* __restrict__ k_ptr;
    void* __restrict__ v_ptr;

    // The out and lse matrix
    void* __restrict__ outs_ptr;
    void* __restrict__ lses_ptr;

    // The KV page info
    int num_pages, num_active_pages, page_size;

    // The scaling factor
    half softmax_scale;
    half softmax_scale_log2;

    // The KV page metadata
    struct KVPageMetadata {
        int q_token_ofs;
        int q_token_length;
        int kv_page_idx;
        int kv_page_length;
    };
    KVPageMetadata* __restrict__ kv_page_metadata_ptr;
};

struct Stage2Params {
    using index_t = int64_t;

    // The out and lse matrix.
    void* __restrict__ outs_ptr;
    void* __restrict__ lses_ptr;

    // The KV node metadata
    struct KVNodeMetadata {
        int o_token_ofs;
        int o_token_length;
        int num_splits;
    };
    KVNodeMetadata* __restrict__ kv_node_metadata_ptr;

    // The number of KV nodes
    int num_nodes;
};

struct Stage3Params {
    using index_t = int64_t;

    // The out and lse matrix.
    void* __restrict__ outs_ptr;
    void* __restrict__ lses_ptr;

    // The KV edge metadata
    struct KVEdgeMetadata {
        int parent_token_ofs;
        int child_token_ofs;
        int token_length;
    };
    KVEdgeMetadata* __restrict__ kv_edge_metadata_ptr;

    // The number of edges
    int num_edges;
};
