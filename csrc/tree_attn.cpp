#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/python.h>

#include "dispatch.hpp"
#include "params.hpp"

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

template <int HeadDim, int NHeads, int NHeadsKV, int PageSize>
void launch_tree_attn_stage1(const Stage1Params& params, cudaStream_t stream);

template <int HeadDim, int NHeads, int NSplits>
void launch_tree_attn_stage2(const Stage2Params& params, cudaStream_t stream);

template <int HeadDim, int NHeads>
void launch_tree_attn_stage3(const Stage3Params& params, cudaStream_t stream);

std::tuple<
    at::Tensor,               // KVPageMetadata
    at::Tensor,               // KVNodeMetadata
    std::vector<at::Tensor>,  // KVEdgeMetadata
    at::Tensor,               // KVLeaves
    int                       // max_splits
    >
build_prefix_tree(
    const at::Tensor& seqlens_k,
    const at::Tensor& block_table,
    const int page_size,
    const int block_q) {
    const auto batch_size = block_table.size(0);
    const auto max_num_pages = block_table.size(1);

    const at::Tensor block_table_cpu = block_table.cpu();
    auto block_table_acc = block_table_cpu.accessor<int, 2>();
    const at::Tensor seqlens_k_cpu = seqlens_k.cpu();
    auto seqlens_k_acc = seqlens_k_cpu.accessor<int, 1>();

    std::vector<Stage1Params::KVPageMetadata> pages;
    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        for (int i = 0; i < max_num_pages; ++i) {
            const int page_idx = block_table_acc[batch_idx][i];
            if (page_idx == -1) {
                break;
            }

            int& remaining_kv_length = seqlens_k_acc[batch_idx];
            TORCH_CHECK(remaining_kv_length > 0, "ill-formed block_table or seqlens_k");
            int kv_page_length = remaining_kv_length > page_size ? page_size : remaining_kv_length;
            auto it = std::find_if(
                pages.rbegin(),
                pages.rend(),
                [=](const auto& e) {
                    return e.kv_page_idx == page_idx;
                });
            if (it == pages.rend() || it->q_token_length == block_q) {
                // Enqueue page
                pages.emplace_back(
                    Stage1Params::KVPageMetadata{
                        .q_token_ofs = batch_idx,
                        .q_token_length = 1,
                        .kv_page_idx = page_idx,
                        .kv_page_length = kv_page_length,
                    });
            } else {
                // Update page metadata
                TORCH_CHECK(it->kv_page_length == kv_page_length, "shared kv block must share the same block length");
                ++(it->q_token_length);
            }
            remaining_kv_length -= kv_page_length;
        }
    }

    const int num_active_pages = pages.size();
    int max_splits = 0;

    int prev_q_token_ofs = pages[0].q_token_ofs;
    Stage2Params::KVNodeMetadata prev_node{
        .o_token_ofs = 0,
        .o_token_length = pages[0].q_token_length,
        .num_splits = 1,
    };
    std::vector<Stage2Params::KVNodeMetadata> nodes;
    std::map<int, std::vector<Stage3Params::KVEdgeMetadata>> edges;  // depth -> edges
    std::vector<int> leaves;
    std::stack<std::tuple<int, int, int>> path;  // q_ofs, q_length, o_ofs
    nodes.reserve(num_active_pages / 2);
    // Add edge (and pop siblings) when enter a new node
    for (int i = 1; i < num_active_pages; ++i) {
        int cur_q_token_ofs = pages[i].q_token_ofs;
        int cur_q_token_length = pages[i].q_token_length;
        if (cur_q_token_ofs == prev_q_token_ofs && cur_q_token_length == prev_node.o_token_length) {
            // Still inside current node
            ++prev_node.num_splits;
        } else {
            // Entering new node
            max_splits = std::max(max_splits, prev_node.num_splits);
            nodes.push_back(std::move(prev_node));
            if (cur_q_token_ofs >= prev_q_token_ofs && cur_q_token_ofs + cur_q_token_length <= prev_q_token_ofs + prev_node.o_token_length) {
                // Entering child node
                path.emplace(prev_q_token_ofs, prev_node.o_token_length, prev_node.o_token_ofs);
                edges[path.size() - 1].emplace_back(Stage3Params::KVEdgeMetadata{
                    .parent_token_ofs = prev_node.o_token_ofs + (cur_q_token_ofs - prev_q_token_ofs),
                    .child_token_ofs = i * block_q,
                    .token_length = cur_q_token_length,
                });
            } else {
                // Arrive passing the end of path
                for (int j = 0; j < prev_node.o_token_length; ++j) {
                    leaves.push_back(prev_node.o_token_ofs + j);
                }
                int q_token_ofs, q_token_length, o_token_ofs;
                while (!path.empty()) {
                    std::tie(q_token_ofs, q_token_length, o_token_ofs) = path.top();
                    if (cur_q_token_ofs >= q_token_ofs && cur_q_token_ofs + cur_q_token_length <= q_token_ofs + q_token_length) {
                        break;
                    }
                    path.pop();
                }
                if (!path.empty()) {
                    edges[path.size() - 1].emplace_back(Stage3Params::KVEdgeMetadata{
                        .parent_token_ofs = o_token_ofs + (cur_q_token_ofs - prev_q_token_ofs),
                        .child_token_ofs = i * block_q,
                        .token_length = cur_q_token_length,
                    });
                }
            }
            prev_node.o_token_ofs = i * block_q;
            prev_node.o_token_length = cur_q_token_length;
            prev_node.num_splits = 1;
            prev_q_token_ofs = cur_q_token_ofs;
        }
    }
    for (int j = 0; j < prev_node.o_token_length; ++j) {
        leaves.push_back(prev_node.o_token_ofs + j);
    }
    nodes.emplace_back(std::move(prev_node));

    at::TensorOptions opts = at::TensorOptions().dtype(at::kInt).device(at::kCPU);

    auto pages_cpu = torch::from_blob(reinterpret_cast<int*>(pages.data()),
                                      {static_cast<long>(pages.size()), 4},
                                      opts);
    auto pages_cpu_pinned = pages_cpu.pin_memory();
    auto pages_gpu = pages_cpu_pinned.to(torch::kCUDA, true);

    auto nodes_cpu = torch::from_blob(reinterpret_cast<int*>(nodes.data()),
                                      {static_cast<long>(nodes.size()), 3},
                                      opts);
    auto nodes_cpu_pinned = nodes_cpu.pin_memory();
    auto nodes_gpu = nodes_cpu_pinned.to(torch::kCUDA, true);

    std::vector<at::Tensor> edges_gpu;
    for (int i = 0; i < edges.size(); ++i) {
        auto& edges_per_depth = edges.at(i);
        auto edges_cpu = torch::from_blob(reinterpret_cast<int*>(edges_per_depth.data()),
                                          {static_cast<long>(edges_per_depth.size()), 3},
                                          opts);
        auto edges_cpu_pinned = edges_cpu.pin_memory();
        edges_gpu.emplace_back(edges_cpu_pinned.to(torch::kCUDA, true));
    }
    auto leaves_gpu = torch::tensor(leaves, torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA));

    return {pages_gpu, nodes_gpu, edges_gpu, leaves_gpu, max_splits};
}

at::Tensor
tree_attn_interface(
    const at::Tensor& q,      // (batch_size, num_heads_qo, head_size)
    const at::Tensor& k,      // (num_pages, page_size, num_heads_kv, head_size)
    const at::Tensor& v,      // (num_pages, page_size, num_heads_kv, head_size)
    const at::Tensor& pages,  // (batch_size,)
    const at::Tensor& nodes,  // (batch_size,)
    const std::vector<at::Tensor>& edges,
    const at::Tensor& leaves,  // (batch_size, max_num_pages)
    const unsigned max_splits) {
    // Otherwise the kernel will be launched from cuda:0 device
    at::cuda::CUDAGuard device_guard{q.device()};
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    static constexpr int kBlockQ = 16;

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16, "only support fp16 data type");

    const int batch_size = q.size(0);
    const int num_heads = q.size(1);
    const int head_dim = q.size(2);
    const int num_pages = k.size(0);
    const int page_size = k.size(1);
    const int num_heads_kv = k.size(2);

    CHECK_DEVICE(q);
    CHECK_CONTIGUOUS(q);
    CHECK_SHAPE(q, batch_size, num_heads, head_dim);

    CHECK_DEVICE(k);
    CHECK_CONTIGUOUS(k);
    CHECK_SHAPE(k, num_pages, page_size, num_heads_kv, head_dim);

    CHECK_DEVICE(v);
    CHECK_CONTIGUOUS(v);
    CHECK_SHAPE(v, num_pages, page_size, num_heads_kv, head_dim);

    const int num_active_pages = pages.size(0);
    // (num_active_pages, block_qo, num_heads, head_dim)
    at::Tensor outs = torch::empty({num_active_pages, kBlockQ, num_heads, head_dim}, q.options());
    // (num_active_pages, block_qo, num_heads)
    at::Tensor lses = torch::empty({num_active_pages, kBlockQ, num_heads}, q.options());

    Stage1Params stage1_params{
        .q_ptr = q.data_ptr(),
        .k_ptr = k.data_ptr(),
        .v_ptr = v.data_ptr(),

        .outs_ptr = outs.data_ptr(),
        .lses_ptr = lses.data_ptr(),

        .num_pages = num_pages,
        .num_active_pages = num_active_pages,
        .page_size = page_size,

        .softmax_scale = static_cast<half>(1 / std::sqrt(head_dim)),
        .softmax_scale_log2 = static_cast<half>(1 / std::sqrt(head_dim) * M_LOG2E),

        .kv_page_metadata_ptr = reinterpret_cast<Stage1Params::KVPageMetadata*>(pages.data_ptr()),
    };

    value_switch<int, 64, 128>(
        head_dim,
        [&](const auto expr) {
            static constexpr int kHeadDim = decltype(expr)::value;
            value_switch<std::pair<int, int>, {32, 8}>(
                {num_heads, num_heads_kv},
                [&](const auto expr) {
                    static constexpr int kNHeads = decltype(expr)::value.first;
                    static constexpr int kNHeadsKV = decltype(expr)::value.second;
                    value_switch<int, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096>(
                        page_size,
                        [&](const auto expr) {
                            static constexpr int kPageSize = decltype(expr)::value;
                            launch_tree_attn_stage1<kHeadDim, kNHeads, kNHeadsKV, kPageSize>(stage1_params, stream);
                        },
                        [] {
                            TORCH_CHECK(false, "unsupported page_size");
                        });
                },
                [] {
                    TORCH_CHECK(false, "unsupported (num_heads, num_heads_kv)");
                });
        },
        [] {
            TORCH_CHECK(false, "unsupported head_dim");
        });

    Stage2Params stage2_params{
        .outs_ptr = outs.data_ptr(),
        .lses_ptr = lses.data_ptr(),

        .kv_node_metadata_ptr = reinterpret_cast<Stage2Params::KVNodeMetadata*>(nodes.data_ptr()),

        .num_nodes = static_cast<int>(nodes.size(0)),
    };

    value_switch<int, 64, 128>(
        head_dim,
        [&](const auto expr) {
            static constexpr int kHeadDim = decltype(expr)::value;
            value_switch<int, 32>(
                num_heads,
                [&](const auto expr) {
                    static constexpr int kNHeads = decltype(expr)::value;
                    value_switch<int, 1, 2, 4, 8, 16, 32, 64, 128>(
                        std::bit_ceil(max_splits),
                        [&](const auto expr) {
                            static constexpr int kNSplits = decltype(expr)::value;
                            if constexpr (kNSplits > 1) {
                                launch_tree_attn_stage2<kHeadDim, kNHeads, kNSplits>(stage2_params, stream);
                            }
                        },
                        [] {
                            TORCH_CHECK(false, "unsupported num_splits");
                        });
                },
                [] {
                    TORCH_CHECK(false, "unsupported num_heads");
                });
        },
        [] {
            TORCH_CHECK(false, "unsupported head_dim");
        });

    Stage3Params stage3_params{
        .outs_ptr = outs.data_ptr(),
        .lses_ptr = lses.data_ptr(),

        .kv_edge_metadata_ptr = nullptr,

        .num_edges = 0,
    };

    value_switch<int, 64, 128>(
        head_dim,
        [&](const auto expr) {
            static constexpr int kHeadDim = decltype(expr)::value;
            value_switch<int, 32>(
                num_heads,
                [&](const auto expr) {
                    static constexpr int kNHeads = decltype(expr)::value;
                    for (const auto& cur_edges : edges) {
                        // const auto& cur_edges = edges[i];
                        stage3_params.num_edges = cur_edges.size(0);
                        stage3_params.kv_edge_metadata_ptr = reinterpret_cast<Stage3Params::KVEdgeMetadata*>(cur_edges.data_ptr());
                        launch_tree_attn_stage3<kHeadDim, kNHeads>(stage3_params, stream);
                    }
                },
                [] {
                    TORCH_CHECK(false, "unsupported num_heads");
                });
        },
        [] {
            TORCH_CHECK(false, "unsupported head_dim");
        });

    return outs.reshape({num_active_pages * kBlockQ, num_heads, head_dim}).index_select(0, leaves);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "build_prefix_tree",
        &::build_prefix_tree,
        py::arg("seqlens_k"),
        py::arg("block_table"),
        py::arg("page_size"),
        py::arg("block_q"));

    m.def(
        "tree_attn",
        &::tree_attn_interface,
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        py::arg("pages"),
        py::arg("nodes"),
        py::arg("edges"),
        py::arg("leaves"),
        py::arg("max_splits"));
}
