#include "launch.hpp"

template void launch_tree_attn_stage1<128, 32, 8, 2048>(const Stage1Params &params, cudaStream_t stream);