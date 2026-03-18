#include "launch.hpp"

template void launch_tree_attn_stage1<128, 32, 8, 32>(const Stage1Params &params, cudaStream_t stream);