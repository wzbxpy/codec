#include "launch.hpp"

template void launch_tree_attn_stage1<64, 32, 8, 128>(const Stage1Params &params, cudaStream_t stream);