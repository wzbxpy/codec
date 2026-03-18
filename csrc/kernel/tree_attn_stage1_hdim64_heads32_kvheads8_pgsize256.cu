#include "launch.hpp"

template void launch_tree_attn_stage1<64, 32, 8, 256>(const Stage1Params &params, cudaStream_t stream);