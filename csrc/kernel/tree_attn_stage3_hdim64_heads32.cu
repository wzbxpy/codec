#include "launch.hpp"

template void launch_tree_attn_stage3<64, 32>(const Stage3Params &params, cudaStream_t stream);