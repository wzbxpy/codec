#include "launch.hpp"

template void launch_tree_attn_stage2<64, 32, 8>(const Stage2Params &params, cudaStream_t stream);