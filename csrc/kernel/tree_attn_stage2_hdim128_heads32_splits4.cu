#include "launch.hpp"

template void launch_tree_attn_stage2<128, 32, 4>(const Stage2Params &params, cudaStream_t stream);