#include "launch.hpp"

template void launch_tree_attn_stage2<128, 32, 128>(const Stage2Params &params, cudaStream_t stream);