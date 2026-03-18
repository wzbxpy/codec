from pathlib import Path

output_dir = Path(__file__).parent

# Stage1 Kernel
for head_dim in [64, 128]:
    for num_heads, num_heads_kv in [(32, 8)]:
        for page_size in [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
            file_name = f"tree_attn_stage1_hdim{head_dim}_heads{num_heads}_kvheads{num_heads_kv}_pgsize{page_size}.cu"
            file_content = f'#include "launch.hpp"\n\ntemplate void launch_tree_attn_stage1<{head_dim}, {num_heads}, {num_heads_kv}, {page_size}>(const Stage1Params &params, cudaStream_t stream);'
            (output_dir / file_name).write_text(file_content)

# Stage2 Kernel
for head_dim in [64, 128]:
    for num_heads in [32]:
        for num_splits in [2, 4, 8, 16, 32, 64, 128]:
            file_name = f"tree_attn_stage2_hdim{head_dim}_heads{num_heads}_splits{num_splits}.cu"
            file_content = f'#include "launch.hpp"\n\ntemplate void launch_tree_attn_stage2<{head_dim}, {num_heads}, {num_splits}>(const Stage2Params &params, cudaStream_t stream);'
            (output_dir / file_name).write_text(file_content)

# Stage3 Kernel
for head_dim in [64, 128]:
    for num_heads in [32]:
        file_name = f"tree_attn_stage3_hdim{head_dim}_heads{num_heads}.cu"
        file_content = f'#include "launch.hpp"\n\ntemplate void launch_tree_attn_stage3<{head_dim}, {num_heads}>(const Stage3Params &params, cudaStream_t stream);'
        (output_dir / file_name).write_text(file_content)
