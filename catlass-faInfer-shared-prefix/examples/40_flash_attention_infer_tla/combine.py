import numpy as np

def read_binary_file(filename, dtype):
    with open(filename, 'rb') as f:
        data = np.fromfile(f, dtype=dtype)
    return data

o_host_prefix = read_binary_file('o_host_prefix_half.bin', np.float16)
o_host_suffix = read_binary_file('o_host_suffix_half.bin', np.float16)

# 读bf16数据
# o_host_prefix = read_binary_file('o_host_prefix_bf16.bin', np.bfloat16)
# o_host_suffix = read_binary_file('o_host_suffix_bf16.bin', np.bfloat16)

row_sum_prefix = read_binary_file('row_sum_host_prefix.bin', np.float32)
row_sum_suffix = read_binary_file('row_sum_host_suffix.bin', np.float32)

# o_host = read_binary_file('o_host.bin', np.float16)

print(f"O Prefix shape : {o_host_prefix.shape}, dtype: {o_host_prefix.dtype}")
print(f"O Suffix shape : {o_host_suffix.shape}, dtype: {o_host_suffix.dtype}")
print(f"RowSum Prefix shape: {row_sum_prefix.shape}, dtype: {row_sum_prefix.dtype}")
print(f"RowSum Suffix shape: {row_sum_suffix.shape}, dtype: {row_sum_suffix.dtype}")

# 后续处理
def merge_attention_outputs(o_prefix, o_suffix, sum_exp_prefix, sum_exp_suffix):
    """
    合并两个FA的输出
    """
    o_prefix_f32 = o_prefix.astype(np.float32)
    o_suffix_f32 = o_suffix.astype(np.float32)

    sum_exp_prefix_f32 = sum_exp_prefix.astype(np.float32)
    sum_exp_suffix_f32 = sum_exp_suffix.astype(np.float32)

    # 计算分母
    denom = sum_exp_prefix_f32 + sum_exp_suffix_f32

    v_merged = (o_prefix_f32 * sum_exp_prefix_f32[:, np.newaxis] +
                o_suffix_f32 * sum_exp_suffix_f32[:, np.newaxis]) / denom[:, np.newaxis]

    sum_exp_merged = denom

    merged_o = v_merged.astype(o_prefix.dtype)
    return merged_o

merged_o = merge_attention_outputs(
    o_host_prefix,
    o_host_suffix,
    row_sum_prefix,
    row_sum_suffix
)

print(f"Merged O Shape: {merged_o.shape}, dtype: {merged_o.dtype}")