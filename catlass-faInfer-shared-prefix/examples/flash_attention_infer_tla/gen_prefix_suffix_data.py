#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os
import sys
import logging
import numpy as np
import random
import ml_dtypes
from ml_dtypes import bfloat16
from dataclasses import dataclass
np.random.seed(1)

WORKSPACE = os.path.dirname(os.path.abspath(__file__))

def gen_seqlen(max_q_seqlen: int, max_kv_seqlen: int, is_varied_len: int, batch: int):
    q_seqlen_list = []
    kv_seqlen_list = []
    if is_varied_len == 0:
        q_seqlen_list = [max_q_seqlen] * batch
        kv_seqlen_list = [max_kv_seqlen] * batch
    else:
        for i in range(batch):
            q_seq = random.randint(1, max_q_seqlen)
            # kv_seq = random.randint(q_seq, max_kv_seqlen)
            kv_seq = 256 + q_seq
            q_seqlen_list.append(q_seq)
            kv_seqlen_list.append(kv_seq)
    print(q_seqlen_list)
    print(kv_seqlen_list)
    return q_seqlen_list, kv_seqlen_list

class TestFlashAttentionInfer():

    @dataclass
    class AttentionInputs:
        query: any
        key_cache: any
        value_cache: any
        block_tables: any
        q_seqlen_list: any
        k_seqlen_list: any
        global_mask: any
        mask_type: any
        inner_prec: int
        shape_param: any

    @dataclass
    class GenDataParams:
        q_seqlen_list: list
        k_seqlen_list: list
        num_heads: int
        kv_heads: int
        head_size: int
        num_blocks: int
        block_size: int
        mask_type: int
        dtype: any
        kv_dtype: int
        layout_dtype: int
        inner_prec: int
        max_q_seqlen: int
        max_kv_seqlen: int

    @classmethod
    def check_attr(cls, batch: int, q_seqlen: int, kv_seqlen: int, num_blocks: int, block_size: int):
        if q_seqlen > kv_seqlen:
            logging("[ERROR] q_seqlen cannot exceed kv_seqlen.")
            sys.exit()

    @classmethod
    def group_matmul(cls, head, kv_head, left, right):
        group_num = head // kv_head
        score = None
        for i in range(kv_head):
            group_score = np.matmul(left[i * group_num:(i + 1) * group_num, :, :].astype(np.float32),
                                    right[i:(i + 1), :, :].astype(np.float32))
            if score is None:
                score = group_score
            else:
                score = np.concatenate((score, group_score), 0)
        return score

    @classmethod
    def softmax_numpy(cls, sim):
        row_max = np.max(sim, axis=-1, keepdims=True)
        sim_sub = sim - row_max # x - m(x)

        sim_sub = np.exp(sim_sub) # e^{x - m(x)}
        row_sum = np.sum(sim_sub, axis=-1, keepdims=True) # (num_heads, qSeqlen, 1)

        soft_res = sim_sub / row_sum
        lse = np.squeeze((np.log(row_sum) + row_max), axis=-1)

        return soft_res, lse, row_max, sim_sub

    def softmax1(
        self,
        qk_result,
        is_first,
        gm,
        data_type = np.float16
    ):
        sim = qk_result
        lm = np.max(sim, axis=-1, keepdims=True)
        if is_first:
            hm = lm
            dm = 0
        else:
            hm = np.maximum(gm, lm)
            dm = gm - hm
        gm = hm
        sim_sub = sim - hm
        sim_sub = np.exp(sim_sub.astype(np.float32))
        row_sum = np.sum(sim_sub, axis=-1, keepdims=True)
        return sim_sub, row_sum, dm, gm


    def qkMM1(
        self,
        query,
        key
    ):
        result = None
        qk_k = key.shape[1]
        for qk_k_split in range(0, qk_k, 128):
            sub_k = 128
            if qk_k_split == 512:
                sub_k = 64
            query_k = query[:, :, qk_k_split: qk_k_split + sub_k]
            key_k = key[:, qk_k_split: qk_k_split + sub_k, :]
            result_split = self.group_matmul(query_k.shape[0], key_k.shape[0], query_k, key_k)
            if result is None:
                result = result_split
            else:
                result = result + result_split
        return result

    def ref_flash_attention(
        self,
        query,
        key,
        value,
        scale,
        mask,
        attention_inputs: AttentionInputs
    ):
        data_type = attention_inputs.shape_param.dtype
        query = np.transpose(query, (1, 0, 2)) # (num_heads, q_seqlen, head_size)
        key = np.transpose(key, (1, 2, 0)) # (kv_heads, head_size, k_seqlen)
        value = np.transpose(value, (1, 0, 2)) # (kv_heads, k_seqlen, head_size)
        context_len = key.shape[2]
        context_size = 512
        group_num = query.shape[0] // key.shape[0]
        gl = None
        gl_high = None
        go = None
        go_high = None

        # 存储累积的exp行和
        cumulative_exp_sums = None

        for kv_start in range(0, context_len, context_size):
            sub_len = context_size
            if kv_start + context_size > context_len:
                sub_len = context_len - kv_start
                print(sub_len)
            sub_key = key[:, :, kv_start: kv_start + sub_len]
            sub_mask = None
            if mask is not None:
                sub_mask = mask[:query.shape[1], kv_start: kv_start + sub_len]
            sub_value = value[:, kv_start: kv_start + sub_len, :]
            qk_result = self.qkMM1(query, sub_key).astype(np.float32)
            qk_result_high = self.qkMM1(query.astype(np.float32), sub_key.astype(np.float32))
            qk_result = qk_result * scale
            qk_result_high = qk_result_high * scale

            if mask is not None:
                qk_result += sub_mask
                qk_result_high += sub_mask.astype(np.float32)
            if kv_start == 0:
                gm = None
            p_result, row_sum, dm, gm = self.softmax1(qk_result, kv_start == 0, gm)
            p_result = p_result.astype(data_type)
            if kv_start == 0:
                gm_high = None
            p_result_high, row_sum_high, dm_high, gm_high = self.softmax1(qk_result_high, kv_start == 0, gm_high)
            lo = self.group_matmul(p_result.shape[0], sub_value.shape[0], p_result, sub_value)
            lo_high = self.group_matmul(p_result.shape[0], sub_value.shape[0], p_result.astype(np.float32), sub_value.astype(np.float32))
            if kv_start == 0:
                gl = row_sum
                gl_high = row_sum_high
                go = lo
                go_high = lo_high
                cumulative_exp_sums = row_sum
            else:
                dm = np.exp(dm)
                dm_high = np.exp(dm_high)
                gl = gl * dm
                gl = gl + row_sum

                go = go * dm
                go = go + lo

                gl_high = gl_high * dm_high
                gl_high = gl_high + row_sum_high

                go_high = go_high * dm_high
                go_high = go_high + lo_high
                # 累积exp行和
                cumulative_exp_sums = cumulative_exp_sums * dm + row_sum
        go = go / gl
        go_high = go_high / gl_high
        go = np.transpose(go, (1, 0, 2))
        go_high = np.transpose(go_high, (1, 0, 2))
        lse = np.squeeze((np.log(gl) + gm), axis=-1)
        lse_high = np.squeeze((np.log(gl_high) + gm_high), axis=-1)

        # 处理exp行和的shape
        exp_sums_high = np.squeeze(cumulative_exp_sums, axis=-1) # (num_heads, q_seqlen)
        # exp_sums_high = np.transpose(exp_sums_high, (1, 0)) # (q_seqlen, num_heads)

        return go.astype(data_type), go_high, lse, lse_high

    def ref_masked_attention(self,
            query,  # (q_seqlen, num_heads, head_size)
            key,    # (k_seqlen, kv_heads, head_size)
            value,
            scale: float,
            mask    # (q_seqlen, k_seqlen)
    ):
        query = np.transpose(query, (1, 0, 2))
        key = np.transpose(key, (1, 2, 0))
        sim_high = self.group_matmul(query.shape[0], key.shape[0], query, key)  # (head_num, q_seqlen, k_seqlen)
        sim_low_prec = sim_high.astype(np.float16) * np.float16(scale)
        sim_high = sim_high * scale
        pre_mask_factor = -10000
        if gen_data_params.dtype is ml_dtypes.bfloat16:
            pre_mask_factor = -3e38
        if mask is not None:
            sim_high = sim_high + (
                mask[:sim_high.shape[-2], :sim_high.shape[-1]]
                ).astype(np.float32) * pre_mask_factor
            sim_low_prec = sim_low_prec + (
                mask[:sim_high.shape[-2], :sim_high.shape[-1]]
                ).astype(np.float16) * -10000
        p_high, lse_high, gm, exp_high = self.softmax_numpy(sim_high)
        p_low_prec, lse_low_prec, gm_low_prec, exp_low = self.softmax_numpy(sim_low_prec)
        lse = lse_high.astype(query.dtype)
        lse_high = lse_high.astype(np.float32)
        p = p_high.astype(query.dtype)
        p_high = p_high.astype(np.float32)
        value = np.transpose(value, (1, 0, 2)) # (kv_heads, k_seqlen, head_size)

        out_low_prec = self.group_matmul(query.shape[0], key.shape[0], p_low_prec, value)
        out_high = self.group_matmul(query.shape[0], key.shape[0], p_high, value)
        out = self.group_matmul(query.shape[0], key.shape[0], p, value)
        out_low_prec = np.transpose(out_low_prec, (1, 0, 2)) # (q_seqlen, head_num, head_size)
        out_high = np.transpose(out_high, (1, 0, 2))
        out = np.transpose(out, (1, 0, 2))
        out_low_prec = out_low_prec.astype(np.float16)
        out = out.astype(query.dtype)
        # 计算exp行和，并转置为最终输出shape
        exp_sums_high = np.sum(exp_high, axis=-1) # (num_heads, q_seqlen)
        #
        return out, out_high, out_low_prec, lse, lse_high, gm, exp_sums_high

    def ref_single_query_cached_kv_attention(self, attention_inputs: AttentionInputs, output, golden_gpu_output, golden_lse_output, golden_gpu_lse_output) -> None:
        num_heads = attention_inputs.shape_param.num_heads
        kv_heads = attention_inputs.shape_param.kv_heads
        head_size_qk = attention_inputs.shape_param.head_size
        head_size_vo = attention_inputs.shape_param.head_size
        block_size = attention_inputs.shape_param.block_size
        max_q_seqlen = attention_inputs.shape_param.max_q_seqlen
        inner_prec = attention_inputs.inner_prec

        batch = len(attention_inputs.shape_param.q_seqlen_list)
        cu_seqlen = 0
        kv_seqlen_now = 0

        # 存储exp行和
        golden_exp_sums = None
        golden_gpu_exp_sums = None

        for i in range(batch):
            q_seqlen = int(attention_inputs.q_seqlen_list[i])
            k_seqlen = int(attention_inputs.k_seqlen_list[i])
            q = None
            if attention_inputs.shape_param.layout_dtype == 1: # 连续布局
                q = attention_inputs.query[cu_seqlen:(cu_seqlen + q_seqlen), :, :]
            else: # 批次布局
                q = attention_inputs.query[i * max_q_seqlen:(i * max_q_seqlen + q_seqlen), :, :]
            keys = None
            values = None
            if attention_inputs.shape_param.kv_dtype == 1: # CacheMode分块
                keys = []
                values = []
                block_table = attention_inputs.block_tables[i]
                for j in range(k_seqlen):
                    block_number = int(block_table[j // block_size]) # 计算块号
                    block_offset = j % block_size # 块内偏移

                    k = attention_inputs.key_cache[block_number, block_offset, :, :]
                    k = k.reshape(kv_heads, head_size_qk)
                    keys.append(k)

                    v = attention_inputs.value_cache[block_number, block_offset, :, :]
                    v = v.reshape(kv_heads, head_size_vo)
                    values.append(v)
                keys = np.stack(keys, axis=0)
                values = np.stack(values, axis=0)
            elif attention_inputs.shape_param.kv_dtype == 0:
                if attention_inputs.shape_param.layout_dtype == 1:
                    keys = attention_inputs.key_cache[kv_seqlen_now: kv_seqlen_now + k_seqlen, :, :]
                    values = attention_inputs.value_cache[kv_seqlen_now: kv_seqlen_now + k_seqlen, :, :]
                else:
                    keys = attention_inputs.key_cache[i, :, :, :]
                    values = attention_inputs.value_cache[i, :, :, :]
            scale = 1.0 / (head_size_qk ** 0.5)
            if attention_inputs.mask_type == 1:
                mask = attention_inputs.global_mask[cu_seqlen:(cu_seqlen + q_seqlen), :]
            elif attention_inputs.mask_type == 2:
                mask = attention_inputs.global_mask
            else:
                mask = None
            out_normal, _,  out_low_prec, lse, _, gm, exp_sums = self.ref_masked_attention(q, keys, values, scale, mask)
            out_high, _, lse_high, _, exp_sums_high = self.ref_flash_attention(q, keys, values, scale, mask, attention_inputs)
            out = None
            if inner_prec == 0:
                out = out_normal.reshape(-1, num_heads, head_size_vo)
            else:
                out = out_low_prec.reshape(-1, num_heads, head_size_vo)
            out_high = out_high.reshape(-1, num_heads, head_size_vo)

            if attention_inputs.shape_param.layout_dtype == 1:
                output[cu_seqlen: cu_seqlen + q_seqlen, :, :] = out
                golden_gpu_output[cu_seqlen: cu_seqlen + q_seqlen, :, :] = out_high

                golden_lse_output[:, cu_seqlen: cu_seqlen + q_seqlen] = lse
                golden_gpu_lse_output[:, cu_seqlen: cu_seqlen + q_seqlen] = lse_high
                # 存储exp行和
                if golden_exp_sums is None:
                    golden_exp_sums = exp_sums # (num_heads, q_seqlen)
                    golden_gpu_exp_sums = exp_sums_high
                else:
                    golden_exp_sums = np.concatenate([golden_exp_sums, exp_sums], axis=1) # (num_heads, q_seqlen)
                    golden_gpu_exp_sums = np.concatenate([golden_gpu_exp_sums, exp_sums_high], axis=1)
            else:
                output[i * max_q_seqlen: i * max_q_seqlen + q_seqlen, :, :] = out
                golden_gpu_output[i * max_q_seqlen: i * max_q_seqlen + q_seqlen, :, :] = out_high

                golden_lse_output[:, i * max_q_seqlen: i * max_q_seqlen + q_seqlen] = lse
                golden_gpu_lse_output[:, i * max_q_seqlen: i * max_q_seqlen + q_seqlen] = lse_high

                # 存储exp行和
                if golden_exp_sums is None:
                    golden_exp_sums = exp_sums
                    golden_gpu_exp_sums = exp_sums_high
                else:
                    # 批次布局特殊处理
                    start_idx = i * max_q_seqlen
                    end_idx = start_idx + q_seqlen
                    golden_exp_sums[:, start_idx:end_idx] = exp_sums
                    golden_gpu_exp_sums[:, start_idx:end_idx] = exp_sums_high

            cu_seqlen += q_seqlen
            kv_seqlen_now += k_seqlen
        return golden_exp_sums, golden_gpu_exp_sums

    def calc_data(self, gen_data_params: GenDataParams):
        head_size_qk = gen_data_params.head_size # Embedding_size
        head_size_vo = gen_data_params.head_size # Embedding_size
        q_min_range = -1.0
        q_max_range = 1.0
        kv_min_range = -1.0
        kv_max_range = 1.0
        num_tokens = np.array(gen_data_params.q_seqlen_list).sum()
        num_kv_tokens = np.array(gen_data_params.k_seqlen_list).sum()
        num_kv_tokens_prefix = num_kv_tokens - num_tokens
        num_kv_tokens_suffix = num_tokens
        batch_size = len(gen_data_params.q_seqlen_list)
        # 生成所有token完整query embedding
        query = np.random.uniform(q_min_range, q_max_range,
            size=(num_tokens, gen_data_params.num_heads, head_size_qk)).astype(gen_data_params.dtype)
        #
        #
        #
        #
        #
        #
        max_k_seqlen = gen_data_params.max_kv_seqlen
        max_q_seqlen = max(gen_data_params.q_seqlen_list)
        block_tables_prefix = []
        block_tables_prefix_golden = []
        block_tables_suffix = []
        key_cache_prefix = None
        value_cache_prefix = None
        key_cache_suffix = None
        value_cache_suffix = None
        # CacheMode=1 ->分块管理
        if gen_data_params.kv_dtype == 1:

            # 计算shared prefix的长度
            shared_prefix_len = max_k_seqlen - max_q_seqlen # 只对不变长度正确
            # 计算shared prefix占用的完整block数
            shared_prefix_blocks = (shared_prefix_len + gen_data_params.block_size - 1) // gen_data_params.block_size
            unique_suffix_blocks = (max_q_seqlen + gen_data_params.block_size - 1) // gen_data_params.block_size
            total_blocks_prefix = batch_size * shared_prefix_blocks
            total_blocks_suffix = batch_size * unique_suffix_blocks

            # 计算每个block中shared prefix的token数
            shared_tokens_per_block = []
            for i in range(shared_prefix_blocks):
                if i < shared_prefix_blocks - 1:
                    shared_tokens_per_block.append(gen_data_params.block_size)
                else:
                    # 最后一个block可能不满
                    shared_tokens_per_block.append(shared_prefix_len % gen_data_params.block_size)

            # 初始化key cache
            # key_cache_prefix = np.zeros((total_blocks_prefix, gen_data_params.block_size,
            #                     gen_data_params.kv_heads, head_size_qk),
            #                     dtype=gen_data_params.dtype)

            # value_cache_prefix = np.zeros((total_blocks_prefix, gen_data_params.block_size,
            #                     gen_data_params.kv_heads, head_size_qk),
            #                     dtype=gen_data_params.dtype)
            key_cache_prefix = np.zeros((shared_prefix_blocks, gen_data_params.block_size,
                                gen_data_params.kv_heads, head_size_qk),
                                dtype=gen_data_params.dtype)

            value_cache_prefix = np.zeros((shared_prefix_blocks, gen_data_params.block_size,
                                gen_data_params.kv_heads, head_size_qk),
                                dtype=gen_data_params.dtype)

            key_cache_prefix_golden = np.zeros((total_blocks_prefix, gen_data_params.block_size,
                                gen_data_params.kv_heads, head_size_qk),
                                dtype=gen_data_params.dtype)

            value_cache_prefix_golden = np.zeros((total_blocks_prefix, gen_data_params.block_size,
                                gen_data_params.kv_heads, head_size_qk),
                                dtype=gen_data_params.dtype)

            key_cache_suffix = np.random.uniform(
                kv_min_range, kv_max_range,
                size=(total_blocks_suffix, gen_data_params.block_size,
                      gen_data_params.kv_heads, head_size_qk)
            ).astype(gen_data_params.dtype)

            value_cache_suffix = np.random.uniform(
                kv_min_range, kv_max_range,
                size=(total_blocks_suffix, gen_data_params.block_size,
                      gen_data_params.kv_heads, head_size_qk)
            ).astype(gen_data_params.dtype)

            # 生成共享前缀数据
            shared_prefix_data_key = np.random.uniform(
                kv_min_range, kv_max_range,
                size=(shared_prefix_blocks, gen_data_params.block_size,
                      gen_data_params.kv_heads, head_size_qk)
            ).astype(gen_data_params.dtype)

            shared_prefix_data_value = np.random.uniform(
                kv_min_range, kv_max_range,
                size=(shared_prefix_blocks, gen_data_params.block_size,
                      gen_data_params.kv_heads, head_size_qk)
            ).astype(gen_data_params.dtype)

            # 为每个batch填充数据
            for batch_idx in range(1):
                block_offset = batch_idx * shared_prefix_blocks

                # 填充shared prefix部分
                for i in range(shared_prefix_blocks):
                    block_idx = block_offset + i
                    actual_tokens = shared_tokens_per_block[i]
                    # 只填充实际token位置
                    key_cache_prefix[block_idx, :actual_tokens] = shared_prefix_data_key[i, :actual_tokens]
                    value_cache_prefix[block_idx, :actual_tokens] = shared_prefix_data_value[i, :actual_tokens]


            for batch_idx in range(batch_size):
                block_offset = batch_idx * shared_prefix_blocks

                # 填充shared prefix部分
                for i in range(shared_prefix_blocks):
                    block_idx = block_offset + i
                    actual_tokens = shared_tokens_per_block[i]
                    # 只填充实际token位置
                    key_cache_prefix_golden[block_idx, :actual_tokens] = shared_prefix_data_key[i, :actual_tokens]
                    value_cache_prefix_golden[block_idx, :actual_tokens] = shared_prefix_data_value[i, :actual_tokens]

            block_table = [j for j in range(shared_prefix_blocks)]
            block_tables_prefix.append(block_table)

            for i in range(batch_size):
                block_table = [
                    shared_prefix_blocks * i + j
                    for j in range(shared_prefix_blocks)
                ]
                block_tables_prefix_golden.append(block_table)

            for i in range(batch_size):
                block_table = [
                    unique_suffix_blocks * i + j
                    for j in range(unique_suffix_blocks)
                ]
                block_tables_suffix.append(block_table)

        elif gen_data_params.kv_dtype == 0:
            assert("Not Support Yet")
            if gen_data_params.layout_dtype == 1: # TND
                key_cache = np.random.uniform(kv_min_range, kv_max_range,
                    size=(num_kv_tokens, gen_data_params.kv_heads, head_size_qk)).astype(gen_data_params.dtype)
                value_cache = np.random.uniform(kv_min_range, kv_max_range,
                    size=(num_kv_tokens, gen_data_params.kv_heads, head_size_vo)).astype(gen_data_params.dtype)
            elif gen_data_params.layout_dtype == 0: # BSND
                key_cache = np.random.uniform(kv_min_range, kv_max_range,
                    size=(batch_size, max_k_seqlen, gen_data_params.kv_heads, head_size_qk)).astype(gen_data_params.dtype)
                value_cache = np.random.uniform(kv_min_range, kv_max_range,
                    size=(batch_size, max_k_seqlen, gen_data_params.kv_heads, head_size_vo)).astype(gen_data_params.dtype)
        if gen_data_params.mask_type == 1:
            mask = np.zeros(shape=(num_tokens, max_q_seqlen)).astype(gen_data_params.dtype)
            pre_qseqlen = 0
            for i in range(batch_size):
                qseqlen = gen_data_params.q_seqlen_list[i]
                tri = np.ones((qseqlen, qseqlen))
                tri = np.triu(tri, 1)
                mask[pre_qseqlen : (pre_qseqlen + qseqlen), 0 : qseqlen] = tri
                pre_qseqlen += qseqlen
            mask = mask.astype(gen_data_params.dtype)
        elif gen_data_params.mask_type == 2:
            mask = np.ones(shape=(max_q_seqlen, max_k_seqlen)).astype(np.float16)
            mask = np.triu(mask, 1)
        elif gen_data_params.mask_type == 0:
            mask = None

        shape_out = (num_tokens, gen_data_params.num_heads, head_size_vo)
        golden_output_prefix = np.zeros(shape_out, dtype=gen_data_params.dtype)
        golden_gpu_output_prefix = np.zeros(shape_out, dtype=np.float32)

        golden_output_suffix = np.zeros(shape_out, dtype=gen_data_params.dtype)
        golden_gpu_output_suffix = np.zeros(shape_out, dtype=np.float32)

        lse_shape_out = (gen_data_params.num_heads, num_tokens)
        golden_lse_output_prefix = np.zeros(lse_shape_out, dtype=gen_data_params.dtype)
        golden_gpu_lse_output_prefix = np.zeros(lse_shape_out, dtype=np.float32)

        golden_lse_output_suffix = np.zeros(lse_shape_out, dtype=gen_data_params.dtype)
        golden_gpu_lse_output_suffix = np.zeros(lse_shape_out, dtype=np.float32)

        prefix_kseqlen_list = [shared_prefix_len for _ in range(batch_size)]
        suffix_kseqlen_list = [max_q_seqlen for _ in range(batch_size)]
        attention_inputs_prefix = self.AttentionInputs(query, key_cache_prefix_golden, value_cache_prefix_golden, block_tables_prefix_golden,
            gen_data_params.q_seqlen_list, prefix_kseqlen_list, mask, 0, gen_data_params.inner_prec, gen_data_params)
        attention_inputs_suffix = self.AttentionInputs(query, key_cache_suffix, value_cache_suffix, block_tables_suffix,
            gen_data_params.q_seqlen_list, suffix_kseqlen_list, mask, gen_data_params.mask_type, gen_data_params.inner_prec, gen_data_params)

        # exp行和的shape (num_heads, q_seqlen)
        exp_sums_shape = (gen_data_params.num_heads, num_tokens)
        golden_exp_sums = np.zeros(exp_sums_shape, dtype=np.float32)
        golden_gpu_exp_sums = np.zeros(exp_sums_shape, dtype=np.float32)

        golden_exp_sums, golden_gpu_exp_sums = self.ref_single_query_cached_kv_attention(
            attention_inputs_prefix,
            golden_output_prefix,
            golden_gpu_output_prefix,
            golden_lse_output_prefix,
            golden_gpu_lse_output_prefix
        )
        # 保存exp行和数据
        golden_exp_sums.astype(np.float32).tofile(os.path.join(WORKSPACE, "data", "golden_prefix_rowsum.bin"))
        golden_gpu_exp_sums.astype(np.float32).tofile(os.path.join(WORKSPACE, "data", "golden_gpu_prefix_rowsum.bin"))

        golden_exp_sums, golden_gpu_exp_sums = self.ref_single_query_cached_kv_attention(
            attention_inputs_suffix,
            golden_output_suffix,
            golden_gpu_output_suffix,
            golden_lse_output_suffix,
            golden_gpu_lse_output_suffix
        )

        golden_exp_sums.astype(np.float32).tofile(os.path.join(WORKSPACE, "data", "golden_suffix_rowsum.bin"))
        golden_gpu_exp_sums.astype(np.float32).tofile(os.path.join(WORKSPACE, "data", "golden_gpu_suffix_rowsum.bin"))

        golden_lse_output_prefix = np.transpose(golden_lse_output_prefix, (1, 0))
        golden_lse_output_prefix = np.expand_dims(golden_lse_output_prefix, axis=2)

        golden_lse_output_suffix = np.transpose(golden_lse_output_suffix, (1, 0))
        golden_lse_output_suffix = np.expand_dims(golden_lse_output_suffix, axis=2)

        num_tokens.astype(np.int32).tofile(os.path.join(WORKSPACE, "data", "q_ntokens.bin"))
        num_kv_tokens_prefix.astype(np.int32).tofile(os.path.join(WORKSPACE, "data", "kv_ntokens_prefix.bin"))
        num_kv_tokens_suffix.astype(np.int32).tofile(os.path.join(WORKSPACE, "data", "kv_ntokens_suffix.bin"))
        query.tofile(os.path.join(WORKSPACE, "data", "q.bin"))
        key_cache_prefix.tofile(os.path.join(WORKSPACE, "data", "k_prefix.bin"))
        value_cache_prefix.tofile(os.path.join(WORKSPACE, "data", "v_prefix.bin"))
        key_cache_suffix.tofile(os.path.join(WORKSPACE, "data", "k_suffix.bin"))
        value_cache_suffix.tofile(os.path.join(WORKSPACE, "data", "v_suffix.bin"))
        np.array(block_tables_prefix).astype(np.int32).tofile(os.path.join(WORKSPACE, "data", "block_table_prefix.bin"))
        np.array(block_tables_suffix).astype(np.int32).tofile(os.path.join(WORKSPACE, "data", "block_table_suffix.bin"))

        np.array([num_tokens, num_kv_tokens]).astype(np.int32).tofile(os.path.join(WORKSPACE, "data", "dataext.bin"))

        if gen_data_params.layout_dtype == 0:
            new_q_seqlen_list = []
            pre_seq_sum = 0
            for i in range(batch):
                pre_seq_sum += gen_data_params.q_seqlen_list[i]
                new_q_seqlen_list.append(pre_seq_sum)
            gen_data_params.q_seqlen_list = new_q_seqlen_list
            if gen_data_params.kv_dtype == 0:
                new_kv_seqlen_list = []
                pre_seq_sum = 0
                for i in range(batch):
                    pre_seq_sum += gen_data_params.k_seqlen_list[i]
                    new_kv_seqlen_list.append(pre_seq_sum)
                gen_data_params.k_seqlen_list = new_kv_seqlen_list

        np.array(gen_data_params.q_seqlen_list).astype(np.int64).tofile(
            os.path.join(WORKSPACE, "data", "q_seqlen.bin"))
        np.array(prefix_kseqlen_list).astype(np.int64).tofile(
            os.path.join(WORKSPACE, "data", "kv_seqlen_prefix.bin"))
        np.array(gen_data_params.q_seqlen_list).astype(np.int64).tofile(
            os.path.join(WORKSPACE, "data", "kv_seqlen_suffix.bin"))
        if gen_data_params.mask_type == 1:
            actual_input_mask_triu = np.triu(np.ones((1024, 1024)), 1).astype(gen_data_params.dtype)
            actual_input_mask_triu.tofile(os.path.join(WORKSPACE, "data", "mask.bin"))
        print(gen_data_params.q_seqlen_list)
        print(prefix_kseqlen_list)
        print(suffix_kseqlen_list)

        golden_output_prefix.astype(np.float32).tofile(os.path.join(WORKSPACE, "data", "golden_prefix.bin"))
        golden_gpu_output_prefix.astype(np.float32).tofile(os.path.join(WORKSPACE, "data", "golden_gpu_prefix.bin"))
        golden_lse_output_prefix.astype(np.float32).tofile(os.path.join(WORKSPACE, "data", "golden_lse_prefix.bin"))
        golden_gpu_lse_output_prefix.astype(np.float32).tofile(os.path.join(WORKSPACE, "data", "golden_gpu_lse_prefix.bin"))

        golden_output_suffix.astype(np.float32).tofile(os.path.join(WORKSPACE, "data", "golden_suffix.bin"))
        golden_gpu_output_suffix.astype(np.float32).tofile(os.path.join(WORKSPACE, "data", "golden_gpu_suffix.bin"))
        golden_lse_output_suffix.astype(np.float32).tofile(os.path.join(WORKSPACE, "data", "golden_lse_suffix.bin"))
        golden_gpu_lse_output_suffix.astype(np.float32).tofile(os.path.join(WORKSPACE, "data", "golden_gpu_lse_suffix.bin"))

if __name__ == "__main__":

    os.makedirs(os.path.join(WORKSPACE, "data"), exist_ok=True)

    print("参数总数：", len(sys.argv) - 1)  # 减去脚本名本身
    print("所有参数：", sys.argv)

    batch = int(sys.argv[1])
    q_seqlen = int(sys.argv[2])
    kv_seqlen = int(sys.argv[3])
    num_head = int(sys.argv[4])
    kv_heads = int(sys.argv[5])
    embedding_size = int(sys.argv[6])
    block_size = 128
    is_varied_len = int(sys.argv[7])
    mask_type = int(sys.argv[8])

    str_dtype = str(sys.argv[9])
    if str_dtype == "half":
        dtype = np.float16
    elif str_dtype == "bf16":
        dtype = bfloat16
    else:
        logging("[ERROR] dtype must be half or bf16")
        sys.exit()

    kv_dtype = int(sys.argv[10])
    layout_dtype = 1
    inner_prec = 0
    q_seqlen_list, kv_seqlen_list = gen_seqlen(q_seqlen, kv_seqlen, is_varied_len, batch)

    max_kv_seqlen = max(kv_seqlen_list)
    num_blocks = batch * ((max_kv_seqlen + block_size - 1) // block_size)
    testObj = TestFlashAttentionInfer()
    gen_data_params = testObj.GenDataParams(q_seqlen_list, kv_seqlen_list, num_head,
                                            kv_heads, embedding_size,
                                            num_blocks, block_size, mask_type, dtype, kv_dtype, layout_dtype, inner_prec, q_seqlen, kv_seqlen)
    testObj.calc_data(gen_data_params)
