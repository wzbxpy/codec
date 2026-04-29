#!/bin/bash
batch=1
qSeqlen=177
kvSeqlen=512
numHeads=4
kvHeads=2
headSize=128
isVariedLen=0
maskType=1
dtype="bf16"
cacheMode=1
layout_dtype=0
num_blocks=2048
inner_prec=0
lse_flag=0
device=0

function build() {
    rm -rf build
    rm -rf output
    bash scripts/build.sh flash_attention_infer_tla
}

function gen_data() {
    python3 examples/flash_attention_infer_tla/gen_prefix_suffix_data.py $batch $qSeqlen $kvSeqlen $numHeads $kvHeads $headSize $isVariedLen $maskType "$dtype" $cacheMode $layout_dtype $num_blocks $inner_prec $lse_flag
    echo "Data gen finished"
}

function run_kernel() {
    echo 'Case: B=' $batch ' qS=' $qSeqlen ' kvS=' $kvSeqlen ' qN=' $numHeads ' kvN=' $kvHeads ' D=' $headSize ' mask=' $maskType
    cd output/bin/
    ./flash_attention_infer_tla $batch $qSeqlen $kvSeqlen $numHeads $kvHeads $headSize $isVariedLen $maskType --device $device --dtype $dtype
}

build
gen_data
run_kernel