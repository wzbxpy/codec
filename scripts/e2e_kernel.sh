#!/bin/bash

# download model
huggingface-cli download --resume-download Qwen/Qwen3-4B-Instruct-2507 --local-dir ~/huggingface/Qwen3-4B/ --local-dir-use-symlinks False

LOG_FILE="data/e2e_kernel_$(date +%Y%m%d_%H%M%S).log"

> "$LOG_FILE"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[1;34m'
NC='\033[0m' 

for var in "vary-seq" "vary-batch" "vary-depth" "vary-ratio" "vary-shape"
do
    for index in 0 1 2 3 4
    do
        echo -e "${BLUE}==================================================${NC}" | tee -a "$LOG_FILE"
        echo -e "${YELLOW}Variable: ${var} Index: ${index} ${NC}"  | tee -a "$LOG_FILE"
        echo -e "${BLUE}==================================================${NC}" | tee -a "$LOG_FILE"

        echo -e "${GREEN}▶ Running with vllm...${NC}" | tee -a "$LOG_FILE"
        SCENE=${var} INDEX=${index} LLM_BACKEND=vllm python3 benchmark/run_e2e.py --dataset "$dataset" 2>&1 | tee -a "$LOG_FILE"

        echo -e "${GREEN}▶ Running with codec...${NC}" | tee -a "$LOG_FILE"
        SCENE=${var} INDEX=${index} LLM_BACKEND=codec python3 benchmark/run_e2e.py --dataset "$dataset" 2>&1 | tee -a "$LOG_FILE"

        echo -e "${BLUE}==================================================${NC}\n" | tee -a "$LOG_FILE"
    done
done
