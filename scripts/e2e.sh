#!/bin/bash

# download model
huggingface-cli download --resume-download Qwen/Qwen3-4B-Instruct-2507 --local-dir ~/huggingface/Qwen3-4B/ --local-dir-use-symlinks False

LOG_FILE="data/e2e_$(date +%Y%m%d_%H%M%S).log"

> "$LOG_FILE"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[1;34m'
NC='\033[0m' 

for dataset in "longdep_qa" "shortdep_qa" "shortdep_cloze"
do
    echo -e "${BLUE}==================================================${NC}" | tee -a "$LOG_FILE"
    echo -e "${YELLOW}Dataset: ${dataset}${NC}" | tee -a "$LOG_FILE"
    echo -e "${BLUE}==================================================${NC}" | tee -a "$LOG_FILE"

    echo -e "${GREEN}▶ Running with vllm...${NC}" | tee -a "$LOG_FILE"
    DATASET=${dataset} LLM_BACKEND=vllm python3 benchmark/run_e2e.py --dataset "$dataset" 2>&1 | tee -a "$LOG_FILE"

    echo -e "${GREEN}▶ Running with codec...${NC}" | tee -a "$LOG_FILE"
    DATASET=${dataset} LLM_BACKEND=codec python3 benchmark/run_e2e.py --dataset "$dataset" 2>&1 | tee -a "$LOG_FILE"

    echo -e "${BLUE}==================================================${NC}\n" | tee -a "$LOG_FILE"
done

