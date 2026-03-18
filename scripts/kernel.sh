#!/bin/bash

LOG_FILE="data/kernel_$(date +%Y%m%d_%H%M%S).log"

> "$LOG_FILE"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[1;34m'
NC='\033[0m' 

for var in "vary-seq" "vary-batch" "vary-depth" "vary-ratio" "vary-shape"
do
    echo -e "${BLUE}==================================================${NC}" | tee -a "$LOG_FILE"
    echo -e "${YELLOW}Variable: ${var}${NC}" | tee -a "$LOG_FILE"
    echo -e "${BLUE}==================================================${NC}" | tee -a "$LOG_FILE"

    echo -e "${GREEN}▶ Running ...${NC}" | tee -a "$LOG_FILE"
    python3 benchmark/kernel.py --scenario "$var" 2>&1 | tee -a "$LOG_FILE"

    echo -e "${BLUE}==================================================${NC}\n" | tee -a "$LOG_FILE"
done

