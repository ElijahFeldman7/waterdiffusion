#!/bin/bash

# Define arrays
CONFIG=(
    "configs/main/Llada/ourWatermark_llada8b_instruct.yaml"
)

NAME=(
    "Llada/OurWatermark"
)

DELTA=(1 2 2.5 3 4 5)
KERNEL=("[-1]")
TOPK=(100)
SEEDING_SCHEME=("sumhash")

# Iterate over configurations and names
for ((i=0; i<${#CONFIG[@]}; i++)); do
    config=${CONFIG[$i]}
    name=${NAME[$i]}

    for delta in "${DELTA[@]}"; do
        for kernel in "${KERNEL[@]}"; do
            for topk in "${TOPK[@]}"; do
                for seeding in "${SEEDING_SCHEME[@]}"; do
                    echo "Running with delta=$delta, kernel=$kernel, topk=$topk, seeding=$seeding"
                    python scripts/ablate_watermark.py --delta $delta --kernel $kernel --topk $topk --config $config --name $name --seeding_scheme $seeding --num_samples 250
                done
            done
        done
    done
done