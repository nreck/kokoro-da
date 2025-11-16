#!/bin/bash

# Monitor GPU usage during multi-GPU training

echo "Monitoring GPU usage (Ctrl+C to stop)..."
echo "========================================="

# Function to get GPU stats
get_gpu_stats() {
    nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits
}

# Function to format size
format_size() {
    echo "$1" | awk '{ printf "%.1f GB", $1/1024 }'
}

# Header
printf "%-5s %-20s %-6s %-7s %-7s %-20s\n" "GPU" "Name" "Temp" "Util%" "Mem%" "Memory Used/Total"
echo "--------------------------------------------------------------------"

# Continuous monitoring
while true; do
    # Clear previous lines (4 GPUs + header + separator)
    tput cuu 6

    # Print header again
    printf "%-5s %-20s %-6s %-7s %-7s %-20s\n" "GPU" "Name" "Temp" "Util%" "Mem%" "Memory Used/Total"
    echo "--------------------------------------------------------------------"

    # Get and display stats
    while IFS=, read -r idx name temp gpu_util mem_util mem_used mem_total; do
        mem_used_gb=$(format_size $mem_used)
        mem_total_gb=$(format_size $mem_total)

        printf "%-5s %-20s %-6s°C %-7s%% %-7s%% %-20s\n" \
            "$idx" \
            "${name// /}" \
            "$temp" \
            "$gpu_util" \
            "$mem_util" \
            "$mem_used_gb / $mem_total_gb"
    done <<< "$(get_gpu_stats)"

    sleep 2
done