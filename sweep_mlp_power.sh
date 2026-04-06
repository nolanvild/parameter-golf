#!/bin/bash
# sweep_mlp_power.sh — MLP nonlinearity sweep: x^1 through x^5, 3 seeds each
# Run from the parameter-golf directory with the venv activated:
#   source .venv/bin/activate
#   bash sweep_mlp_power.sh

powers=(1 2 3 4 5)
seeds=(1337 42 123)

for power in "${powers[@]}"; do
    for seed in "${seeds[@]}"; do
        runId="power${power}_seed${seed}"
        echo "=== $runId ==="

        export RUN_ID="$runId"
        export MLP_POWER="$power"
        export SEED="$seed"
        export ITERATIONS="50"
        export TRAIN_BATCH_TOKENS="4096"
        export TRAIN_SEQ_LEN="256"
        export VAL_BATCH_SIZE="4096"
        export MAX_VAL_TOKENS="65536"
        export VAL_LOSS_EVERY="0"
        export WARMUP_STEPS="2"
        export WARMDOWN_ITERS="5"
        export TRAIN_LOG_EVERY="5"
        export MAX_WALLCLOCK_SECONDS="120"
        export NUM_LAYERS="3"
        export MODEL_DIM="128"
        export NUM_HEADS="4"
        export NUM_KV_HEADS="2"

        python train_gpt.py
    done
done

# --- Summary table ---
echo ""
echo "=== RESULTS ==="
printf "%-28s %10s %10s\n" "run" "val_bpb" "val_loss"
echo "----------------------------------------------------"

for power in "${powers[@]}"; do
    bpbs=()
    for seed in "${seeds[@]}"; do
        runId="power${power}_seed${seed}"
        logFile="logs/$runId.txt"
        if [[ -f "$logFile" ]]; then
            line=$(grep "final_int8_zlib_roundtrip val_loss" "$logFile" | tail -1)
            if [[ -n "$line" ]]; then
                bpb=$(echo "$line" | grep -oP 'val_bpb:\K[\d.]+')
                loss=$(echo "$line" | grep -oP 'val_loss:\K[\d.]+')
                printf "%-28s %10s %10s\n" "$runId" "$bpb" "$loss"
                [[ -n "$bpb" ]] && bpbs+=("$bpb")
            fi
        else
            printf "%-28s %10s\n" "$runId" "MISSING"
        fi
    done
    if [[ ${#bpbs[@]} -gt 0 ]]; then
        avg=$(printf '%s\n' "${bpbs[@]}" | awk '{s+=$1} END {printf "%.4f", s/NR}')
        printf "  power=%s avg (%d/3)          %10s\n" "$power" "${#bpbs[@]}" "$avg"
    fi
    echo ""
done
