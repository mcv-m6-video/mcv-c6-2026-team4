#!/bin/bash

# Preliminary hyperparameter search for still model with Optuna
# Runs multiple parallel processes for both non-temporal and temporal studies

set -e

# Default parameters
N_PROCESSES=4
N_TRIALS=100
SCRIPT="still_model_hyperparameter_search.py"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--n-processes)
            N_PROCESSES="$2"
            shift 2
            ;;
        -t|--n-trials)
            N_TRIALS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -n, --n-processes NUM    Number of parallel processes (default: 4)"
            echo "  -t, --n-trials NUM       Number of trials per process (default: 100)"
            echo "  -h, --help               Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 -n 8 -t 50    # Run 8 parallel processes with 50 trials each"
            echo ""
            echo "This will run TWO studies sequentially:"
            echo "  1. Still model without temporal (study: still_no_temporal)"
            echo "  2. Still model with temporal (study: still_with_temporal)"
            echo ""
            echo "Each study runs N_PROCESSES in parallel, for a total of:"
            echo "  - N_PROCESSES × N_TRIALS trials per study"
            echo "  - 2 × N_PROCESSES × N_TRIALS trials in total"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Calculate totals
TRIALS_PER_STUDY=$((N_PROCESSES * N_TRIALS))
TOTAL_TRIALS=$((2 * TRIALS_PER_STUDY))

echo "=========================================="
echo "Still Model Hyperparameter Search"
echo "=========================================="
echo "Parallel processes per study: $N_PROCESSES"
echo "Trials per process: $N_TRIALS"
echo "Trials per study: $TRIALS_PER_STUDY"
echo "Total trials (both studies): $TOTAL_TRIALS"
echo "=========================================="
echo ""

# Function to run a study
run_study() {
    local STUDY_NAME=$1
    local USE_TEMPORAL=$2
    local DB_NAME=$3

    echo ""
    echo "=========================================="
    echo "Starting study: $STUDY_NAME"
    echo "Temporal detector: $USE_TEMPORAL"
    echo "Database: $DB_NAME"
    echo "=========================================="
    echo ""

    # Array to store background process PIDs
    PIDS=()

    # Launch N_PROCESSES in parallel
    for i in $(seq 1 $N_PROCESSES); do
        LOG_FILE="logs/${STUDY_NAME}_process_${i}.log"

        echo "Launching process $i/$N_PROCESSES (log: $LOG_FILE)"

        if [ "$USE_TEMPORAL" = "true" ]; then
            python $SCRIPT \
                --n-trials $N_TRIALS \
                --study-name $STUDY_NAME \
                --db-path $DB_NAME \
                --use-temporal \
                > "$LOG_FILE" 2>&1 &
        else
            python $SCRIPT \
                --n-trials $N_TRIALS \
                --study-name $STUDY_NAME \
                --db-path $DB_NAME \
                > "$LOG_FILE" 2>&1 &
        fi

        PIDS+=($!)

	sleep 5
    done

    echo ""
    echo "All processes launched. Waiting for completion..."
    echo "PIDs: ${PIDS[@]}"
    echo ""

    # Wait for all processes to complete
    for i in "${!PIDS[@]}"; do
        PID=${PIDS[$i]}
        PROCESS_NUM=$((i + 1))
        echo "Waiting for process $PROCESS_NUM (PID: $PID)..."
        wait $PID
        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 0 ]; then
            echo "  ✓ Process $PROCESS_NUM completed successfully"
        else
            echo "  ✗ Process $PROCESS_NUM failed with exit code $EXIT_CODE"
        fi
    done

    echo ""
    echo "=========================================="
    echo "Study '$STUDY_NAME' completed!"
    echo "=========================================="
}

# Create logs directory if it doesn't exist
mkdir -p logs

# Study 1: Without temporal
run_study "still_no_temporal" "false" "optuna_still_no_temporal.db"

# Study 2: With temporal
run_study "still_with_temporal" "true" "optuna_still_with_temporal.db"

# Final summary
echo ""
echo "=========================================="
echo "ALL STUDIES COMPLETED!"
echo "=========================================="
echo ""
echo "Results:"
echo "  - Still (no temporal): optuna_still_no_temporal.db"
echo "  - Still (with temporal): optuna_still_with_temporal.db"
echo ""
echo "To visualize results:"
echo "  optuna-dashboard sqlite:///optuna_still_no_temporal.db"
echo "  optuna-dashboard sqlite:///optuna_still_with_temporal.db"
echo ""
echo "Logs saved in: logs/"
echo "=========================================="
