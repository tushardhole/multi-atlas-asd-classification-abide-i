#!/bin/bash
# =====================================================
# run.sh — Launch Multi-Atlas GNN Experiments with nohup
# =====================================================

# --- Create logs directory if it doesn't exist ---
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

# --- Generate timestamp for this run ---
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/run_$TIMESTAMP.log"

# --- Print info to terminal ---
echo "Starting GNN experiment..."
echo "Log file: $LOG_FILE"
echo "Experiment started at $(date)"
echo "--------------------------------------"

# --- Run main.py with nohup in background ---
nohup python3 main.py > "$LOG_FILE" 2>&1 &

# --- Print PID ---
echo "Process is running in background with PID: $!"
echo "Use 'tail -f $LOG_FILE' to monitor progress."
