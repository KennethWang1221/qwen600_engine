#!/bin/bash
# benchmark.sh
# Comprehensive benchmarking script for Qwen3-0.6B engine

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

MODEL_DIR="${1:-Qwen3-0.6B}"
EXECUTABLE="${2:-./qwen600_engine}"

if [ ! -d "$MODEL_DIR" ]; then
    echo -e "${RED}Error: Model directory '$MODEL_DIR' not found${NC}"
    exit 1
fi

if [ ! -f "$EXECUTABLE" ]; then
    echo -e "${RED}Error: Executable '$EXECUTABLE' not found${NC}"
    echo -e "${YELLOW}Please build first: cd build && make${NC}"
    exit 1
fi

echo -e "${CYAN}╔═══════════════════════════════════════╗${NC}"
echo -e "${CYAN}║    Qwen3-0.6B Benchmark Suite        ║${NC}"
echo -e "${CYAN}╚═══════════════════════════════════════╝${NC}"
echo ""

# Test prompts
PROMPTS=(
    "What is machine learning?"
    "Write a haiku about programming."
    "Explain quantum computing in simple terms."
    "What are the benefits of exercise?"
)

# Temperature settings
TEMPS=(0.0 0.6 1.0)

# Top-k settings
TOP_KS=(10 20 40)

echo -e "${YELLOW}=== Benchmark 1: Temperature Variation ===${NC}"
for temp in "${TEMPS[@]}"; do
    echo -e "${CYAN}Temperature: $temp${NC}"
    result=$($EXECUTABLE "$MODEL_DIR" -t "$temp" -i "${PROMPTS[0]}" 2>&1 | grep "tk/s" || echo "Failed")
    echo "  Result: $result"
    sleep 1
done
echo ""

echo -e "${YELLOW}=== Benchmark 2: Top-K Variation ===${NC}"
for k in "${TOP_KS[@]}"; do
    echo -e "${CYAN}Top-K: $k${NC}"
    result=$($EXECUTABLE "$MODEL_DIR" -k "$k" -i "${PROMPTS[1]}" 2>&1 | grep "tk/s" || echo "Failed")
    echo "  Result: $result"
    sleep 1
done
echo ""

echo -e "${YELLOW}=== Benchmark 3: Multiple Prompts ===${NC}"
total_tps=0
count=0
for i in "${!PROMPTS[@]}"; do
    echo -e "${CYAN}Prompt $((i+1)): ${PROMPTS[$i]}${NC}"
    output=$($EXECUTABLE "$MODEL_DIR" -t 0.6 -i "${PROMPTS[$i]}" 2>&1)
    result=$(echo "$output" | grep "tk/s" || echo "Failed")
    echo "  Result: $result"
    
    # Extract tokens/sec for average
    tps=$(echo "$result" | grep -oP '\d+\.\d+(?= tk/s)' || echo "0")
    if [ "$tps" != "0" ]; then
        total_tps=$(echo "$total_tps + $tps" | bc)
        count=$((count + 1))
    fi
    sleep 1
done

if [ $count -gt 0 ]; then
    avg_tps=$(echo "scale=2; $total_tps / $count" | bc)
    echo -e "${GREEN}Average tokens/sec: $avg_tps${NC}"
fi
echo ""

echo -e "${YELLOW}=== Benchmark 4: Memory Usage ===${NC}"
# Run in background and monitor GPU memory
$EXECUTABLE "$MODEL_DIR" -i "Tell me a story." > /dev/null 2>&1 &
PID=$!

echo "Monitoring GPU 0 memory (5 samples)..."
for i in {1..5}; do
    if ps -p $PID > /dev/null; then
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id=0
        sleep 0.5
    fi
done

# Wait for process to complete
wait $PID 2>/dev/null || true
echo ""

echo -e "${YELLOW}=== Benchmark 5: Reasoning Mode ===${NC}"
echo -e "${CYAN}Standard mode:${NC}"
result=$($EXECUTABLE "$MODEL_DIR" -r 0 -i "Solve: 2+2=?" 2>&1 | grep "tk/s" || echo "Failed")
echo "  Result: $result"

echo -e "${CYAN}Reasoning mode:${NC}"
result=$($EXECUTABLE "$MODEL_DIR" -r 1 -i "Solve: 2+2=?" 2>&1 | grep "tk/s" || echo "Failed")
echo "  Result: $result"
echo ""

echo -e "${GREEN}╔═══════════════════════════════════════╗${NC}"
echo -e "${GREEN}║    Benchmark Complete                 ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════╝${NC}"

# Optional: Profile with Nsight Systems if available
if command -v nsys &> /dev/null; then
    echo ""
    echo -e "${YELLOW}Nsight Systems profiler detected!${NC}"
    echo -e "${CYAN}Run detailed profiling with:${NC}"
    echo "  nsys profile -o qwen_profile $EXECUTABLE $MODEL_DIR -i \"Hello\""
    echo "  nsys-ui qwen_profile.nsys-rep"
fi

