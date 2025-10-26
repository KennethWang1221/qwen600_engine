# CUDA-GDB Quick Guide

## 1. Start Debugging
```bash
# Build with debug info
cd ~/qwen600_engine
mkdir -p build && cd build
rm -rf *
cmake -DCMAKE_BUILD_TYPE=Debug ..
make -j$(nproc)

# Start debugger with TUI (split-screen) mode
cuda-gdb-python3.12-tui qwen600_engine

# Program parameters:
# -r <int>     # reasoning mode (0/1)
# -t <float>   # temperature (e.g., 0.65)
# -p <float>   # top-p sampling (e.g., 0.9)
# -k <int>     # top-k sampling (e.g., 20)
```

## 2. Essential Commands

### Basic Control
```bash
n, next          # Next line
s, step          # Step into function
c, continue      # Continue to next breakpoint
q, quit          # Exit debugger
```

### Display Control
```bash
layout src       # Show source code view
layout regs      # Show registers
refresh          # Refresh screen if messy
Ctrl-L          # Clear/refresh screen
Ctrl-X-o        # Switch between windows
```

### Stack Navigation
```bash
bt               # Show call stack (backtrace)
frame N          # Jump to frame N (0 is current)
up               # Go up one frame
down             # Go back down one frame
frame            # Show current frame location
```

### Breakpoints
```bash
break main                  # Set breakpoint at function
break file.cpp:123         # Set breakpoint at line
info breakpoints           # List breakpoints
delete N                   # Delete breakpoint N
```

### Variables/Memory
```bash
print var                  # Print variable
print *array@10           # Print 10 elements of array
info locals               # Show local variables
```

### Fix Common Issues
```bash
# If display repeats lines:
set pagination on
set height 0
set logging off

# If variables show "optimized out":
set print static-members off
set print object on
```

## 3. Typical Debug Session
```bash
(cuda-gdb) layout src                  # Show source view
(cuda-gdb) break main                  # Set breakpoint
# Run with parameters
(cuda-gdb) run ../../Qwen3-0.6B -r 1 -t 0.65 -p 0.9 -k 20

# Or if you need to run again with different parameters:
(cuda-gdb) run /path/to/model -r 1 -t 0.7 -k 30    # Change parameters as needed
(cuda-gdb) n                          # Step through code
(cuda-gdb) bt                         # Check call stack
(cuda-gdb) up                         # Go up stack
(cuda-gdb) down                       # Go back down
(cuda-gdb) print var                  # Check variable
```
