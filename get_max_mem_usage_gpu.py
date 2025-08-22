import pynvml  as nvml

nvml.nvmlInit()
handle = nvml.nvmlDeviceGetHandleByIndex(0)  # GPU 0

# Get current memory usage
info = nvml.nvmlDeviceGetMemoryInfo(handle)
print(f"Used: {info.used / 1024**2:.2f} MB")
print(f"Total: {info.total / 1024**2:.2f} MB")

# For tracking max, you'd need to poll periodically
import time
max_memory = 0
while True:
    info = nvml.nvmlDeviceGetMemoryInfo(handle)
    max_memory = max(max_memory, info.used)
    print(f"\rCurrent max memory usage: {max_memory / 1024**2:.2f} MB", end="")
    time.sleep(0.01)  # Poll every 100ms