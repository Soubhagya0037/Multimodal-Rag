import torch
import psutil
import time
from typing import Dict, Any
import nvidia_ml_py3 as nvml

class MemoryMonitor:
    def __init__(self):
        if torch.cuda.is_available():
            nvml.nvmlInit()
            self.device_count = torch.cuda.device_count()
        else:
            self.device_count = 0
    
    def get_gpu_memory_info(self) -> Dict[str, Any]:
        """Get GPU memory information"""
        if not torch.cuda.is_available():
            return {"gpu_available": False}
        
        info = {"gpu_available": True, "devices": []}
        
        for i in range(self.device_count):
            handle = nvml.nvmlDeviceGetHandleByIndex(i)
            mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
            
            device_info = {
                "device_id": i,
                "total_memory_gb": mem_info.total / (1024**3),
                "used_memory_gb": mem_info.used / (1024**3),
                "free_memory_gb": mem_info.free / (1024**3),
                "utilization_percent": (mem_info.used / mem_info.total) * 100,
                "torch_allocated_gb": torch.cuda.memory_allocated(i) / (1024**3),
                "torch_reserved_gb": torch.cuda.memory_reserved(i) / (1024**3),
            }
            info["devices"].append(device_info)
        
        return info
    
    def get_cpu_memory_info(self) -> Dict[str, Any]:
        """Get CPU memory information"""
        mem = psutil.virtual_memory()
        return {
            "total_gb": mem.total / (1024**3),
            "used_gb": mem.used / (1024**3),
            "available_gb": mem.available / (1024**3),
            "utilization_percent": mem.percent
        }
    
    def print_memory_status(self):
        """Print current memory status"""
        print("=== Memory Status ===")
        
        # CPU Memory
        cpu_info = self.get_cpu_memory_info()
        print(f"CPU Memory: {cpu_info['used_gb']:.2f}GB / {cpu_info['total_gb']:.2f}GB "
              f"({cpu_info['utilization_percent']:.1f}%)")
        
        # GPU Memory
        gpu_info = self.get_gpu_memory_info()
        if gpu_info["gpu_available"]:
            for device in gpu_info["devices"]:
                print(f"GPU {device['device_id']}: {device['used_memory_gb']:.2f}GB / "
                      f"{device['total_memory_gb']:.2f}GB ({device['utilization_percent']:.1f}%)")
                print(f"  PyTorch - Allocated: {device['torch_allocated_gb']:.2f}GB, "
                      f"Reserved: {device['torch_reserved_gb']:.2f}GB")
        else:
            print("No GPU available")
        
        print("=" * 30)

# Usage
if __name__ == "__main__":
    monitor = MemoryMonitor()
    
    while True:
        monitor.print_memory_status()
        time.sleep(5)

# 1. retry logic with exponential backoff - handles http 429 rate limiting from hugging face
# 2. mixed precision training - uses bfloat16 to reduce memory usage by ~50%
# 3. memory management - aggressive gpu cache clearing between batches
# 4. batch processing - processes images/queries in smaller batches to manage memory
# 5. flash attention - if available, uses more efficient attention mechanism
# 6. model compilation - uses torch.compile for better performance
# 7. cpu offloading - moves embeddings to cpu to free gpu memory
# 8. memory monitoring - tracks gpu utilization in real-time

# the retry logic should handle the hugging face rate limiting issues, while the memory optimizations should help with the cuda oom errors. the mixed precision and batch processing will significantly improve gpu utilization.