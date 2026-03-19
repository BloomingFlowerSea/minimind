import torch

def check_gpu_devices():
    """检查可用的GPU显卡数量"""
    num_gpus = torch.cuda.device_count()
    print(f"可用的GPU数量: {num_gpus}")
    
    if num_gpus > 0:
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"GPU {i}: {gpu_name}")
    else:
        print("没有可用的GPU")
    
    return num_gpus

if __name__ == "__main__":
    check_gpu_devices()