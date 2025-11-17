import torch
print(torch.__version__)          # PyTorch版本
print(torch.version.cuda)         # PyTorch编译使用的CUDA版本
print(torch.cuda.is_available())  
print(torch.cuda.device_count())

import torch

# 检查是否有GPUs可用
if torch.cuda.is_available():
    print(f"可用的GPU设备: {torch.cuda.get_device_name(0)}")
else:
    print("没有可用的GPU设备！")