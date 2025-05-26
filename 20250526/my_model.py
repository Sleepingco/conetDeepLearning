import torch

print(f"PyTorch 버전: {torch.__version__}")
print(f"CUDA 버전: {torch.version.cuda}")
print(f"cuDNN 버전: {torch.backends.cudnn.version()}")
print(f"CUDA 사용 가능 여부: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"사용 가능한 GPU 개수: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

        