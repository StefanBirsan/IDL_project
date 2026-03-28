import torch
if torch.cuda.is_available():
    print("CUDA is available. You can use GPU for training.")
    #torch.__version__ and torch.version.cuda inside the failing script.
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Supported CUDA architectures: {torch.cuda.get_arch_list()}")