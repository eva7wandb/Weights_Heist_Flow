import torch

def is_cuda(debug=True):
    cuda = torch.cuda.is_available()
    if debug:
        print("[INFO] Cuda Avaliable : ", cuda)
    return cuda


def get_device():
    use_cuda = is_cuda(debug=False)
    device = torch.device("cuda" if use_cuda else "cpu")
    print("[INFO] device : ", device)
    return device


def set_seed(seed=1):
    cuda = is_cuda(debug=False)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
    print(f"[INFO] seed set {seed}")