import torch


def alex_op_pure_pytorch(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    h, n, d = x.shape
    h, c, d = w.shape
    
    x = x[:,  :,None, :]
    x = x.broadcast_to((h, n, c, d))
    
    w = w[:, None, :, :]
    w = w.broadcast_to((h, n, c, d))

    sub = (x - w) ** 2
    distance = sub.sum(dim=-1)
    result = torch.argmin(distance, dim=-1)
    return result

def gabriel_version(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    



if __name__ == "__main__":
    h = 128
    n = 128
    d = 64
    c = 256

    x = torch.randn(h, n, d).cuda()
    w = torch.randn(h, c, d).cuda()
    with torch.no_grad():
        out1 = alex_op_pure_pytorch(x, w)
    print(out1)

    # Get the ideal gpu memory usage (x + w + out1)
    ideal_mem = (x.numel() + w.numel() + out1.numel()) * 4
    print(f"Ideal memory usage: {ideal_mem / 1024**2:.2f} MB")
    
    # Get maximum memory usage since the beginning
    print(f"Max Allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
    print(f"Max Cached: {torch.cuda.max_memory_reserved() / 1024**2:.2f} MB")
