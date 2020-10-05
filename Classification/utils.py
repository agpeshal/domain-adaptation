import torch

def random(shape, device="cpu"):
    dist = torch.distributions.normal.Normal(0, 1)
    samples = dist.rsample(shape)
    samples = samples / samples.reshape(shape[0], -1).norm(dim=1)[:, None, None, None]
    return samples.to(device)
