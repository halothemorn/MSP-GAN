import torch
from torch import nn


def depth(size):
    if size < 8 or size & (size - 1):
        raise ValueError("Network input width must be a power of two of at least 8.")
    return size.bit_length() - 3


class Generator3D(nn.Module):
    def __init__(self, size, latent=32, width=64):
        super().__init__()
        blocks = []
        channels = latent
        for stage in range(depth(size)):
            last = stage == depth(size) - 1
            out = 3 if last else width * min(2 ** (depth(size) - 2 - stage), 8)
            blocks.append(nn.ConvTranspose3d(channels, out, 4, 2, 1, bias=last))
            if not last:
                blocks.extend([nn.GroupNorm(8, out), nn.ReLU(inplace=True)])
            channels = out
        self.body = nn.Sequential(*blocks)

    def forward(self, z):
        return self.body(z).softmax(dim=1)


def critic(size, width=64):
    blocks = []
    channels = 3
    for stage in range(depth(size)):
        out = width * min(2 ** stage, 8)
        blocks.extend([nn.Conv2d(channels, out, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)])
        channels = out
    blocks.append(nn.Conv2d(channels, 1, 4))
    return nn.Sequential(*blocks)


def cross_sections(volume, count):
    planes = []
    for axis in (3, 4):
        indices = torch.randint(volume.shape[axis], (count,), device=volume.device)
        sections = volume.index_select(axis, indices).movedim(axis, 1)
        planes.append(sections.flatten(0, 1))
    return torch.cat(planes)

