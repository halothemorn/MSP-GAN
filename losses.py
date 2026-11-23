import torch


def gradient_penalty(critic, real, fake):
    weight = torch.rand(len(real), 1, 1, 1, device=real.device)
    mixed = (weight * real + (1 - weight) * fake).requires_grad_(True)
    gradients = torch.autograd.grad(critic(mixed).sum(), mixed, create_graph=True)[0]
    return (gradients.flatten(1).norm(2, dim=1) - 1).square().mean()


def fraction_loss(volume, target):
    hard = torch.zeros_like(volume).scatter_(1, volume.argmax(1, keepdim=True), 1)
    fraction = (hard + (volume - volume.detach())).mean(dim=(0, 2, 3, 4))
    return ((fraction - target) / (target + 0.02)).square().mean()


def correlation_loss(volume, target):
    """Match both in-plane axes to horizontal SEM S2 and thickness to vertical S2."""
    if target.ndim != 2 or target.shape[0] != 2:
        raise ValueError("Expected horizontal and vertical SEM S2 curves of shape (2, lags).")
    void = volume[:, 0] + volume[:, 1]
    phi = void.mean(dim=(1, 2, 3), keepdim=True)
    spectrum = torch.fft.rfftn(void - phi, dim=(1, 2, 3))
    ac = torch.fft.irfftn(spectrum * spectrum.conj(), s=void.shape[1:], dim=(1, 2, 3))
    phi = phi.reshape(-1, 1)
    horizontal, vertical = target
    lags = target.shape[1]
    lines = torch.stack([ac[:, :lags, 0, 0], ac[:, 0, :lags, 0], ac[:, 0, 0, :lags]])
    estimate = lines / (ac[:, 0, 0, 0, None] + 1e-8) * (phi - phi.square()) + phi.square()
    # The first spatial axis is thickness. Broadcast each curve across the batch.
    axis_target = torch.stack([vertical, horizontal, horizontal]).unsqueeze(1)
    return ((estimate - axis_target) / (axis_target + 1e-3)).square().mean()

