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
    void = volume[:, 0] + volume[:, 1]
    phi = void.mean(dim=(1, 2, 3), keepdim=True)
    spectrum = torch.fft.rfftn(void - phi, dim=(1, 2, 3))
    ac = torch.fft.irfftn(spectrum * spectrum.conj(), s=void.shape[1:], dim=(1, 2, 3))
    phi = phi.reshape(-1, 1)
    lags = len(target)
    lines = torch.stack([ac[:, :lags, 0, 0], ac[:, 0, :lags, 0], ac[:, 0, 0, :lags]])
    estimate = lines / (ac[:, 0, 0, 0, None] + 1e-8) * (phi - phi.square()) + phi.square()
    return ((estimate - target) / (target + 1e-3)).square().mean()

