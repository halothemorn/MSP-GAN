import argparse
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler

from data import TrainDataset, full_field_targets, training_field_paths
from losses import correlation_loss, fraction_loss, gradient_penalty
from models import Generator3D, critic, cross_sections


def noisy_scale(images, scale, sigma):
    if scale > 1:
        images = F.avg_pool2d(images, scale)
    return images + sigma * torch.randn_like(images)


def optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.0, 0.9))


def train(args):
    torch.manual_seed(args.seed)
    dataset = TrainDataset(args.data)
    size = dataset.size
    maps = training_field_paths(args.fields, dataset.fields)
    fractions, s2 = full_field_targets(maps, size)
    device = torch.device(args.device)
    generator = Generator3D(size).to(device)
    scales = (1, 2)
    critics = [critic(size // scale).to(device) for scale in scales]
    opt_g = optimizer(generator)
    optimizers = [optimizer(model) for model in critics]
    fractions = torch.tensor(fractions, device=device, dtype=torch.float32)
    target = torch.tensor(s2, device=device, dtype=torch.float32)
    batch = args.volumes * args.slices * 2
    draws = RandomSampler(dataset, replacement=True, num_samples=batch * 5 * args.steps)
    batches = iter(DataLoader(dataset, sampler=draws, batch_size=batch))
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def generate():
        z = torch.randn(args.volumes, 32, 4, 4, 4, device=device)
        return generator(z)

    for step in range(1, args.steps + 1):
        sigma = max(0.1 * (1 - step / args.steps), 0.02)
        penalty_weight = 10 * 0.9 ** (step // 5000)
        for model in critics:
            model.requires_grad_(True)
        for _ in range(5):
            with torch.no_grad():
                fake = cross_sections(generate(), args.slices)
            real = next(batches).to(device)
            for model, opt, scale in zip(critics, optimizers, scales):
                rf = noisy_scale(real, scale, sigma)
                ff = noisy_scale(fake, scale, sigma)
                loss_d = model(ff).mean() - model(rf).mean()
                loss_d = loss_d + penalty_weight * gradient_penalty(model, rf, ff)
                opt.zero_grad()
                loss_d.backward()
                opt.step()

        for model in critics:
            model.requires_grad_(False)
        volume = generate()
        fake = cross_sections(volume, args.slices)
        adv = -sum(model(noisy_scale(fake, scale, sigma)).mean()
                   for model, scale in zip(critics, scales))
        loss_g = adv + 30 * fraction_loss(volume, fractions) + 10 * correlation_loss(volume, target)
        opt_g.zero_grad()
        loss_g.backward()
        opt_g.step()

        if step % 200 == 0 or step == args.steps:
            print(f"Step {step}  generator {loss_g.item():.5f}")
        if step % 1000 == 0 or step == args.steps:
            checkpoint = dict(generator=generator.state_dict(), size=size, step=step,
                              fields=dataset.fields, patches=len(dataset), seed=args.seed,
                              volumes=args.volumes, slices=args.slices, anisotropy="aniso")
            temporary = args.output.with_suffix(args.output.suffix + ".tmp")
            torch.save(checkpoint, temporary)
            temporary.replace(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--fields", type=Path, required=True,
                        help="Root containing one complete label map per training field in train/.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--volumes", type=int, default=3)
    parser.add_argument("--slices", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if min(args.steps, args.volumes, args.slices) < 1:
        parser.error("Steps, volumes and slices must be positive.")
    train(args)

