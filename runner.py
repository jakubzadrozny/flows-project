from argparse import ArgumentParser
from itertools import islice

import torch
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal
from torchvision import transforms, datasets

import pytorch_lightning as pl
from pytorch_lightning.loggers import TestTubeLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from realNVP.realnvp import RealNVP, Hyperparameters
from realNVP.data_utils import DataInfo
from VAE.models import VAE
from GAN.models import GAN
from glow.utils import preprocess
from glow.model import Glow

def get_args():
    parser = ArgumentParser(description="Run training on one of the generative models.")
    parser.add_argument("model", type=str, help="Name of the model you wish to train. ",
                        choices=["realnvp", "glow", "vae", "gan"])
    parser.add_argument("--data_root", default="data/", help="Path to data root directory.")
    parser.add_argument("--logs_root", default="logs/", help="Path to logs root directory.")
    parser.add_argument("--lr", default=3e-3, type=float, help="Learning rate.")
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
    parser.add_argument("--seed", default=42, help="Random seed.")
    parser.add_argument("--workers", default=6, type=int, help="Number of dataloader workers.")
    parser.add_argument("--latent_dim", default=128, help="Latent dimension in VAE / GAN.")
    parser.add_argument("--kl_weight", default=1e-3, help="Weight of the KL-divergence in VAE.")
    parser.add_argument("--generator_features", default=64, help="Number of generator features in GAN.")
    parser.add_argument("--discriminator_features", default=64, help="Number of discriminator features in GAN.")
    parser.add_argument("--flow_coupling", default="additive", choices=["affine", "additive"],
                        help="Type of flow coupling used in Glow model.")
    parser.add_argument("--n_init_batches", default=8, help="Number of batches to use for Act Norm initialisation (for Glow).")
    parser.add_argument("--warmup", default=15, type=int, help="Linearly warmup lr for this many epochs (for Glow).")
    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args()

def main(args):
    pl.seed_everything(args.seed)

    initial_transforms = [
        transforms.CenterCrop(168),
        transforms.Resize(64),
    ]
    augmentation_transforms = [
        transforms.RandomHorizontalFlip(p=0.5),
    ]
    final_transforms = []

    if args.model.lower() == "realnvp":
        datainfo = DataInfo("celeba", 3, 64)
        prior = Normal(loc=0.0, scale=1.0)
        hyperparams = Hyperparameters(base_dim=32, res_blocks=2)

        model = RealNVP(datainfo, prior, hyperparams, args.lr)
    elif args.model.lower() == "vae":
        model = VAE(latent_dim=args.latent_dim,
                    lr=args.lr,
                    kl_weight=args.kl_weight)
    elif args.model.lower() == "gan":
        final_transforms = [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        model = GAN(nz=args.latent_dim,
                    ngf=args.generator_features,
                    ndf=args.discriminator_features,
                    lr=args.lr)
    elif args.model.lower() == "glow":
        final_transforms = [preprocess]

        model = Glow((64, 64, 3), flow_coupling=args.flow_coupling, lr=args.lr, warmup=args.warmup)
    else:
        raise ValueError(f"{args.model} is not a valid model name. Use -h flag for help.")

    train_transform = transforms.Compose(initial_transforms + augmentation_transforms
                                         + [transforms.ToTensor()] + final_transforms)
    test_transform = transforms.Compose(initial_transforms + [transforms.ToTensor()]
                                        + final_transforms)

    train_dataset = datasets.CelebA(root=args.data_root, split="train", transform=train_transform)
    val_dataset = datasets.CelebA(root=args.data_root, split="test", transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    if args.model.lower() == "glow":
        model.train()
        if args.gpus and args.gpus > 0:
            model.to('cuda')

        init_batches = []
        for x, _ in islice(train_loader, None, args.n_init_batches):
            init_batches.append(x)
        init_batches = torch.cat(init_batches).to(model.device)

        with torch.no_grad():
            model(init_batches)

        model.cpu()

    tt_logger = TestTubeLogger(args.logs_root, name=args.model.lower())
    checkpoint_callback = ModelCheckpoint(
        verbose=True,
        save_last=True,
    )

    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=tt_logger,
        deterministic=True,
        min_epochs=5,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    args = get_args()
    main(args)
