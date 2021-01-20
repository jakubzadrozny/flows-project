from argparse import ArgumentParser

from torch.utils.data import DataLoader
from torch.distributions.normal import Normal
from torch.backends import cudnn
from torchvision import transforms, datasets

import pytorch_lightning as pl
from pytorch_lightning.loggers import TestTubeLogger

from realNVP.realnvp import RealNVP, Hyperparameters
from realNVP.data_utils import DataInfo
from VAE.models import VAE
from GAN.models import GAN

def get_args():
    parser = ArgumentParser(description="Run training on one of the generative models.")
    parser.add_argument("model", type=str, help="Name of the model you wish to train. "
                        "Available: RealNVP, VAE, GAN (case insensitive)")
    parser.add_argument("--data_root", default="data/", help="Path to data root directory.")
    parser.add_argument("--logs_root", default="logs/", help="Path to logs root directory.")
    parser.add_argument("--lr", default=3e-3, help="Learning rate.")
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
    parser.add_argument("--seed", default=42, help="Random seed.")
    parser.add_argument("--workers", default=6, type=int, help="Number of dataloader workers.")
    parser.add_argument("--latent_dim", default=128, help="Latent dimension in VAE / GAN.")
    parser.add_argument("--kl_weight", default=1e-3, help="Weight of the KL-divergence in VAE.")
    parser.add_argument("--generator_features", default=64, help="Number of generator features in GAN.")
    parser.add_argument("--discriminator_features", default=64, help="Number of discriminator features in GAN.")
    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args()

def main(args):
    pl.seed_everything(args.seed)

    train_transform = transforms.Compose([
        transforms.CenterCrop(168),
        transforms.Resize(64),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.CenterCrop(168),
        transforms.Resize(64),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.CelebA(root=args.data_root, split="train", transform=train_transform)
    val_dataset = datasets.CelebA(root=args.data_root, split="test", transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

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
        model = GAN(nz=args.latent_dim,
                    ngf=args.generator_features,
                    ndf=args.discriminator_features,
                    lr=args.lr)
    else:
        raise ValueError(f"{args.model} is not a valid model name. Use -h flag for help.")

    tt_logger = TestTubeLogger(args.logs_root, name=args.model.lower())

    trainer = pl.Trainer.from_argparse_args(args, logger=tt_logger, deterministic=True, min_epochs=5)
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    args = get_args()
    main(args)
