import torch
from torch import nn
from torch.nn import functional as F
from torchvision import utils as vutils

import pytorch_lightning as pl

# custom weights initialization called on generator and discriminator
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, nc=3, nz=100, ngf=64):
        super(Generator, self).__init__()
        self.nz = nz
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()

            # state size. (nc) x 64 x 64
        )

        self.main.apply(weights_init)

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, nc=3, nz=100, ndf=64):
        super(Discriminator, self).__init__()
        self.nz = nz
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

        self.main.apply(weights_init)

    def forward(self, input):
        return self.main(input).flatten()


class GAN(pl.LightningModule):
    def __init__(self, nz=100, ngf=64, ndf=64, lr=1e-3, nc=3):
        super().__init__()

        self.lr = lr
        self.nz = nz
        self.save_hyperparameters()

        self.real_label = 1
        self.fake_label = 0

        self.generator = Generator(nc=nc, nz=nz, ngf=ngf)
        self.discriminator = Discriminator(nc=nc, nz=nz, ndf=ndf)
        self.criterion = nn.BCELoss()

        self.fixed_noise = torch.randn(64, self.nz, 1, 1)


    def forward(self, x):
        return self.generator(x)


    def sample(self, n):
        noise = torch.randn(n, self.nz, 1, 1, device=self.device)
        return self(noise)


    def configure_optimizers(self):
        D_optim = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr)
        G_optim = torch.optim.Adam(self.generator.parameters(), lr=self.lr)
        return [D_optim, G_optim], []


    def training_step(self, batch, batch_idx, optimizer_idx):
        x = batch[0]

        if optimizer_idx == 0:
            return self.disc_step(x)
        elif optimizer_idx == 1:
            return self.gen_step(x)


    def disc_step(self, real_x):
        # Forward pass real batch through D
        output = self.discriminator(real_x)
        real_labels = torch.full_like(output, self.real_label)

        # Calculate loss on all-real batch
        real_loss = self.criterion(output, real_labels)
        self.log("disc_real_loss", real_loss)
        self.log("disc_conf_on_real", output.mean())

        # Train with all-fake batch
        with torch.no_grad():
            fake_x = self.sample(real_x.shape[0])
        output = self.discriminator(fake_x)
        fake_labels = torch.full_like(output, self.fake_label)

        # Calculate D's loss on the all-fake batch
        fake_loss = self.criterion(output, fake_labels)
        self.log("disc_fake_loss", fake_loss)
        self.log("disc_conf_on_fake", output.mean())

        loss = real_loss + fake_loss
        self.log("disc_loss", loss, prog_bar=True)

        return loss


    def gen_step(self, real_x):
        fake_x = self.sample(real_x.shape[0])
        output = self.discriminator(fake_x)

        labels = torch.full_like(output, self.real_label)
        loss = self.criterion(output, labels)
        self.log("gen_loss", loss, prog_bar=True)
        return loss

    def training_epoch_end(self, outputs):
        samples = self(self.fixed_noise.to(self.device))
        vutils.save_image(samples.detach().cpu(),
                          f"{self.logger.save_dir}/{self.logger.name}/version_{self.logger.version}/"
                          f"media/{self.logger.name}_{self.current_epoch}_samples.png",
                          nrow=8, normalize=True)
