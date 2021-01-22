import torch
from torch import nn
from torch.nn import functional as F
from torchvision import utils as vutils

import pytorch_lightning as pl


class VAE(pl.LightningModule):

    def __init__(self, in_channels=3, latent_dim=128, hidden_dims=None, lr=1e-3, kl_weight=1e-3):
        super().__init__()

        self.latent_dim = latent_dim
        self.lr = lr
        self.kl_weight = kl_weight

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Sigmoid())

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), mu, log_var]

    def loss_function(self, input, recons, mu, log_var):
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons_loss = F.binary_cross_entropy(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + self.kl_weight * kld_loss
        return {'loss': loss, 'rec_loss': recons_loss, 'kl': kld_loss}

    def sample(self, num_samples):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim,
                        device=self.device)

        samples = self.decode(z)
        return samples

    def generate(self, x):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]


    def training_step(self, batch, batch_idx):
        x = batch[0]
        rec, mu, log_var = self(x)
        losses = self.loss_function(x, rec, mu, log_var)

        self.log('train_loss', losses['loss'])
        self.log('train_rec_loss', losses['rec_loss'])
        self.log('train_kl', losses['kl'])

        return losses['loss']


    def validation_step(self, batch, batch_idx):
        x = batch[0]
        rec, mu, log_var = self(x)
        losses = self.loss_function(x, rec, mu, log_var)

        self.log('val_loss', losses['loss'])
        self.log('val_rec_loss', losses['rec_loss'])
        self.log('val_kl', losses['kl'])

        if batch_idx == 0:
            vutils.save_image(rec.detach().cpu(),
                              f"{self.logger.save_dir}/{self.logger.name}/version_{self.logger.version}/"
                              f"media/{self.logger.name}_{self.current_epoch}_rec.png",
                              nrow=8)

        return losses['loss']


    def validation_epoch_end(self, outputs):
        samples = self.sample(64)
        vutils.save_image(samples.detach().cpu(),
                          f"{self.logger.save_dir}/{self.logger.name}/version_{self.logger.version}/"
                          f"media/{self.logger.name}_{self.current_epoch}_samples.png",
                          nrow=8)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
