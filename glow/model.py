import math

import torch
import torch.nn as nn
from torchvision import utils as vutils

import pytorch_lightning as pl

from .modules import (
    Conv2d,
    Conv2dZeros,
    ActNorm2d,
    InvertibleConv1x1,
    Permute2d,
    LinearZeros,
    SqueezeLayer,
    Split2d,
    gaussian_likelihood,
    gaussian_sample,
)
from .utils import split_feature, uniform_binning_correction, postprocess


def get_block(in_channels, out_channels, hidden_channels):
    block = nn.Sequential(
        Conv2d(in_channels, hidden_channels),
        nn.ReLU(inplace=False),
        Conv2d(hidden_channels, hidden_channels, kernel_size=(1, 1)),
        nn.ReLU(inplace=False),
        Conv2dZeros(hidden_channels, out_channels),
    )
    return block


class FlowStep(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        actnorm_scale,
        flow_permutation,
        flow_coupling,
        LU_decomposed,
    ):
        super().__init__()
        self.flow_coupling = flow_coupling

        self.actnorm = ActNorm2d(in_channels, actnorm_scale)

        # 2. permute
        if flow_permutation == "invconv":
            self.invconv = InvertibleConv1x1(in_channels, LU_decomposed=LU_decomposed)
            self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)
        elif flow_permutation == "shuffle":
            self.shuffle = Permute2d(in_channels, shuffle=True)
            self.flow_permutation = lambda z, logdet, rev: (
                self.shuffle(z, rev),
                logdet,
            )
        elif flow_permutation == "reverse":
            self.reverse = Permute2d(in_channels, shuffle=False)
            self.flow_permutation = lambda z, logdet, rev: (
                self.reverse(z, rev),
                logdet,
            )
        else:
            raise ValueError(f"Value {flow_permutation} for parameter flow_permutation unsupported.")

        # 3. coupling
        if flow_coupling == "additive":
            self.block = get_block(in_channels // 2, in_channels // 2, hidden_channels)
        elif flow_coupling == "affine":
            self.block = get_block(in_channels // 2, in_channels, hidden_channels)
        else:
            raise ValueError(f"Value {flow_coupling} for parameter flow_coupling unsupported.")

    def forward(self, input, logdet=None, reverse=False):
        if not reverse:
            return self.normal_flow(input, logdet)
        else:
            return self.reverse_flow(input, logdet)

    def normal_flow(self, input, logdet):
        assert input.size(1) % 2 == 0

        # 1. actnorm
        z, logdet = self.actnorm(input, logdet=logdet, reverse=False)

        # 2. permute
        z, logdet = self.flow_permutation(z, logdet, False)

        # 3. coupling
        z1, z2 = split_feature(z, "split")
        if self.flow_coupling == "additive":
            z2 = z2 + self.block(z1)
        elif self.flow_coupling == "affine":
            h = self.block(z1)
            shift, scale = split_feature(h, "cross")
            scale = torch.sigmoid(scale + 2.0)
            z2 = z2 + shift
            z2 = z2 * scale
            logdet = torch.sum(torch.log(scale), dim=[1, 2, 3]) + logdet
        z = torch.cat((z1, z2), dim=1)

        return z, logdet

    def reverse_flow(self, input, logdet):
        assert input.size(1) % 2 == 0

        # 1.coupling
        z1, z2 = split_feature(input, "split")
        if self.flow_coupling == "additive":
            z2 = z2 - self.block(z1)
        elif self.flow_coupling == "affine":
            h = self.block(z1)
            shift, scale = split_feature(h, "cross")
            scale = torch.sigmoid(scale + 2.0)
            z2 = z2 / scale
            z2 = z2 - shift
            logdet = -torch.sum(torch.log(scale), dim=[1, 2, 3]) + logdet
        z = torch.cat((z1, z2), dim=1)

        # 2. permute
        z, logdet = self.flow_permutation(z, logdet, True)

        # 3. actnorm
        z, logdet = self.actnorm(z, logdet=logdet, reverse=True)

        return z, logdet


class FlowNet(nn.Module):
    def __init__(
        self,
        image_shape,
        hidden_channels,
        K,
        L,
        actnorm_scale,
        flow_permutation,
        flow_coupling,
        LU_decomposed,
    ):
        super().__init__()

        self.layers = nn.ModuleList()
        self.output_shapes = []

        self.K = K
        self.L = L

        H, W, C = image_shape

        for i in range(L):
            # 1. Squeeze
            C, H, W = C * 4, H // 2, W // 2
            self.layers.append(SqueezeLayer(factor=2))
            self.output_shapes.append([-1, C, H, W])

            # 2. K FlowStep
            for _ in range(K):
                self.layers.append(
                    FlowStep(
                        in_channels=C,
                        hidden_channels=hidden_channels,
                        actnorm_scale=actnorm_scale,
                        flow_permutation=flow_permutation,
                        flow_coupling=flow_coupling,
                        LU_decomposed=LU_decomposed,
                    )
                )
                self.output_shapes.append([-1, C, H, W])

            # 3. Split2d
            if i < L - 1:
                self.layers.append(Split2d(num_channels=C))
                self.output_shapes.append([-1, C // 2, H, W])
                C = C // 2

    def forward(self, input, logdet=0.0, reverse=False, temperature=None):
        if reverse:
            return self.decode(input, temperature)
        else:
            return self.encode(input, logdet)

    def encode(self, z, logdet=0.0):
        for layer, shape in zip(self.layers, self.output_shapes):
            z, logdet = layer(z, logdet, reverse=False)
        return z, logdet

    def decode(self, z, temperature=None):
        for layer in reversed(self.layers):
            if isinstance(layer, Split2d):
                z, logdet = layer(z, logdet=0, reverse=True, temperature=temperature)
            else:
                z, logdet = layer(z, logdet=0, reverse=True)
        return z


class Glow(pl.LightningModule):
    def __init__(
        self,
        image_shape,
        hidden_channels=512,
        K=32,
        L=3,
        flow_permutation="invconv",
        flow_coupling="affine",
        actnorm_scale=1.0,
        LU_decomposed=True,
        y_classes=0,
        learn_top=True,
        y_condition=False,
        lr=1e-3,
        warmup=15,
    ):
        super().__init__()
        self.flow = FlowNet(
            image_shape=image_shape,
            hidden_channels=hidden_channels,
            K=K,
            L=L,
            actnorm_scale=actnorm_scale,
            flow_permutation=flow_permutation,
            flow_coupling=flow_coupling,
            LU_decomposed=LU_decomposed,
        )
        self.y_classes = y_classes
        self.y_condition = y_condition

        self.learn_top = learn_top
        self.lr = lr
        self.warmup = warmup

        # learned prior
        if learn_top:
            C = self.flow.output_shapes[-1][1]
            self.learn_top_fn = Conv2dZeros(C * 2, C * 2)

        if y_condition:
            C = self.flow.output_shapes[-1][1]
            self.project_ycond = LinearZeros(y_classes, 2 * C)
            self.project_class = LinearZeros(C, y_classes)

        self.register_buffer(
            "prior_h",
            torch.zeros(
                [
                    1,
                    self.flow.output_shapes[-1][1] * 2,
                    self.flow.output_shapes[-1][2],
                    self.flow.output_shapes[-1][3],
                ]
            ),
        )

        mean, logs = self.prior(64)
        self.fixed_input = gaussian_sample(mean, logs, 0.8)

    def prior(self, n, y_onehot=None):
        h = self.prior_h.repeat(n, 1, 1, 1)
        channels = h.size(1)

        if self.learn_top:
            h = self.learn_top_fn(h)

        if self.y_condition:
            assert y_onehot is not None
            yp = self.project_ycond(y_onehot)
            h += yp.view(h.shape[0], channels, 1, 1)

        return split_feature(h, "split")

    def forward(self, x, y_onehot=None):
        return self.normal_flow(x, y_onehot)

    def normal_flow(self, x, y_onehot):
        b, c, h, w = x.shape

        x, logdet = uniform_binning_correction(x)

        z, objective = self.flow(x, logdet=logdet, reverse=False)

        mean, logs = self.prior(x.shape[0], y_onehot)
        objective += gaussian_likelihood(mean, logs, z)

        if self.y_condition:
            y_logits = self.project_class(z.mean(2).mean(2))
        else:
            y_logits = None

        # Full objective - converted to bits per dimension
        bpd = (-objective) / (math.log(2.0) * c * h * w)

        return z, bpd, y_logits

    def reverse_flow(self, z, temperature):
        return self.flow(z, temperature=temperature, reverse=True)

    def sample(self, n, temperature, y_onehot=None):
        mean, logs = self.prior(n, y_onehot)
        z = gaussian_sample(mean, logs, temperature)
        return self.reverse_flow(z, temperature)

    def set_actnorm_init(self):
        for name, m in self.named_modules():
            if isinstance(m, ActNorm2d):
                m.inited = True

    def training_step(self, batch, batch_idx):
        x = batch[0]
        _, nll, _ = self(x)

        loss = torch.mean(nll)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0]
        _, nll, _ = self(x)

        loss = torch.mean(nll)
        self.log('val_loss', loss)
        return loss

    def validation_epoch_end(self, outputs):
        samples = self.reverse_flow(self.fixed_input.to(self.device), temperature=0.8)
        samples = postprocess(samples)
        vutils.save_image(samples.detach().cpu(),
                          f"{self.logger.save_dir}/{self.logger.name}/version_{self.logger.version}/"
                          f"media/{self.logger.name}_{self.current_epoch}.png",
                          nrow=8)

    def configure_optimizers(self):
        optimizer = torch.optim.Adamax(self.parameters(), lr=self.lr, weight_decay=5e-5)
        lr_lambda = lambda epoch: min(1.0, (epoch + 1) / self.warmup)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return [optimizer], [scheduler]
