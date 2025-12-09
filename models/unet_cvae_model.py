import torch
import torch.nn as nn
import functools

from .unetbaseline_model import get_norm_layer, init_net


class VAEBottleneck(nn.Module):
    """
    Simple VAE bottleneck operating on [B, C, 1, 1] feature maps.

    Returns:
        h_recon: reconstructed feature map with same shape as input
        kl:      KL divergence loss term (scalar tensor)
    """

    def __init__(self, in_channels: int, latent_dim: int = 128):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim

        # Since spatial size is expected to be 1x1, we only use channel dimension
        feat_dim = in_channels

        self.fc_mu = nn.Linear(feat_dim, latent_dim)
        self.fc_logvar = nn.Linear(feat_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, feat_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, h: torch.Tensor):
        # h: [B, C, 1, 1]
        B, C, _, _ = h.shape
        h_flat = h.view(B, C)

        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        z = self.reparameterize(mu, logvar)
        h_recon = self.fc_dec(z).view(B, C, 1, 1)

        # KL divergence averaged over batch
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        return h_recon, kl


class UnetSkipConnectionBlockVAE(nn.Module):
    """
    U-Net submodule with skip connection and optional VAE bottleneck at the innermost block.

        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(
        self,
        cfg,
        outer_nc,
        inner_nc,
        input_nc=None,
        submodule=None,
        outermost=False,
        innermost=False,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
        latent_dim: int = 128,
    ):
        super().__init__()
        self.outermost = outermost
        self.innermost = innermost
        self.submodule = submodule

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        if input_nc is None:
            input_nc = outer_nc

        # Common modules
        self.downconv = nn.Conv2d(
            input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias
        )
        self.downrelu = nn.LeakyReLU(0.2, True)
        self.downnorm = norm_layer(inner_nc)

        self.uprelu = nn.ReLU(True)
        self.upnorm = norm_layer(outer_nc)

        if outermost:
            # Outermost: no norm on input, no skip-connection concat at the very end
            self.upconv = nn.ConvTranspose2d(
                inner_nc * 2,
                outer_nc,
                kernel_size=4,
                stride=2,
                padding=1,
            )

            # Depth normalization rule follows original implementation
            if cfg.dataset.depth_norm:
                # No final activation (allows outputs beyond [0,1])
                self.use_final_relu = False
            else:
                self.use_final_relu = True
                self.final_relu = nn.ReLU()

            # submodule is another UnetSkipConnectionBlockVAE
        elif innermost:
            # Innermost: this is where we insert the VAE bottleneck
            self.upconv = nn.ConvTranspose2d(
                inner_nc,
                outer_nc,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=use_bias,
            )
            self.vae = VAEBottleneck(inner_nc, latent_dim=latent_dim)
        else:
            # Intermediate layers
            # Note: if the submodule is the innermost block, its output has
            # `inner_nc` channels (not 2 * inner_nc). For deeper blocks, the
            # submodule output has 2 * inner_nc channels due to skip-connections.
            if isinstance(submodule, UnetSkipConnectionBlockVAE) and submodule.innermost:
                up_in_channels = inner_nc
            else:
                up_in_channels = inner_nc * 2

            self.upconv = nn.ConvTranspose2d(
                up_in_channels,
                outer_nc,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=use_bias,
            )
            self.use_dropout = use_dropout
            if use_dropout:
                self.dropout = nn.Dropout(0.5)
            else:
                self.dropout = None

        self.cfg = cfg

    def forward(self, x):
        """
        Returns:
            out: feature map
            kl:  accumulated KL divergence from all VAE bottlenecks below (scalar tensor)
        """
        if self.outermost:
            # Down
            x_down = self.downconv(x)

            # Submodule
            assert self.submodule is not None, "Outermost block must have a submodule"
            x_sub, kl = self.submodule(x_down)

            # Up
            x_up = self.uprelu(x_sub)
            x_up = self.upconv(x_up)
            if self.use_final_relu:
                x_up = self.final_relu(x_up)

            return x_up, kl

        elif self.innermost:
            # Down
            h = self.downrelu(x)
            h = self.downconv(h)

            # VAE bottleneck
            h_recon, kl = self.vae(h)

            # Up
            x_up = self.uprelu(h_recon)
            x_up = self.upconv(x_up)
            x_up = self.upnorm(x_up)

            return x_up, kl

        else:
            # Down
            h = self.downrelu(x)
            h = self.downconv(h)
            h = self.downnorm(h)

            # Submodule
            assert self.submodule is not None, "Intermediate block must have a submodule"
            h_sub, kl = self.submodule(h)

            # Up
            x_up = self.uprelu(h_sub)
            x_up = self.upconv(x_up)
            x_up = self.upnorm(x_up)

            if self.dropout is not None:
                x_up = self.dropout(x_up)

            # Skip connection
            out = torch.cat([x, x_up], 1)
            return out, kl


class UnetGeneratorVAE(nn.Module):
    """
    U-Net generator with a VAE bottleneck at the innermost layer.

    Forward:
        depth, kl = model(audio)
    """

    def __init__(
        self,
        cfg,
        input_nc,
        output_nc,
        num_downs,
        ngf=64,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
        latent_dim: int = 128,
    ):
        super().__init__()

        # Construct Unet structure from innermost to outermost
        unet_block = UnetSkipConnectionBlockVAE(
            cfg,
            ngf * 8,
            ngf * 8,
            input_nc=None,
            submodule=None,
            norm_layer=norm_layer,
            innermost=True,
            latent_dim=latent_dim,
        )
        for _ in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlockVAE(
                cfg,
                ngf * 8,
                ngf * 8,
                input_nc=None,
                submodule=unet_block,
                norm_layer=norm_layer,
                use_dropout=use_dropout,
                latent_dim=latent_dim,
            )
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlockVAE(
            cfg,
            ngf * 4,
            ngf * 8,
            input_nc=None,
            submodule=unet_block,
            norm_layer=norm_layer,
            latent_dim=latent_dim,
        )
        unet_block = UnetSkipConnectionBlockVAE(
            cfg,
            ngf * 2,
            ngf * 4,
            input_nc=None,
            submodule=unet_block,
            norm_layer=norm_layer,
            latent_dim=latent_dim,
        )
        unet_block = UnetSkipConnectionBlockVAE(
            cfg,
            ngf,
            ngf * 2,
            input_nc=None,
            submodule=unet_block,
            norm_layer=norm_layer,
            latent_dim=latent_dim,
        )
        self.model = UnetSkipConnectionBlockVAE(
            cfg,
            output_nc,
            ngf,
            input_nc=input_nc,
            submodule=unet_block,
            outermost=True,
            norm_layer=norm_layer,
            latent_dim=latent_dim,
        )

    def forward(self, input):
        """Returns depth prediction and KL divergence."""
        depth, kl = self.model(input)
        return depth, kl


def define_G_cvae(
    cfg,
    input_nc,
    output_nc,
    ngf,
    netG,
    norm="batch",
    use_dropout=False,
    init_type="normal",
    init_gain=0.02,
    gpu_ids=None,
    latent_dim: int = 128,
):
    """
    Create a generator with a VAE bottleneck.

    This mirrors `define_G` in `unetbaseline_model.py` but uses `UnetGeneratorVAE`.
    """
    if gpu_ids is None:
        gpu_ids = []

    norm_layer = get_norm_layer(norm_type=norm)

    if netG == "unet_128":
        net = UnetGeneratorVAE(
            cfg,
            input_nc,
            output_nc,
            num_downs=7,
            ngf=ngf,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
            latent_dim=latent_dim,
        )
    elif netG == "unet_256":
        net = UnetGeneratorVAE(
            cfg,
            input_nc,
            output_nc,
            num_downs=8,
            ngf=ngf,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
            latent_dim=latent_dim,
        )
    else:
        raise NotImplementedError(f"Generator model name [{netG}] is not recognized for cVAE U-Net")

    return init_net(net, init_type, init_gain, gpu_ids)


