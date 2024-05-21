import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch
import einops


def weight_init(shape, mode, fan_in, fan_out):
    if mode == "kaiming_uniform":
        return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == "kaiming_normal":
        return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')


class Linear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        init_mode="kaiming_normal",
        init_weight=1,
        init_bias=0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(
            weight_init([out_features, in_features], **init_kwargs) * init_weight
        )
        self.bias = (
            torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias)
            if bias
            else None
        )

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x


class Conv1d(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel=1,
        bias=False,
        dilation=1,
        init_mode="kaiming_normal",
        init_weight=1,
        init_bias=0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilation = dilation
        init_kwargs = dict(
            mode=init_mode, fan_in=in_channels * kernel, fan_out=out_channels * kernel
        )
        self.weight = torch.nn.Parameter(
            weight_init([out_channels, in_channels, kernel], **init_kwargs)
            * init_weight
        )
        self.bias = (
            torch.nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias)
            if bias
            else None
        )

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        if w is not None:
            x = torch.nn.functional.conv1d(x, w, padding="same", dilation=self.dilation)
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1))
        return x


class Conv2d(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel=(1, 1),
        bias=False,
        dilation=1,
        init_mode="kaiming_normal",
        init_weight=1,
        init_bias=0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilation = dilation
        init_kwargs = dict(
            mode=init_mode,
            fan_in=in_channels * kernel[0] * kernel[1],
            fan_out=out_channels * kernel[0] * kernel[1],
        )
        self.weight = torch.nn.Parameter(
            weight_init(
                [out_channels, in_channels, kernel[0], kernel[1]], **init_kwargs
            )
            * init_weight
        )
        self.bias = (
            torch.nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias)
            if bias
            else None
        )

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        if w is not None:
            x = torch.nn.functional.conv2d(x, w, padding="same", dilation=self.dilation)
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1, 1))
        return x


class BiasFreeGroupNorm(nn.Module):

    def __init__(self, num_features, num_groups=32, eps=1e-7):
        super(BiasFreeGroupNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x):
        N, C, F, T = x.size()
        gc = C // self.num_groups
        x = einops.rearrange(
            x, "n (g gc) f t -> n g (gc f t)", g=self.num_groups, gc=gc
        )

        std = x.std(-1, keepdim=True)  # reduce over channels and time

        ## normalize
        x = (x) / (std + self.eps)
        # normalize
        x = einops.rearrange(
            x, "n g (gc f t) -> n (g gc) f t", g=self.num_groups, gc=gc, f=F, t=T
        )
        return x * self.gamma


class RFF_MLP_Block(nn.Module):
    """
    Encoder of the noise level embedding
    Consists of:
        -Random Fourier Feature embedding
        -MLP
    """

    def __init__(self, emb_dim=512, rff_dim=32, inputs=1, init=None):
        super().__init__()
        self.inputs = inputs
        self.RFF_freq = nn.Parameter(
            16 * torch.randn([1, rff_dim]), requires_grad=False
        )
        self.MLP = nn.ModuleList(
            [
                Linear(2 * rff_dim * self.inputs, 128, **init),
                Linear(128, 256, **init),
                Linear(256, emb_dim, **init),
            ]
        )

    def forward(self, x):
        """
        Arguments:
          sigma:
              (shape: [B, 1], dtype: float32)

        Returns:
          x: embedding of sigma
              (shape: [B, 512], dtype: float32)
        """
        x = [
            self._build_RFF_embedding(x[:, i].unsqueeze(-1)) for i in range(self.inputs)
        ]
        x = torch.cat(x, dim=1)

        for layer in self.MLP:
            x = F.relu(layer(x))
        return x

    def _build_RFF_embedding(self, sigma):
        """
        Arguments:
          sigma:
              (shape: [B, 1], dtype: float32)
        Returns:
          table:
              (shape: [B, 64], dtype: float32)
        """
        freqs = self.RFF_freq
        table = 2 * np.pi * sigma * freqs
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        use_norm=True,
        num_dils=6,
        bias=False,
        kernel_size=(5, 3),
        emb_dim=512,
        proj_place="before",  # using 'after' in the decoder out blocks
        init=None,
        init_zero=None,
    ):
        super().__init__()

        if emb_dim == 0:
            self.no_emb = True
        else:
            self.no_emb = False

        self.bias = bias
        self.use_norm = use_norm
        self.num_dils = num_dils
        self.proj_place = proj_place

        if self.proj_place == "before":
            # dim_out is the block dimension
            N = dim_out
        else:
            # dim in is the block dimension
            N = dim
            self.proj_out = (
                Conv2d(N, dim_out, bias=bias, **init) if N != dim_out else nn.Identity()
            )  # linear projection

        self.res_conv = (
            Conv2d(dim, dim_out, bias=bias, **init) if dim != dim_out else nn.Identity()
        )  # linear projection
        self.proj_in = (
            Conv2d(dim, N, bias=bias, **init) if dim != N else nn.Identity()
        )  # linear projection

        self.H = nn.ModuleList()
        self.affine = nn.ModuleList()
        self.gate = nn.ModuleList()
        if self.use_norm:
            self.norm = nn.ModuleList()

        for i in range(self.num_dils):

            if self.use_norm:
                self.norm.append(BiasFreeGroupNorm(N, 8))

            if not self.no_emb:
                self.affine.append(Linear(emb_dim, N, **init))
                self.gate.append(Linear(emb_dim, N, **init_zero))
            # freq convolution (dilated)
            self.H.append(
                Conv2d(N, N, kernel=kernel_size, dilation=(2**i, 1), bias=bias, **init)
            )

    def forward(self, input_x, sigma):

        x = input_x

        x = self.proj_in(x)

        if self.no_emb:
            for norm, conv in zip(self.norm, self.H):
                x0 = x
                if self.use_norm:
                    x = norm(x)
                x = (x0 + conv(F.gelu(x))) / (2**0.5)
        else:
            for norm, affine, gate, conv in zip(
                self.norm, self.affine, self.gate, self.H
            ):
                x0 = x
                if self.use_norm:
                    x = norm(x)
                gamma = affine(sigma)
                scale = gate(sigma)

                x = x * (gamma.unsqueeze(2).unsqueeze(3) + 1)  # no bias

                x = (x0 + conv(F.gelu(x)) * scale.unsqueeze(2).unsqueeze(3)) / (2**0.5)

        # one residual connection here after the dilated convolutions

        if self.proj_place == "after":
            x = self.proj_out(x)

        x = (x + self.res_conv(input_x)) / (2**0.5)

        return x


_kernels = {
    "linear": [1 / 8, 3 / 8, 3 / 8, 1 / 8],
    "cubic": [
        -0.01171875,
        -0.03515625,
        0.11328125,
        0.43359375,
        0.43359375,
        0.11328125,
        -0.03515625,
        -0.01171875,
    ],
}


class UpDownResample(nn.Module):
    def __init__(
        self,
        up=False,
        down=False,
        mode_resample="T",  # T for time, F for freq, TF for both
        resample_filter="cubic",
        pad_mode="reflect",
    ):
        super().__init__()
        assert not (up and down)  # you cannot upsample and downsample at the same time
        assert up or down  # you must upsample or downsample
        self.down = down
        self.up = up
        if up or down:
            # upsample block
            self.pad_mode = pad_mode  # I think reflect is a goof choice for padding
            self.mode_resample = mode_resample
            if mode_resample == "T":
                kernel_1d = torch.tensor(_kernels[resample_filter], dtype=torch.float32)
            elif mode_resample == "F":
                kernel_1d = torch.tensor(_kernels[resample_filter], dtype=torch.float32)
            else:
                raise NotImplementedError("Only time upsampling is implemented")
            self.pad = kernel_1d.shape[0] // 2 - 1
            self.register_buffer("kernel", kernel_1d)

    def forward(self, x):
        shapeorig = x.shape
        x = x.view(-1, x.shape[-2], x.shape[-1])
        if self.mode_resample == "F":
            x = x.permute(0, 2, 1)

        if self.down:
            x = F.pad(x, (self.pad,) * 2, self.pad_mode)
        elif self.up:
            x = F.pad(x, ((self.pad + 1) // 2,) * 2, self.pad_mode)

        weight = x.new_zeros([x.shape[1], x.shape[1], self.kernel.shape[0]])
        indices = torch.arange(x.shape[1], device=x.device)

        weight[indices, indices] = self.kernel.to(weight)
        if self.down:
            x_out = F.conv1d(x, weight, stride=2)
        elif self.up:
            x_out = F.conv_transpose1d(x, weight, stride=2, padding=self.pad * 2 + 1)

        if self.mode_resample == "F":
            x_out = x_out.permute(0, 2, 1).contiguous()
            return x_out.view(shapeorig[0], -1, x_out.shape[-2], shapeorig[-1])
        else:
            return x_out.view(shapeorig[0], -1, shapeorig[2], x_out.shape[-1])


class STFTbackbone(nn.Module):
    """
    Main U-Net model based on the STFT
    """

    def __init__(
        self,
        stft_args=None,
        depth=8,
        emb_dim=256,
        use_norm=True,
        Ns=None,
        Ss=None,
        num_dils=None,
        bottleneck_type=None,
        num_bottleneck_layers=None,
        device="cuda",
        time_conditional=False,
        param_conditional=False,
        num_cond_params=2,
        output_channels=1,
    ):
        """
        Args:
            args (dictionary): hydra dictionary
            device: torch device ("cuda" or "cpu")
        """
        super(STFTbackbone, self).__init__()
        self.stft_args = stft_args
        self.win_size = stft_args.win_length
        self.hop_size = stft_args.hop_length

        self.time_conditional = time_conditional
        self.param_conditional = param_conditional
        self.num_cond_params = num_cond_params
        self.depth = depth

        init = dict(init_mode="kaiming_uniform", init_weight=np.sqrt(1 / 3))
        init_zero = dict(init_mode="kaiming_uniform", init_weight=1e-7)

        self.emb_dim = emb_dim
        self.total_emb_dim = 0
        if self.time_conditional:
            self.embedding = RFF_MLP_Block(emb_dim=emb_dim, inputs=1, init=init)
            self.total_emb_dim += emb_dim
        if self.param_conditional:
            self.embedding_param = RFF_MLP_Block(
                emb_dim=emb_dim, inputs=num_cond_params, init=init
            )
            self.total_emb_dim += emb_dim

        self.use_norm = use_norm

        self.device = device

        Nin = 2

        Nout = 2 * output_channels

        # Encoder
        self.Ns = Ns
        self.Ss = Ss

        self.num_dils = num_dils

        self.downsamplerT = UpDownResample(down=True, mode_resample="T")
        self.downsamplerF = UpDownResample(down=True, mode_resample="F")
        self.upsamplerT = UpDownResample(up=True, mode_resample="T")
        self.upsamplerF = UpDownResample(up=True, mode_resample="F")

        self.downs = nn.ModuleList([])
        self.middle = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        self.init_block = ResnetBlock(
            Nin,
            self.Ns[0],
            self.use_norm,
            num_dils=1,
            bias=False,
            kernel_size=(1, 1),
            emb_dim=self.total_emb_dim,
            init=init,
            init_zero=init_zero,
        )
        self.out_block = ResnetBlock(
            self.Ns[0],
            Nout,
            use_norm=self.use_norm,
            num_dils=1,
            bias=False,
            kernel_size=(1, 1),
            proj_place="after",
            emb_dim=self.total_emb_dim,
            init=init,
            init_zero=init_zero,
        )

        for i in range(self.depth):
            if i == 0:
                dim_in = self.Ns[i]
                dim_out = self.Ns[i]
            else:
                dim_in = self.Ns[i - 1]
                dim_out = self.Ns[i]

            self.downs.append(
                nn.ModuleList(
                    [
                        ResnetBlock(
                            dim_in,
                            dim_out,
                            self.use_norm,
                            num_dils=self.num_dils[i],
                            bias=False,
                            emb_dim=self.total_emb_dim,
                            init=init,
                            init_zero=init_zero,
                        ),
                    ]
                )
            )

        self.bottleneck_type = bottleneck_type
        self.num_bottleneck_layers = num_bottleneck_layers
        if self.bottleneck_type == "res_dil_convs":
            for i in range(self.num_bottleneck_layers):

                self.middle.append(
                    nn.ModuleList(
                        [
                            ResnetBlock(
                                self.Ns[-1],
                                self.Ns[-1],
                                self.use_norm,
                                num_dils=self.num_dils[-1],
                                bias=False,
                                emb_dim=self.total_emb_dim,
                                init=init,
                                init_zero=init_zero,
                            )
                        ]
                    )
                )
        else:
            raise NotImplementedError("bottleneck type not implemented")

        for i in range(self.depth - 1, -1, -1):

            if i == 0:
                dim_in = self.Ns[i] * 2
                dim_out = self.Ns[i]
            else:
                dim_in = self.Ns[i] * 2
                dim_out = self.Ns[i - 1]

            self.ups.append(
                nn.ModuleList(
                    [
                        ResnetBlock(
                            dim_in,
                            dim_out,
                            use_norm=self.use_norm,
                            num_dils=self.num_dils[i],
                            bias=False,
                            emb_dim=self.total_emb_dim,
                            init=init,
                            init_zero=init_zero,
                        ),
                    ]
                )
            )

    def forward_backbone(self, inputs, time_cond=None, param_cond=None):
        """
        Args:
            inputs (Tensor):  Input signal in frequency-domsin, shape (B,C,F,T)
            sigma (Tensor): noise levels,  shape (B,1)
        Returns:
            pred (Tensor): predicted signal in time-domain, shape (B,C,F,T)
        """
        # apply RFF embedding+MLP of the noise level
        emb = None
        if self.time_conditional:

            time_cond = time_cond.unsqueeze(-1)
            time_cond = self.embedding(time_cond)
            if not self.param_conditional:
                emb = time_cond

        if self.param_conditional:
            param_cond = param_cond
            param_cond = self.embedding_param(param_cond)
            if not self.time_conditional:
                emb = param_cond
            else:
                emb = torch.cat((time_cond, param_cond), dim=1)

        hs = []

        X = self.init_block(inputs, emb)

        for i, modules in enumerate(self.downs):
            (ResBlock,) = modules

            X = ResBlock(X, emb)
            hs.append(X)

            # downsample the main signal path
            # we do not need to downsample in the inner layer
            if i < len(self.downs) - 1:
                # no downsampling in the last layer
                X = self.downsamplerT(X)
                X = self.downsamplerF(X)

        # middle layers
        if self.bottleneck_type == "res_dil_convs":
            for i in range(self.num_bottleneck_layers):
                (ResBlock,) = self.middle[i]
                X = ResBlock(X, emb)

        for i, modules in enumerate(self.ups):
            j = len(self.ups) - i - 1

            (ResBlock,) = modules

            skip = hs.pop()
            # print("skip", skip.shape)
            X = torch.cat((X, skip), dim=1)
            X = ResBlock(X, emb)
            if j > 0:
                # no upsampling in the first layer
                X = self.upsamplerT(X)  # call contiguous() here?
                X = self.upsamplerF(X)  # call contiguous() here?

        X = self.out_block(X, emb)

        return X

    def do_stft(self, x):
        """
        x shape: (batch, C, time)
        """
        window = torch.hamming_window(window_length=self.win_size, device=x.device)

        x = torch.cat(
            (
                x,
                torch.zeros(
                    (x.shape[0], x.shape[1], self.win_size - 1), device=x.device
                ),
            ),
            -1,
        )
        B, C, T = x.shape
        x = x.view(-1, x.shape[-1])
        stft_signal = torch.stft(
            x,
            self.win_size,
            hop_length=self.hop_size,
            window=window,
            center=False,
            return_complex=True,
        )
        stft_signal = torch.view_as_real(stft_signal)

        stft_signal = stft_signal.view(B, C, *stft_signal.shape[1:])
        # shape (batch, C, freq, time, 2)

        return stft_signal

    def do_istft(self, x):
        """
        x shape: (batch, C, freq, time, 2)
        """
        B, C, F, T, _ = x.shape
        x = torch.view_as_complex(x)
        window = torch.hamming_window(
            window_length=self.win_size, device=x.device
        )  # this is slow! consider optimizing
        x = einops.rearrange(x, "b c f t -> (b c) f t ")
        pred_time = torch.istft(
            x,
            self.win_size,
            hop_length=self.hop_size,
            window=window,
            center=False,
            return_complex=False,
        )
        pred_time = einops.rearrange(pred_time, "(b c) t -> b c t", b=B)
        return pred_time

    def forward(self, x, time_cond=None, cond=None):
        B, C, T = x.shape
        # apply stft
        x = self.do_stft(x)

        x = einops.rearrange(x, "b c f t ri -> b (c ri) f t")
        x = self.forward_backbone(x, time_cond, cond)
        # apply istft
        x = einops.rearrange(x, " b (c ri) f t -> b c f t ri", ri=2)
        x = x.contiguous()
        x = self.do_istft(x)
        x = x[:, :, :T]
        return x
