from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class Mamba2(nn.Module):
    """
    Pure-PyTorch Mamba-2 replacement for mamba_ssm.Mamba2.

    Goals:
      - Same parameter names/layout as mamba_ssm.Mamba2
      - No Triton / causal-conv1d dependency
      - Works as a direct replacement inside MambaClassifier
      - Provides a stateful `step()` suitable for ONNX export
      - `forward()` reproduces the normal Mamba2 sequence API

    The recurrence is implemented directly from the Mamba-2 inference
    equations rather than the fused Triton kernels.
    """

    def __init__(
            self,
            d_model: int,
            d_state: int = 128,
            d_conv: int = 4,
            expand: int = 2,
            headdim: int = 64,
            d_ssm: Optional[int] = None,
            ngroups: int = 1,
            D_has_hdim: bool = False,
            rmsnorm: bool = True,
            norm_before_gate: bool = False,
            bias: bool = False,
            conv_bias: bool = True,
            **_: object,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.headdim = headdim

        self.d_inner = expand * d_model
        self.d_ssm = self.d_inner if d_ssm is None else d_ssm

        if self.d_ssm % headdim != 0:
            raise ValueError(
                f'd_ssm={self.d_ssm} must be divisible by headdim={headdim}'
            )

        if self.d_ssm % ngroups != 0:
            raise ValueError(
                f'd_ssm={self.d_ssm} must be divisible by ngroups={ngroups}'
            )

        if self.d_state <= 0:
            raise ValueError('d_state must be > 0')

        self.ngroups = ngroups
        self.nheads = self.d_ssm // self.headdim
        self.D_has_hdim = D_has_hdim
        self.rmsnorm = rmsnorm
        self.norm_before_gate = norm_before_gate

        # Same projection layout as official Mamba2:
        # [z, x, B, C, dt]
        self.d_mlp = (
                self.d_inner - self.d_ssm
        )

        self.d_in_proj = (
                2 * self.d_inner
                + 2 * self.ngroups * self.d_state
                + self.nheads
        )

        self.in_proj = nn.Linear(
            self.d_model,
            self.d_in_proj,
            bias=bias,
        )

        self.conv_dim = (
                self.d_ssm
                + 2 * self.ngroups * self.d_state
        )

        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size=self.d_conv,
            groups=self.conv_dim,
            padding=self.d_conv - 1,
            bias=conv_bias,
        )

        self.act = nn.SiLU()

        self.dt_bias = nn.Parameter(
            torch.empty(self.nheads)
        )

        self.A_log = nn.Parameter(
            torch.empty(self.nheads)
        )

        if self.D_has_hdim:
            self.D = nn.Parameter(
                torch.ones(self.d_ssm)
            )
        else:
            self.D = nn.Parameter(
                torch.ones(self.nheads)
            )

        if self.rmsnorm:
            self.norm = _RMSNormGated(
                self.d_ssm,
                eps=1e-5,
                norm_before_gate=self.norm_before_gate,
            )

        self.out_proj = nn.Linear(
            self.d_inner,
            self.d_model,
            bias=bias,
        )

        # This module is an inference/export implementation.
        # The parameters are expected to be loaded from a trained Mamba2.
        self._reset_non_checkpoint_parameters()

    def _reset_non_checkpoint_parameters(self) -> None:
        """
        Initialize only when this module is constructed without a checkpoint.
        Loading a real Mamba2 state_dict replaces these values.
        """
        nn.init.zeros_(self.dt_bias)
        nn.init.zeros_(self.A_log)

    # State allocation

    def allocate_inference_cache(
            self,
            batch_size: int,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None,
    ) -> tuple[Tensor, Tensor]:
        if dtype is None:
            dtype = self.in_proj.weight.dtype

        if device is None:
            device = self.in_proj.weight.device

        conv_state = torch.zeros(
            batch_size,
            self.conv_dim,
            self.d_conv,
            dtype=dtype,
            device=device,
        )

        ssm_state = torch.zeros(
            batch_size,
            self.nheads,
            self.headdim,
            self.d_state,
            dtype=dtype,
            device=device,
        )

        return conv_state, ssm_state

    # Pure PyTorch gated RMSNorm

    def _rms_norm(
            self,
            x: Tensor,
    ) -> Tensor:
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)

        x_norm = x.float() * torch.rsqrt(
            variance + self.norm_eps
        )

        return x_norm.to(x.dtype) * self.norm.to(x.dtype)

    def _norm_gated(
            self,
            y: Tensor,
            z: Tensor,
    ) -> Tensor:
        return self.norm(y, z)

    # One-token recurrent step

    def step(
            self,
            hidden_states: Tensor,
            conv_state: Tensor,
            ssm_state: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        One autoregressive Mamba-2 step.

        Inputs:
            hidden_states:
                [batch, 1, d_model]

            conv_state:
                [batch, conv_dim, d_conv]

            ssm_state:
                [batch, nheads, headdim, d_state]

        Returns:
            output:
                [batch, 1, d_model]

            new_conv_state:
                [batch, conv_dim, d_conv]

            new_ssm_state:
                [batch, nheads, headdim, d_state]
        """

        if hidden_states.ndim != 3:
            raise ValueError(
                f'hidden_states must be [B, 1, D], got {tuple(hidden_states.shape)}'
            )

        if hidden_states.shape[1] != 1:
            raise ValueError(
                "Mamba2ONNXStep.step() expects exactly one token."
            )

        x_in = hidden_states[:, 0, :]

        # Projection

        zxbcdt = self.in_proj(x_in)

        # Official Mamba2 projection order:
        # [z0, x0, z, xBC, dt]
        #
        # For the default Mamba2 configuration used by your classifier:
        #   d_ssm == d_inner
        #   => d_mlp == 0
        #
        # Keeping the generic split also supports d_ssm < d_inner.
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [
                self.d_mlp,
                self.d_mlp,
                self.d_ssm,
                self.conv_dim,
                self.nheads,
            ],
            dim=-1,
        )

        # Depthwise causal convolution

        new_conv_state = torch.cat(
            (
                conv_state[:, :, 1:],
                xBC.unsqueeze(-1),
            ),
            dim=-1,
        )

        conv_weight = self.conv1d.weight[:, 0, :]

        xBC = (
                new_conv_state * conv_weight.unsqueeze(0)
        ).sum(dim=-1)

        if self.conv1d.bias is not None:
            xBC = xBC + self.conv1d.bias

        xBC = self.act(xBC)

        x, B, C = torch.split(
            xBC,
            [
                self.d_ssm,
                self.ngroups * self.d_state,
                self.ngroups * self.d_state,
            ],
            dim=-1,
        )

        # SSM
        # A is stored as log(A), actual state matrix is -exp(A_log).
        A = -torch.exp(self.A_log.float())

        # dt softplus parameterization
        dt = F.softplus(
            dt + self.dt_bias.to(dtype=dt.dtype)
        )

        # [B, nheads]
        dA = torch.exp(
            dt * A.to(dtype=dt.dtype)
        )

        # x: [B, nheads, headdim]
        x = x.reshape(
            -1,
            self.nheads,
            self.headdim,
        )

        # B/C:
        # [B, ngroups, d_state]
        B = B.reshape(
            -1,
            self.ngroups,
            self.d_state,
        )

        C = C.reshape(
            -1,
            self.ngroups,
            self.d_state,
        )

        # Each group is shared by a contiguous set of heads.
        heads_per_group = self.nheads // self.ngroups

        if heads_per_group == 1:
            B = B
            C = C
        else:
            B = B.repeat_interleave(
                heads_per_group,
                dim=1,
            )
            C = C.repeat_interleave(
                heads_per_group,
                dim=1,
            )

        # State update
        #
        # s[t] =
        #   exp(dt*A) * s[t-1]
        #   + dt * B * x

        dA_state = dA.unsqueeze(-1).unsqueeze(-1)

        dBx = (
                dt.unsqueeze(-1).unsqueeze(-1)
                * B.unsqueeze(2)
                * x.unsqueeze(-1)
        )

        new_ssm_state = (
                ssm_state * dA_state
                + dBx
        )

        # Readout
        y = (
                new_ssm_state
                * C.unsqueeze(2)
        ).sum(dim=-1)

        # D skip connection
        if self.D_has_hdim:
            D = self.D.reshape(
                self.nheads,
                self.headdim,
            )
        else:
            D = self.D.reshape(
                self.nheads,
                1,
            )

        y = y + D.to(y.dtype) * x

        # [B, d_ssm]
        y = y.reshape(
            -1,
            self.d_ssm,
        )

        # Gated RMSNorm
        if self.rmsnorm:
            y = self._norm_gated(y, z)
        else:
            y = y * self.act(z)

        # Optional gated MLP branch used when d_ssm < d_inner
        if self.d_mlp > 0:
            y = torch.cat(
                (
                    self.act(z0) * x0,
                    y,
                ),
                dim=-1,
            )

        # Output projection
        output = self.out_proj(y)

        return (
            output.unsqueeze(1),
            new_conv_state,
            new_ssm_state,
        )

    def forward(
            self,
            u: Tensor,
    ) -> Tensor:
        """
        Same external API as Mamba2:

            input : [B, L, d_model]
            output: [B, L, d_model]

        The sequence is evaluated recurrently using step().
        """

        if u.ndim != 3:
            raise ValueError(
                f'Expected [B, L, D], got {tuple(u.shape)}'
            )

        batch_size = u.shape[0]

        conv_state, ssm_state = self.allocate_inference_cache(
            batch_size=batch_size,
            dtype=u.dtype,
            device=u.device,
        )

        outputs = []

        seq_len = u.shape[1]

        for t in range(seq_len):
            token = u[:, t:t + 1, :]

            token_out, conv_state, ssm_state = self.step(
                token,
                conv_state,
                ssm_state,
            )

            outputs.append(token_out)

        return torch.cat(outputs, dim=1)


class _RMSNormGated(nn.Module):
    """
    Pure-PyTorch equivalent of the RMSNormGated used by mamba_ssm.Mamba2.

    State dict:
        norm.weight
    """

    def __init__(
            self,
            dim: int,
            eps: float = 1e-5,
            norm_before_gate: bool = False,
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
        self.norm_before_gate = norm_before_gate

    def forward(
            self,
            x: Tensor,
            z: Tensor | None = None,
    ) -> Tensor:

        if self.norm_before_gate:
            x = self._rms_norm(x)

            if z is not None:
                x = x * F.silu(z)

            return x

        if z is not None:
            x = x * F.silu(z)

        return self._rms_norm(x)

    def _rms_norm(self, x: Tensor) -> Tensor:
        # Accumulate normalization in FP32, matching the numerical
        # behavior expected from the Mamba RMSNorm implementation.
        x_float = x.float()

        variance = x_float.pow(2).mean(
            dim=-1,
            keepdim=True,
        )

        x = x_float * torch.rsqrt(
            variance + self.eps
        )

        return (
                x.to(dtype=self.weight.dtype)
                * self.weight
        )
