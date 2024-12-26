from collections.abc import Sequence
from typing import NamedTuple, Optional

import torch
from einops import einsum, rearrange, repeat


class RetentionState(NamedTuple):
    hidden_state: torch.Tensor
    time: torch.Tensor


def empty_state(
    dtype: torch.dtype = torch.float, device: str | torch.device = "cpu"
) -> RetentionState:
    return RetentionState(
        torch.zeros(1, 1, 1, 1, dtype=dtype, device=device),
        torch.tensor(0, dtype=dtype, device=device),
    )


def detach_state(state: RetentionState) -> RetentionState:
    return RetentionState(state.hidden_state.detach(), state.time.detach())


class MultiScaleRetention(torch.nn.Module):
    def __init__(
        self,
        in_dims: int,
        num_heads: int = 1,
        qk_dims: Optional[int] = None,
        v_dims: Optional[int] = None,
        gammas: Optional[torch.Tensor] = None,
        thetas: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.in_dims = in_dims

        if qk_dims is None:
            qk_dims = in_dims
        if v_dims is None:
            v_dims = qk_dims
        if qk_dims % 2:
            raise ValueError("`qk_dims` must be even")
        if gammas is None:
            gammas = 1 - 2 ** (-5 - torch.arange(num_heads, dtype=torch.float))
        else:
            assert len(gammas) == num_heads
        if thetas is None:
            thetas = 10_000 ** (
                -2 * torch.arange(qk_dims // 2, dtype=torch.float) / qk_dims
            )
        else:
            assert len(thetas) == qk_dims // 2

        self.register_buffer("gammas", rearrange(gammas, "h -> h 1 1 1").contiguous())
        self.gammas: torch.Tensor
        self.register_buffer("thetas", thetas.contiguous())
        self.thetas: torch.Tensor

        shift = in_dims**-0.5
        scale = 2 * shift
        self.w_k = torch.nn.Parameter(
            torch.rand(num_heads, in_dims, qk_dims) * scale - shift
        )
        self.w_q = torch.nn.Parameter(
            torch.rand(num_heads, in_dims, qk_dims) * scale - shift
        )
        self.w_v = torch.nn.Parameter(
            torch.rand(num_heads, in_dims, v_dims) * scale - shift
        )

        self.out_dims = num_heads * v_dims
        self.group_norm = torch.nn.GroupNorm(num_heads, self.out_dims)

        self.w_g = torch.nn.Parameter(
            torch.rand(in_dims, self.out_dims) * scale - shift
        )
        shift = (self.out_dims) ** -0.5
        scale = 2 * shift
        self.w_o = torch.nn.Parameter(
            torch.rand(self.out_dims, self.out_dims) * scale - shift
        )

    def forward(
        self,
        sequence: torch.Tensor,
        times: torch.Tensor,
        state: Optional[RetentionState] = None,
        mode: str = "parallel",
    ) -> torch.Tensor | tuple[torch.Tensor, RetentionState]:
        match mode:
            case "parallel":
                return self.forward_parallel(sequence, times)
            case "recurrent":
                return self.forward_recurrent(sequence, times, state)
            case "chunkwise":
                return self.forward_chunkwise(sequence, times, state)
            case _:
                raise ValueError(f"Invalid mode: `{mode}`")

    def forward_parallel(
        self, sequence: torch.Tensor, times: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if times is None:
            times = repeat(
                torch.arange(
                    sequence.size(1), device=self.w_q.device, dtype=self.w_k.dtype
                ),
                "i -> b i",
                b=sequence.size(0),
            )

        q, k, v = self.get_qkv(sequence, times)

        ret = einsum(q, k, "h b l1 qk, h b l2 qk -> h b l1 l2") * self.get_d(times)
        retention = ret @ v

        return self.norm(sequence, retention)

    def forward_recurrent(
        self,
        in_vec: torch.Tensor,
        time: torch.Tensor,
        state: Optional[RetentionState] = None,
    ) -> tuple[torch.Tensor, RetentionState]:
        if state is None:
            state = empty_state(self.w_q.dtype, self.w_v.device)

        if time.dim() == 0:
            time = repeat(time, " -> b", b=in_vec.size(0))

        in_vec = rearrange(in_vec, "b dim -> b 1 dim")

        q, k, v = self.get_qkv(in_vec, rearrange(time, "b -> b 1"))

        new_state = self.gammas ** rearrange(
            time - state.time, "b -> b 1 1"
        ) * state.hidden_state + einsum(k, v, "h b l qk, h b l v -> h b qk v")
        retention = einsum(q, new_state, "h b l qk, h b qk v -> h b l v")

        return rearrange(
            self.norm(in_vec, retention), "b 1 dim -> b dim"
        ), RetentionState(new_state, time)

    def forward_chunkwise(
        self,
        chunk: torch.Tensor,
        times: torch.Tensor,
        state: Optional[RetentionState] = None,
    ) -> tuple[torch.Tensor, RetentionState]:
        if state is None:
            state = empty_state(self.w_q.dtype, self.w_v.device)

        if times.dim() == 0:
            times = (
                repeat(
                    torch.arange(
                        chunk.size(1), dtype=self.w_q.dtype, device=self.w_k.device
                    ),
                    "l -> b l",
                    b=chunk.size(0),
                )
                + times
            )

        deltas = times - state.time
        chunk_length = times[:, -1:] - state.time

        q, k, v = self.get_qkv(chunk, times)

        v_decay = self.gammas ** rearrange(chunk_length - deltas, "b l -> b l 1")
        state_decay = self.gammas ** rearrange(chunk_length, "b l -> b l 1")
        new_state = (
            einsum(k, v * v_decay, "h b l qk, h b l v -> h b qk v")
            + state_decay * state.hidden_state
        )

        intra_chunk = einsum(q, k, "h b l1 qk, h b l2 qk -> h b l1 l2") * self.get_d(
            deltas
        )
        intra_chunk = intra_chunk @ v

        inter_decay = self.gammas ** rearrange(deltas, "b l -> b l 1")
        inter_chunk = (
            einsum(q, state.hidden_state, "h b l qk, h b qk v -> h b l v") * inter_decay
        )

        retention = intra_chunk + inter_chunk

        return self.norm(chunk, retention), RetentionState(new_state, times[:, -1:])

    def norm(self, sequence: torch.Tensor, ret: torch.Tensor) -> torch.Tensor:
        batch_size = sequence.size(0)
        ret = self.group_norm(rearrange(ret, "h b l v -> (b l) (h v)"))
        ret = rearrange(ret, "(b l) dim -> b l dim", b=batch_size)

        return (swish(sequence @ self.w_g) * ret) @ self.w_o

    def get_rotor(self, times: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        angles = einsum(times, self.thetas, "b t, dim -> b t dim")
        cos = rearrange([torch.cos(angles)] * 2, "n b t dim -> b t (dim n)")
        sin = rearrange([torch.sin(angles)] * 2, "n b t dim -> b t (dim n)")

        return (cos, sin)

    def get_d(self, times: torch.Tensor) -> torch.Tensor:
        n = rearrange(times, "b t -> b t 1")
        m = rearrange(times, "b t -> b 1 t")

        mat = n - m
        d = (self.gammas**mat) * (mat >= 0)
        d[d != d] = 0

        return d

    def get_qkv(
        self, sequence: torch.Tensor, times: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = einsum(sequence, self.w_q, "b l dim, h dim qk -> h b l qk")
        k = einsum(sequence, self.w_k, "b l dim, h dim qk -> h b l qk")
        v = einsum(sequence, self.w_v, "b l dim, h dim v -> h b l v")

        cos, sin = self.get_rotor(times)

        q = cos * q + sin * rotate_by_i(q)
        k = cos * k - sin * rotate_by_i(k)

        return q, k, v


def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.nn.functional.sigmoid(x)


def rotate_by_i(x: torch.Tensor) -> torch.Tensor:
    real = x[..., ::2]
    imag = x[..., 1::2]
    return rearrange([-imag, real], "i ... dim -> ... (dim i)")


class RetNet(torch.nn.Module):
    def __init__(
        self,
        in_dims: int,
        layers: int,
        num_heads: int,
        ffn_dims: Optional[Sequence[int] | int] = None,
    ):
        super().__init__()

        if in_dims % num_heads:
            raise ValueError("`in_dims` must be divisible by `num_heads`")

        if ffn_dims is None:
            ffn_dims = in_dims
        if isinstance(ffn_dims, int):
            ffn_dims = [ffn_dims] * layers
        if len(ffn_dims) != layers:
            raise ValueError(
                "Specified number of `ffn_dims` must match the number of layers"
            )

        qkv_dims = in_dims // num_heads
        self.retention_layers = torch.nn.ModuleList(
            MultiScaleRetention(in_dims, num_heads, qkv_dims) for _ in range(layers)
        )

        self.in_norms = torch.nn.ModuleList(
            torch.nn.LayerNorm(in_dims, elementwise_affine=False) for _ in range(layers)
        )
        self.out_norms = torch.nn.ModuleList(
            torch.nn.LayerNorm(in_dims, elementwise_affine=False) for _ in range(layers)
        )

        in_shift = in_dims**-0.5
        in_scale = in_shift * 2

        ffn0 = []
        ffn1 = []
        for out in ffn_dims:
            out_shift = out**-0.5
            out_scale = out_shift * 2

            ffn0.append(torch.rand(in_dims, out) * in_scale - in_shift)
            ffn1.append(torch.rand(out, in_dims) * out_scale - out_scale)

        self.ffn0 = torch.nn.ParameterList(ffn0)
        self.ffn1 = torch.nn.ParameterList(ffn1)

    def forward(
        self,
        sequence: torch.Tensor,
        times: Optional[torch.Tensor] = None,
        states: Optional[Sequence[RetentionState]] = None,
        mode: str = "parallel",
    ) -> torch.Tensor | tuple[torch.Tensor, list[RetentionState]]:
        x = sequence
        match mode:
            case "parallel":
                for layer, in_norm, out_norm, ffn0, ffn1 in zip(
                    self.retention_layers,
                    self.in_norms,
                    self.out_norms,
                    self.ffn0,
                    self.ffn1,
                    strict=True,
                ):
                    y = layer(in_norm(x), times) + x
                    x = (torch.nn.functional.gelu(out_norm(y) @ ffn0) @ ffn1) + y
                return x
            case "recurrent" | "chunkwise":
                if states is None:
                    states = [
                        empty_state(self.ffn0[0].dtype, self.ffn1[0].device)
                        for _ in range(len(self.retention_layers))
                    ]
                new_states = []
                for layer, in_norm, out_norm, ffn0, ffn1, state in zip(
                    self.retention_layers,
                    self.in_norms,
                    self.out_norms,
                    self.ffn0,
                    self.ffn1,
                    states,
                    strict=True,
                ):
                    o, new_state = layer(in_norm(x), times, state, mode)
                    new_states.append(new_state)
                    y = o + x
                    x = (torch.nn.functional.gelu(out_norm(y) @ ffn0) @ ffn1) + y

                return x, new_states
            case _:
                raise ValueError(f"Invalid mode: `{mode}`")


# tests
if __name__ == "__main__":
    import time

    dev = torch.device("cuda")
    data = torch.float

    hidden = 64
    test = RetNet(hidden, 3, 16)
    test.to(device=dev, dtype=data)
    test.eval()
    tol = 1e-3

    with dev:
        torch.set_default_dtype(data)
        print("computing...")

        seq = torch.rand(10, 1000, hidden)
        times = torch.cumsum(torch.rand(10, 1000) * 10, dim=-1)
        assert torch.all(times >= 0)
        assert torch.all((times[:, 1:] - times[:, :-1]) > 0)

        start = time.perf_counter()
        out = test(seq, times)
        end = time.perf_counter()
        print(out.shape)
        print(end - start)

        l = torch.mean(out)
        l.backward()
        print("backwards for retnet is working")

        outs = []
        l = 128
        state = None

        for i in range(l):
            out_vec, state = test(seq[:, i, :], times[:, i], state, mode="recurrent")
            outs.append(out_vec)

        rec = rearrange(outs, "l b dim -> b l dim").detach()
        true = out[:, :l, :].detach()

        # checking tolerance since exact equality is impossible due to floating point error
        assert torch.all(torch.abs(rec - true) < tol)
        print("asserted recurrent equivalent to parallel")

        chunks = 16

        assert not l % chunks
        size = l // chunks
        state = None

        print("chunking...")
        outs = []

        for i in range(chunks):
            out_chunk, state = test(
                seq[:, size * i : size * (i + 1), :],
                times[:, size * i : size * (i + 1)],
                state,
                mode="chunkwise",
            )
            outs.append(out_chunk)

        out_chunk, state = test(
            seq[:, l : l + 21], times[:, l : l + 21], state, mode="chunkwise"
        )
        outs.append(out_chunk)

        outs = torch.cat(outs, dim=1)
        assert torch.all(torch.abs(out[:, : l + 21] - outs) < tol)

        print("asserted chunkwise equal to parallel")

    print("\nAll tests passed!\n")
