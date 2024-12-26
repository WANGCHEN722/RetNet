from collections.abc import Sequence
from typing import Iterable, NamedTuple, Optional

import lightning.pytorch as pl
import retention
import torch
from einops import rearrange, repeat
from retention import RetentionState, RetNet

eps = 1e-8
torch.distributions.Distribution.set_default_validate_args(False)


class RetentiveMetaTPPEncoderState(NamedTuple):
    pool: Sequence[RetentionState]
    prev_encoding_pool: torch.Tensor
    num_toks: int


def empty_state(
    batch_size: int,
    encoding_dim: int,
    pool_layers: int,
    squeeze: bool,
    dtype: torch.dtype = torch.float,
    device: str | torch.device = "cpu",
):
    pool = [retention.empty_state(dtype, device) for _ in range(pool_layers)]
    if squeeze:
        prev = torch.zeros((batch_size, encoding_dim), dtype=dtype, device=device)
    else:
        prev = torch.zeros((batch_size, 1, encoding_dim), dtype=dtype, device=device)

    return RetentiveMetaTPPEncoderState(pool, prev, 1)


def detach_state(state: RetentiveMetaTPPEncoderState) -> RetentiveMetaTPPEncoderState:
    pool = [retention.detach_state(t) for t in state.pool]

    return RetentiveMetaTPPEncoderState(
        pool, state.prev_encoding_pool.detach(), state.num_toks
    )


class RetentiveMetaTPPEncoder(torch.nn.Module):
    def __init__(
        self,
        in_dims: int,
        pool_layers: int,
        pool_num_heads: int,
        out_dims: Optional[int] = None,
    ):
        super().__init__()

        if out_dims is None:
            out_dims = in_dims

        self.pool = RetNet(in_dims, pool_layers, pool_num_heads)
        self.to_dims = torch.nn.Sequential(
            torch.nn.Linear(in_dims, out_dims),
            torch.nn.GELU(),
            torch.nn.Linear(out_dims, out_dims),
        )

        self.register_buffer("DATA", torch.zeros(1))
        self.DATA: torch.Tensor

        self.to_pooled = torch.nn.Sequential(
            torch.nn.Linear(out_dims, out_dims),
            torch.nn.GELU(),
            torch.nn.Linear(out_dims, out_dims),
        )

        self.pool_num_layers = pool_layers

    def forward(
        self,
        sequence: torch.Tensor,
        times: torch.Tensor,
        state: Optional[RetentiveMetaTPPEncoderState] = None,
        mode: str = "parallel",
    ) -> (
        tuple[torch.Tensor, torch.distributions.Independent]
        | tuple[
            torch.Tensor, torch.distributions.Independent, RetentiveMetaTPPEncoderState
        ]
    ):
        match mode:
            case "parallel":
                encodings = self.to_dims(self.pool(sequence, times))
                tok_ind = repeat(
                    torch.arange(
                        1,
                        sequence.size(1) + 1,
                        dtype=self.DATA.dtype,
                        device=self.DATA.device,
                    ),
                    "l -> b l 1",
                    b=sequence.size(0),
                )
                avg_pooled = (
                    torch.cumsum(self.delay_encodings(encodings)[:, :-1], dim=1)
                    / tok_ind
                )

                return encodings, self.to_pooled(avg_pooled)
            case "chunkwise":
                if state is None:
                    state = empty_state(
                        sequence.size(0),
                        sequence.size(2),
                        self.pool_num_layers,
                        False,
                        self.DATA.dtype,
                        self.DATA.device,
                    )

                encodings, pool_states = self.pool(
                    sequence, times, state.pool, "chunkwise"
                )
                encodings = self.to_dims(encodings)

                num_toks = state.num_toks + sequence.size(1) + 1
                tok_ind = repeat(
                    torch.arange(
                        state.num_toks,
                        num_toks,
                        dtype=self.DATA.dtype,
                        device=self.DATA.device,
                    ),
                    "l -> b l 1",
                    b=sequence.size(0),
                )
                rescaled = state.num_toks / tok_ind * state.prev_encoding_pool
                avg_pooled = (
                    torch.cumsum(self.delay_encodings(encodings), dim=1) / tok_ind
                    + rescaled
                )

                return (
                    encodings,
                    self.to_pooled(avg_pooled[:, :-1]),
                    RetentiveMetaTPPEncoderState(
                        pool_states, avg_pooled[:, -1:], num_toks
                    ),
                )
            case "recurrent":
                if state is None:
                    state = empty_state(
                        sequence.size(0),
                        sequence.size(1),
                        self.pool_num_layers,
                        True,
                        self.DATA.dtype,
                        self.DATA.device,
                    )

                encodings, pool_states = self.pool(
                    sequence, times, state.pool, "recurrent"
                )
                encodings = self.to_dims(encodings)

                next_tok = state.num_toks + 1
                avg_pooled = (
                    state.num_toks / next_tok * state.prev_encoding_pool
                    + encodings / next_tok
                )

                return (
                    encodings,
                    self.to_pooled(state.prev_encoding_pool),
                    RetentiveMetaTPPEncoderState(pool_states, avg_pooled, next_tok),
                )
            case _:
                raise ValueError(f"Invalid mode: `{mode}`")

    @staticmethod
    def delay_encodings(
        encodings: torch.Tensor, prev_encoding: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if prev_encoding is None:
            s = encodings.size()
            prev_encoding = torch.zeros(
                (s[0], 1) + s[2:], dtype=encodings.dtype, device=encodings.device
            )

        delayed = torch.cat([prev_encoding, encodings], dim=1)

        return delayed


class MuseEmbedding(torch.nn.Module):
    def __init__(
        self,
        action_dims: int,
        velocity_dims: int,
        delta_dims: int,
        out_dims: Optional[int] = None,
    ):
        super().__init__()
        if out_dims is None:
            out_dims = action_dims + velocity_dims + delta_dims

        self.action = torch.nn.Sequential(
            torch.nn.Linear(256, action_dims),
            torch.nn.GELU(),
            torch.nn.Linear(action_dims, action_dims),
            torch.nn.GELU(),
        )
        self.vel = torch.nn.Sequential(
            torch.nn.Linear(128, velocity_dims),
            torch.nn.GELU(),
            torch.nn.Linear(velocity_dims, velocity_dims),
            torch.nn.GELU(),
        )
        self.delta = torch.nn.Sequential(
            torch.nn.Linear(1, delta_dims),
            torch.nn.GELU(),
            torch.nn.Linear(delta_dims, delta_dims),
            torch.nn.GELU(),
        )

        self.combined = torch.nn.Sequential(
            torch.nn.Linear(action_dims + velocity_dims + delta_dims, out_dims),
            torch.nn.GELU(),
            torch.nn.Linear(out_dims, out_dims),
        )

    def forward(
        self, actions: torch.Tensor, velocities: torch.Tensor, times: torch.Tensor
    ):
        act = self.action(actions)
        vel = self.vel(velocities)
        delta = self.delta(rearrange(times, "... -> ... 1"))

        combined = torch.cat([act, vel, delta], dim=-1)

        return self.combined(combined)


class MuseTPPDecoder(torch.nn.Module):
    def __init__(
        self,
        in_dims: int,
        action_dims: int,
        velocity_dims: int,
        velocity_num_distr: int,
        delta_dims: int,
        delta_num_distr: int,
    ):
        super().__init__()

        combined_dims = action_dims + delta_dims + velocity_dims

        self.combined = torch.nn.Sequential(
            torch.nn.Linear(in_dims, combined_dims),
            torch.nn.GELU(),
            torch.nn.Linear(combined_dims, combined_dims),
            torch.nn.GELU(),
        )

        self.delta_dims = delta_dims
        self.vel_dims = velocity_dims
        self.act_dims = action_dims

        self.action = torch.nn.Sequential(
            torch.nn.Linear(action_dims, action_dims),
            torch.nn.GELU(),
            torch.nn.Linear(action_dims, action_dims),
            torch.nn.GELU(),
        )
        self.vel = torch.nn.Sequential(
            torch.nn.Linear(velocity_dims, velocity_dims),
            torch.nn.GELU(),
            torch.nn.Linear(velocity_dims, velocity_dims),
            torch.nn.GELU(),
        )
        self.delta = torch.nn.Sequential(
            torch.nn.Linear(delta_dims, delta_dims),
            torch.nn.GELU(),
            torch.nn.Linear(delta_dims, delta_dims),
            torch.nn.GELU(),
        )

        self.action_probs = torch.nn.Sequential(
            torch.nn.Linear(action_dims, action_dims),
            torch.nn.GELU(),
            torch.nn.Linear(action_dims, 256),
            torch.nn.Sigmoid(),
        )

        self.vel_0_conc = torch.nn.Sequential(
            torch.nn.Linear(velocity_dims, velocity_dims),
            torch.nn.GELU(),
            torch.nn.Linear(velocity_dims, 128 * velocity_num_distr),
            torch.nn.Softplus(),
        )
        self.vel_1_conc = torch.nn.Sequential(
            torch.nn.Linear(velocity_dims, velocity_dims),
            torch.nn.GELU(),
            torch.nn.Linear(velocity_dims, 128 * velocity_num_distr),
            torch.nn.Softplus(),
        )
        self.vel_weights = torch.nn.Sequential(
            torch.nn.Linear(velocity_dims, velocity_dims),
            torch.nn.GELU(),
            torch.nn.Linear(velocity_dims, velocity_num_distr),
        )

        self.delta_mean = torch.nn.Sequential(
            torch.nn.Linear(delta_dims, delta_dims),
            torch.nn.GELU(),
            torch.nn.Linear(delta_dims, delta_num_distr),
        )
        self.delta_stdev = torch.nn.Sequential(
            torch.nn.Linear(delta_dims, delta_dims),
            torch.nn.GELU(),
            torch.nn.Linear(delta_dims, delta_num_distr),
            torch.nn.Softplus(),
        )
        self.delta_weights = torch.nn.Sequential(
            torch.nn.Linear(delta_dims, delta_dims),
            torch.nn.GELU(),
            torch.nn.Linear(delta_dims, delta_num_distr),
        )

    def forward(
        self, encodings: torch.Tensor, latent: torch.Tensor
    ) -> tuple[
        torch.distributions.Independent,
        torch.distributions.Independent,
        torch.distributions.MixtureSameFamily,
    ]:
        features = torch.cat([encodings, latent], dim=-1)

        combined = self.combined(features)
        act_features = self.action(combined[..., : self.act_dims])
        vel_features = self.vel(
            combined[..., self.act_dims : self.act_dims + self.vel_dims]
        )
        delta_features = self.delta(combined[..., self.act_dims + self.vel_dims :])

        action_distr = torch.distributions.Independent(
            torch.distributions.Bernoulli(probs=self.action_probs(act_features)), 1
        )

        vel_0_conc = rearrange(
            self.vel_0_conc(vel_features),
            "... (distr vels) -> ... vels distr",
            vels=128,
        )
        vel_1_conc = rearrange(
            self.vel_1_conc(vel_features),
            "... (distr vels) -> ... vels distr",
            vels=128,
        )
        vel_weights = repeat(
            self.vel_weights(vel_features), "... distr -> ... v distr", v=128
        )
        vel_distr = torch.distributions.Beta(vel_1_conc + eps, vel_0_conc + eps)
        vel_mix_distr = torch.distributions.Independent(
            torch.distributions.MixtureSameFamily(
                torch.distributions.Categorical(logits=vel_weights), vel_distr
            ),
            1,
        )

        delta_means = self.delta_mean(delta_features)
        delta_stdev = self.delta_stdev(delta_features)
        delta_weights = self.delta_weights(delta_features)
        delta_distr = torch.distributions.LogNormal(delta_means, delta_stdev + eps)
        delta_mix_distr = torch.distributions.MixtureSameFamily(
            torch.distributions.Categorical(logits=delta_weights), delta_distr
        )

        return action_distr, vel_mix_distr, delta_mix_distr


class MuseTPP(pl.LightningModule):
    def __init__(
        self,
        embedding: MuseEmbedding,
        encoder: RetentiveMetaTPPEncoder,
        decoder: MuseTPPDecoder,
        training_chunk_length: int = 1 << 64,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["embedding", "encoder", "decoder"])    

        self.chunks = training_chunk_length

        self.embedding = embedding
        self.encoder = encoder
        self.decoder = decoder

        self.automatic_optimization = False

    def forward(
        self,
        actions: torch.Tensor,
        velocities: torch.Tensor,
        times: torch.Tensor,
        state: Optional[RetentiveMetaTPPEncoderState] = None,
        mode: str = "parallel",
    ) -> (
        tuple[
            torch.distributions.Independent,
            torch.distributions.MixtureSameFamily,
            torch.distributions.MixtureSameFamily,
        ]
        | tuple[
            torch.distributions.Independent,
            torch.distributions.MixtureSameFamily,
            torch.distributions.MixtureSameFamily,
            RetentiveMetaTPPEncoderState,
        ]
    ):
        times = times * 10
        embeddings = self.embedding(actions, velocities, times)
        new = self.encoder(embeddings, times, state, mode)
        new_state = None
        match mode:
            case "parallel":
                encodings, pooled = new
            case "recurrent" | "chunkwise":
                encodings, pooled, new_state = new
            case _:
                raise RuntimeError(f"Invalid mode `{mode}`")
        pooled: torch.Tensor
        encodings: torch.Tensor
        new_state: Optional[RetentiveMetaTPPEncoderState]

        act_distr, vel_distr, delta_distr = self.decoder(encodings, pooled)

        if new_state is None:
            return act_distr, vel_distr, delta_distr
        else:
            return act_distr, vel_distr, delta_distr, new_state

    def training_step(self, batch, batch_idx):
        optim = self.optimizers()
        optim.zero_grad()  # type: ignore

        state = None
        prev_batch = zip(*map(self.chunkify, map(lambda v: v[:, :-1], batch)))
        next_batch = zip(*map(self.chunkify, map(lambda v: v[:, 1:], batch)))
        batches = batch[3].size(0)

        for (actions, velocities, times, mask), (
            next_actions,
            next_velocities,
            next_times,
            next_mask,
        ) in zip(prev_batch, next_batch, strict=True):
            times = times * 10
            next_times = next_times * 10
            embeds = self.embedding(actions, velocities, times)
            embeds = embeds * rearrange(mask, " b l -> b l 1")

            encodings, pooled, state = self.encoder(
                embeds, times, state, mode="chunkwise"
            )
            state = detach_state(state)

            act_distr, vel_distr, delta_distr = self.decoder(encodings, pooled)

            act_ll = torch.sum(act_distr.log_prob(next_actions) * next_mask) / batches

            next_velocities = torch.clamp(next_velocities, eps, 1 - eps)
            vel_ll = (
                torch.sum(
                    torch.sum(
                        vel_distr.base_dist.log_prob(next_velocities)
                        * rearrange(next_actions[..., :128], "b l d -> 1 b l d"),
                        dim=3,
                    )
                    * next_mask
                )
                / batches
            )

            next_deltas = torch.clamp(next_times - times, eps)
            delta_ll = (
                torch.sum(delta_distr.log_prob(next_deltas) * next_mask) / batches
            )

            loss = -(act_ll + delta_ll + vel_ll)

            self.manual_backward(loss)
            self.log("NLL", loss, on_step=True, prog_bar=True, reduce_fx="sum")

            self.log_dict(
                {
                    "Action LL": act_ll,
                    "Delta LL": delta_ll,
                    "Velocity LL": vel_ll,
                },
                on_step=True,
                reduce_fx="sum",
            )

        self.clip_gradients(
            optim,  # type: ignore
            gradient_clip_val=0.1,
            gradient_clip_algorithm="norm",
        )
        optim.step()  # type: ignore

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), amsgrad=True)

    def chunkify(self, sequence: torch.Tensor) -> Iterable[torch.Tensor]:
        if sequence.size(1) <= self.chunks:
            yield sequence
        else:
            n = sequence.size(1) // self.chunks
            for i in range(n):
                yield sequence[:, i * self.chunks : self.chunks * (i + 1)]

            if sequence.size(1) % self.chunks:
                yield sequence[:, n * self.chunks :]


# tests
if __name__ == "__main__":
    from torchviz import make_dot

    torch.distributions.Distribution.set_default_validate_args(True)

    import time

    device = torch.device("cuda")
    data = torch.float

    embedding = MuseEmbedding(10, 10, 10)
    encoder = RetentiveMetaTPPEncoder(30, 8, 3)
    test_decoder = MuseTPPDecoder(60, 10, 10, 8, 10, 8)
    test_model = MuseTPP(embedding, encoder, test_decoder, 8)
    test_model.to(device=device, dtype=data)

    with device:
        torch.set_default_dtype(data)

        times = torch.cumsum(torch.rand(10, 500) * 10, dim=-1)
        assert torch.all((times[:, 1:] - times[:, :-1]) > 0)
        assert torch.all(times >= 0)
        seq = torch.rand(10, 500, 30)
        tol = 1e-3

        start = time.perf_counter()
        _, true = encoder(seq, times)
        end = time.perf_counter()

        print(f"Time taken: {end-start}")

        l = 256
        state = None
        outs = []

        for i in range(l):
            _, pool, state = encoder(seq[:, i], times[:, i], state, "recurrent")
            outs.append(pool)

        outs = rearrange(outs, "l b dim -> b l dim")
        assert torch.all(torch.abs(outs - true[:, :l]) < tol)
        print("asserted recurrent equals parallel")

        num_chunks = 8
        assert not l % num_chunks
        chunk_size = l // num_chunks
        state = None
        outs = []

        for i in range(num_chunks):
            _, pd, state = encoder(
                seq[:, chunk_size * i : chunk_size * (i + 1)],
                times[:, chunk_size * i : chunk_size * (i + 1)],
                state,
                "chunkwise",
            )
            outs.append(pd)

        outs = rearrange(outs, "l b c dim -> b (l c) dim")

        assert torch.all(torch.abs(outs - true[:, :l]) < tol)
        print("asserted chunkwise equals parallel")

        act = torch.round(torch.rand(1, 10, 256))
        vel = torch.rand(1, 10, 128)
        d = torch.rand(1, 10)
        times = torch.cumsum(d, dim=-1)

        act_distr, vel_distr, delta_distr = test_model(act, vel, times)

        act_ll = torch.mean(act_distr.log_prob(act))
        vel_ll = torch.mean(vel_distr.log_prob(vel))
        t_ll = torch.mean(delta_distr.log_prob(d))

        loss = -act_ll - vel_ll - t_ll

        # dot = make_dot(loss, params=dict(test_model.named_parameters()))
        # dot.format = "svg"
        # print("saving graph image...")
        # dot.render()

        loss.backward()
