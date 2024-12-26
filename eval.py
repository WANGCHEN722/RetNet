import torch
from einops import rearrange
from meta_tpp import MuseEmbedding, MuseTPP, MuseTPPDecoder, RetentiveMetaTPPEncoder

torch.distributions.Distribution.set_default_validate_args(True)


def piano_note(state: torch.Tensor, actions: torch.Tensor):
    formatted_actions = actions.clone()
    formatted_actions[128:] += state[:128] * actions[:128]
    formatted_actions[128:] *= state[:128]
    formatted_actions = formatted_actions.clamp(0.0, 1.0)

    new_state = state + formatted_actions[:128] - formatted_actions[128:]

    assert torch.all(new_state == new_state.clamp(0.0, 1.0))

    return new_state, formatted_actions


@torch.inference_mode()
def load_model(state_dict: dict) -> torch.nn.Module:
    embedding = MuseEmbedding(256, 256, 256)
    encoder = RetentiveMetaTPPEncoder(768, 12, 8)
    decoder = MuseTPPDecoder(1536, 256, 256, 8, 256, 8)
    model = MuseTPP(embedding, encoder, decoder, 2048)
    model.to("cuda")

    model.load_state_dict(state_dict)
    model.eval()

    return model


@torch.inference_mode()
def predict_sequence(
    model: torch.nn.Module,
    start: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    max_length: int = 1 << 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    state = None
    out_acts = []
    out_vels = []
    out_times = []

    if any(s.nelement() == 0 for s in start):
        raise ValueError("Empty start sequence")

    for actions, velocities, times in zip(
        *map(lambda s: rearrange(s, "l ... -> l 1 ..."), start), strict=True
    ):
        out_acts.append(actions[0])
        out_vels.append(velocities[0])
        out_times.append(times[0])
        act, vel, delta, state = model(
            actions, velocities, times, state, mode="recurrent"
        )

    piano_state = torch.zeros_like(out_vels[0])
    for acti in out_acts[1:]:
        piano_state, _ = piano_note(piano_state, acti)

    counter = 0
    actions = act.sample()  # type: ignore
    print(out_times)
    # print(counter, out_times[-1], torch.sum(actions[:, 128:]))

    while counter < max_length:
        # piano_state, actions = piano_note(piano_state, actions[0])
        # actions = actions.unsqueeze(0)
        print(
            counter,
            out_times[-1],
            torch.sum(actions[:, :128]),
            torch.sum(actions[:, 128:]),
        )
        out_acts.append(actions[0])

        velocities = vel.sample()  # type: ignore
        out_vels.append(velocities[0])

        times = delta.sample() * 0.1 + out_times[-1]  # type: ignore
        out_times.append(times[0])

        act, vel, delta, state = model(
            actions, velocities * actions[:, :128], times, state, mode="recurrent"
        )
        # print(actions)
        # print(velocities)
        # print(times)
        counter += 1
        actions = act.sample()

    p = torch.stack(out_acts), torch.stack(out_vels), torch.stack(out_times)
    return p


if __name__ == "__main__":
    from load_midi import LoadMidis, decode

    print("Processing midi...")
    new = tuple(
        map(
            lambda s: s.to("cuda"),
            LoadMidis.process_midi(
                r"tests\HappyBirthday.mid",
                max_length=10,
                min_step=0.025,
            ),
        )
    )
    print(new)

    print("Loading model...")
    model = load_model(torch.load(r"mp_rank_00_model_states.pt")["module"])

    print("Generating music...")
    act, vel, times = predict_sequence(model, new, 100)  # type: ignore

    print("Saving output...")
    decode(act[1:], vel[1:], times[1:]).save(r"tests\HappyBirthToYou.mid")
    print("Done!")
