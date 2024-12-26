import multiprocessing
import os
from typing import Optional, Sequence

import mido
import torch
import torch.utils.data
from torch.nn.utils.rnn import pad_sequence


def pad_fixed_length(
    tensors: Sequence[torch.Tensor], length: int, value: float = 0.0
) -> torch.Tensor:
    padded = pad_sequence(tensors, batch_first=True, padding_value=value)

    max_len = padded.size()[1]
    if max_len < length:
        shape = list(padded.size())
        shape[1] = length - max_len
        return torch.concat((padded, torch.full(shape, value)), dim=1)

    return padded[:, :max_len]


def collate_midi_seq(
    tensors: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    length: int = 1 << 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    actions, params, times = zip(*tensors)

    times = pad_fixed_length(times, length, 5e4)

    return (
        pad_fixed_length(actions, length),
        pad_fixed_length(params, length),
        times,
        (times < 5e4).type_as(times),
    )


class LoadPreprocessed(torch.utils.data.Dataset):
    def __init__(self, tensors_path: str, max_length: int = 100000000):
        super().__init__()
        m = multiprocessing.Manager()
        self.dir = m.list(list(map(lambda x: x.path, os.scandir(tensors_path))))
        self.args = m.Namespace()
        self.args.max_len = max_length

    def __len__(self) -> int:
        return len(self.dir)

    def __getitem__(self, index: int) -> tuple:
        return tuple(map(lambda t: t[: self.args.max_len], torch.load(self.dir[index])))


class LoadMidis(torch.utils.data.Dataset):
    def __init__(
        self, files_path: str, max_length: int = 100000000, min_step: float = 0.025
    ):
        super().__init__()
        m = multiprocessing.Manager()
        self.dir = m.list(list(map(lambda x: x.path, os.scandir(files_path))))
        self.args = m.Namespace()
        self.args.max_len = max_length
        self.args.min_step = min_step

    def __len__(self) -> int:
        return len(self.dir)

    # returns action, value tuple
    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return LoadMidis.process_midi(
            self.dir[index], self.args.max_len, self.args.min_step
        )

    @staticmethod
    def process_midi(
        path: str, max_length: int, min_step: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with mido.MidiFile(path) as midi:
            actions = [torch.ones(256)]
            params = [torch.full((128,), 0.5)]
            timestamps = [0.0]
            held_notes = set()
            pedal_down = False
            accumulated_time = 0.0
            skip_time = 1.0
            current_state = PianoState()
            note_states = [False] * 128
            first_message = True

            for message in midi:
                if current_state.is_nonempty():
                    accumulated_time += message.time
                    first_message = False

                    if accumulated_time > min_step:
                        actions.append(current_state.actions)
                        params.append(current_state.params)
                        current_state = PianoState()
                        timestamps.append(skip_time + timestamps[-1])
                        skip_time = accumulated_time
                        accumulated_time = 0.0
                elif not first_message:
                    skip_time += message.time

                match message:
                    case (
                        mido.Message(type="note_on", velocity=0, note=n)
                        | mido.Message(type="note_off", note=n)
                    ):
                        if pedal_down:
                            held_notes.add(n)
                        else:
                            current_state.turn_note_off(n)
                            note_states[n] = False

                    case mido.Message(type="note_on", velocity=vel, note=n):
                        if note_states[n]:
                            current_state.turn_note_off(n)
                        current_state.turn_note_on(n, vel)
                        held_notes.discard(n)
                        note_states[n] = True

                    case mido.Message(type="control_change", control=64, value=val):
                        now = val >= 64
                        if pedal_down and not now:
                            for note in held_notes:
                                current_state.turn_note_off(note)
                                note_states[note] = False
                            held_notes = set()
                        pedal_down = now

                if len(actions) >= max_length:
                    break

            else:
                if current_state.is_nonempty():
                    actions.append(current_state.actions)
                    params.append(current_state.params)
                    timestamps.append(accumulated_time + skip_time + timestamps[-1])

                timestamps.append(timestamps[-1] + 1.0)
                actions.append(torch.zeros(256))
                params.append(torch.zeros(128))

        return torch.stack(actions), torch.stack(params), torch.tensor(timestamps)


def decode(
    actions: torch.Tensor,
    params: torch.Tensor,
    timestamps: torch.Tensor,
    name: Optional[str] = None,
    ticks_per_beat: Optional[int] = None,
    tempo: Optional[int] = None,
) -> mido.MidiFile:
    midi = mido.MidiFile()
    if ticks_per_beat is not None:
        midi.ticks_per_beat = ticks_per_beat

    track = mido.MidiTrack()
    midi.tracks.append(track)
    if tempo is None:
        tempo = 500_000
    if name is None:
        name = ""

    track.append(mido.MetaMessage("track_name", name=name))
    track.append(mido.MetaMessage("set_tempo", tempo=tempo))
    delta_t = 0
    prev_time = 0

    for action, param, timestamp in zip(actions, params, timestamps, strict=True):
        delta_t += mido.second2tick(
            float(timestamp - prev_time), midi.ticks_per_beat, tempo
        )

        for note, velocity in zip(range(len(param)), param):
            if action[note + 128]:
                track.append(mido.Message("note_off", note=note, time=delta_t))
                delta_t = 0

            if action[note]:
                track.append(
                    mido.Message(
                        "note_on",
                        note=note,
                        velocity=round(float(velocity) * 127),
                        time=delta_t,
                    )
                )
                delta_t = 0
        prev_time = timestamp
    return midi


class PianoState:
    def __init__(self):
        # Notes ON, Notes OFF, Start, Stop
        self.actions = torch.zeros(256)
        # Note Velocities
        self.params = torch.full((128,), 0.5)

    def turn_note_on(self, note: int, velocity: int):
        self.actions[note] = True
        self.params[note] = velocity / 127

    def turn_note_off(self, note: int):
        self.actions[note + 128] = True
        self.actions[note] = False
        self.params[note] = 0.0

    def is_nonempty(self) -> bool:
        return any(self.actions)


# tests
if __name__ == "__main__":
    import shutil

    x = LoadMidis("midis", max_length=100)
    print("processing")
    z = x[1000]
    print(z[0].shape)
    print(z[1].shape)

    test_file_path = (
        "midis\\MIDI-UNPROCESSED_19-20-21_R2_2014_MID--AUDIO_21_R2_2014_wav.midi"
    )
    shutil.copyfile(test_file_path, "tests\\original.mid")
    original = mido.MidiFile("tests\\original.mid")

    original_tempo = 500_000
    for message in original:
        if message.type == "set_tempo":
            original_tempo = message.tempo
            break

    print("testing load and decode")
    test = LoadMidis.process_midi(test_file_path, max_length=1000000000, min_step=0.0)
    print(test[0].shape, test[1].shape, test[2].shape)
    assert torch.all((test[2][1:] - test[2][:-1]) > 0)
    full = decode(
        *test, name="full", ticks_per_beat=original.ticks_per_beat, tempo=original_tempo
    )
    full.save("tests\\test_unreduced.mid")

    print("testing reduction")
    test = LoadMidis.process_midi(test_file_path, max_length=1000000000, min_step=0.025)
    print(test[0].shape, test[1].shape, test[2].shape)
    assert torch.all((test[2][1:] - test[2][:-1]) > 0)
    reduced = decode(
        *test,
        name="reduced",
        ticks_per_beat=original.ticks_per_beat,
        tempo=original_tempo,
    )
    reduced.save("tests\\test_reduced.mid")

    combined = mido.MidiFile(type=1)
    combined.tracks += full.tracks + reduced.tracks + original.tracks
    combined.ticks_per_beat = original.ticks_per_beat
    combined.save("tests\\merged.mid")
