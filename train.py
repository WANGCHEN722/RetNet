import lightning.pytorch as pl
import torch
import torch.utils.data
from load_midi import LoadMidis, LoadPreprocessed, collate_midi_seq
from meta_tpp import MuseEmbedding, MuseTPP, MuseTPPDecoder, RetentiveMetaTPPEncoder


def main():
    pl.seed_everything(4406, workers=True)
    midis = LoadMidis("midis")

    count = 1

    if count is None:
        raise RuntimeError("CPU count could not be determined automatically")

    data = torch.utils.data.DataLoader(
        midis,
        batch_size=2,
        shuffle=True,
        num_workers=count,
        collate_fn=collate_midi_seq,
        pin_memory=True,
        persistent_workers=True,
    )

    embedding = MuseEmbedding(2, 2, 2)
    encoder = RetentiveMetaTPPEncoder(6, 1, 3)
    decoder = MuseTPPDecoder(12, 1, 1, 1, 1, 1)
    model = MuseTPP(embedding, encoder, decoder, 10)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        strategy="auto",
        default_root_dir=".",
        max_epochs=16,
    )
    trainer.fit(model, data)


if __name__ == "__main__":
    main()
