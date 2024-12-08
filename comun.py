import torch
from torch import nn
from pytorch_lightning import LightningModule
import evaluate
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import whisper
import torchaudio
import torchaudio.transforms as at
import numpy as np
from pytorch_lightning import seed_everything
from pathlib import Path


LANG = "es"
CHANNELS = 80
SAMPLE_RATE = 16000
AUDIO_DIR = "audios"
TRANSCRPTS = "transcripciones.tsv"
SAMPLE_RATE = 16000
BATCH_SIZE = 2
TRAIN_RATE = 0.8
AUDIO_MAX_LENGTH = 480000
DEVICE = "gpu" if torch.cuda.is_available() else "cpu"
MODEL = "small"
SEED = 42
seed_everything(SEED, workers=True)


def load_wave(wave_path, sample_rate: int = 16000) -> torch.Tensor:
    waveform, sr = torchaudio.load(wave_path, normalize=True)
    if sample_rate != sr:
        waveform = at.Resample(sr, sample_rate)(waveform)
    return waveform


def get_audio_file_list(audio_dir,
                        transcripts_path,
                        audio_max_sample_length=480000,
                        sample_rate=16000):
    audio_transcript_pair_list = []
    audio_dir = Path(audio_dir)
    if not audio_dir.exists():
        return []

    with open(transcripts_path, "r") as f:
        text_list = f.readlines()
    for text in text_list:
        audio_id, text = text.replace("\n", "").split("\t")

        audio_path = audio_dir / f"{audio_id}.wav"
        if audio_path.exists():
            audio = load_wave(audio_path, sample_rate=sample_rate)[0]
            if len(audio) > audio_max_sample_length:
                print(len(text), len(audio), audio_id, text)
                continue
            audio_transcript_pair_list.append(
                (audio_id, str(audio_path), text))
    return audio_transcript_pair_list


def get_audio_file_split(audio_dir,
                         transcripts_path,
                         audio_max_sample_length=480000,
                         sample_rate=16000):
    train_num = 0
    with open(TRANSCRPTS, 'r') as fp:
        train_num = int(len(fp.readlines()) * TRAIN_RATE)
    parejas = get_audio_file_list(AUDIO_DIR, TRANSCRPTS,
                                  AUDIO_MAX_LENGTH, SAMPLE_RATE)
    return (parejas[:train_num], parejas[train_num:])


train_dataset, eval_dataset = get_audio_file_split(
    AUDIO_DIR, TRANSCRPTS, AUDIO_MAX_LENGTH, SAMPLE_RATE)


class WhisperDataCollatorWithPadding:
    def __call__(sefl, features):
        input_ids, labels, dec_input_ids = [], [], []
        for f in features:
            input_ids.append(f["input_ids"])
            labels.append(f["labels"])
            dec_input_ids.append(f["dec_input_ids"])

        input_ids = torch.concat([input_id[None, :] for input_id in input_ids])

        label_lengths = [len(lab) for lab in labels]
        dec_input_ids_length = [len(e) for e in dec_input_ids]
        max_label_len = max(label_lengths+dec_input_ids_length)

        labels = [
            np.pad(lab, (0, max_label_len - lab_len),
                   'constant', constant_values=-100)
            for lab, lab_len in zip(labels, label_lengths)]
        EOF_TOKEN = 50257
        dec_input_ids = [
            np.pad(e, (0, max_label_len - e_len),
                   'constant', constant_values=EOF_TOKEN)
            for e, e_len in zip(dec_input_ids, dec_input_ids_length)]

        batch = {
            "labels": labels,
            "dec_input_ids": dec_input_ids
        }

        batch = {k: torch.tensor(np.array(v), requires_grad=False)
                 for k, v in batch.items()}
        batch["input_ids"] = input_ids

        return batch


class SpeechDataset(torch.utils.data.Dataset):
    def __init__(self, audio_info_list, tokenizer, sample_rate,
                 channels) -> None:
        super().__init__()

        self.audio_info_list = audio_info_list
        self.sample_rate = sample_rate
        self.tokenizer = tokenizer
        self.channels = channels

    def __len__(self):
        return len(self.audio_info_list)

    def __getitem__(self, id):
        audio_id, audio_path, text = self.audio_info_list[id]

        audio = load_wave(audio_path, sample_rate=self.sample_rate)
        audio = whisper.pad_or_trim(audio.flatten())
        mel = whisper.log_mel_spectrogram(audio, n_mels=self.channels)

        text = [*self.tokenizer.sot_sequence_including_notimestamps] + \
            self.tokenizer.encode(text)
        labels = text[1:] + [self.tokenizer.eot]

        return {
            "input_ids": mel,
            "labels": labels,
            "dec_input_ids": text
        }


class Config:
    learning_rate = 0.0005
    weight_decay = 0.01
    adam_epsilon = 1e-8
    warmup_steps = 2
    batch_size = 16
    num_worker = 2
    num_train_epochs = 10
    gradient_accumulation_steps = 1
    sample_rate = SAMPLE_RATE


class WhisperModelModule(LightningModule):
    def __init__(self, cfg: Config,
                 model_name=MODEL, lang=LANG,
                 train_dataset=[], eval_dataset=[]) -> None:
        super().__init__()
        self.options = whisper.DecodingOptions(
            language=lang, without_timestamps=True)
        self.model = whisper.load_model(model_name)
        self.tokenizer = whisper.tokenizer.get_tokenizer(
            True, language=LANG, task=self.options.task)

        for p in self.model.encoder.parameters():
            p.requires_grad = False

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.metrics_wer = evaluate.load("wer")
        self.metrics_cer = evaluate.load("cer")

        self.cfg = cfg
        self.__train_dataset = train_dataset
        self.__eval_dataset = eval_dataset

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_id):
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()

        with torch.no_grad():
            audio_features = self.model.encoder(input_ids)

        out = self.model.decoder(dec_input_ids, audio_features)
        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))
        self.log("train/loss", loss, on_step=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_id):
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()

        audio_features = self.model.encoder(input_ids)
        out = self.model.decoder(dec_input_ids, audio_features)

        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))

        out[out == -100] = self.tokenizer.eot
        labels[labels == -100] = self.tokenizer.eot

        o_list, l_list = [], []
        for o, l in zip(out, labels):
            o = torch.argmax(o, dim=1)
            o_list.append(self.tokenizer.decode(o))
            l_list.append(self.tokenizer.decode(l))
        cer = self.metrics_cer.compute(references=l_list, predictions=o_list)
        wer = self.metrics_wer.compute(references=l_list, predictions=o_list)

        self.log("val/loss", loss, on_step=True, prog_bar=True, logger=True)
        self.log("val/cer", cer, on_step=True, prog_bar=True, logger=True)
        self.log("val/wer", wer, on_step=True, prog_bar=True, logger=True)

        return {
            "cer": cer,
            "wer": wer,
            "loss": loss
        }

    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters()
                           if not any(nd in n for nd in no_decay)],
                "weight_decay": self.cfg.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters()
                           if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.cfg.learning_rate,
                          eps=self.cfg.adam_epsilon)
        self.optimizer = optimizer

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.cfg.warmup_steps,
            num_training_steps=self.t_total
        )
        self.scheduler = scheduler

        return [optimizer], [{
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1
        }]

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.t_total = (
                (len(self.__train_dataset) // (self.cfg.batch_size))
                // self.cfg.gradient_accumulation_steps
                * float(self.cfg.num_train_epochs)
            )

    def train_dataloader(self):
        dataset = SpeechDataset(
            self.__train_dataset, self.tokenizer, self.cfg.sample_rate,
            CHANNELS)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=self.cfg.num_worker,
            collate_fn=WhisperDataCollatorWithPadding()
        )

    def val_dataloader(self):
        dataset = SpeechDataset(
            self.__eval_dataset, self.tokenizer, self.cfg.sample_rate,
            CHANNELS)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_worker,
            collate_fn=WhisperDataCollatorWithPadding()
        )
