from pathlib import Path

from comun import SAMPLE_RATE, Config, WhisperModelModule
from comun import WhisperDataCollatorWithPadding, SpeechDataset
from comun import train_dataset, eval_dataset
from comun import DEVICE, MODEL, CHANNELS, LANG

import torch
import whisper

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer

print("TRAIN:", len(train_dataset))
print("EVAL:", len(eval_dataset))

woptions = whisper.DecodingOptions(language=LANG, without_timestamps=True)
wmodel = whisper.load_model(MODEL)
wtokenizer = whisper.tokenizer.get_tokenizer(
    True, language=LANG, task=woptions.task)


dataset = SpeechDataset(eval_dataset,
                        wtokenizer, SAMPLE_RATE, CHANNELS)
loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, collate_fn=WhisperDataCollatorWithPadding())

for b in loader:
    print(b["labels"].shape)
    print(b["input_ids"].shape)
    print(b["dec_input_ids"].shape)

    for token, dec in zip(b["labels"], b["dec_input_ids"]):
        token[token == -100] = wtokenizer.eot
        text = wtokenizer.decode(token)
        print(text)

        dec[dec == -100] = wtokenizer.eot
        text = wtokenizer.decode(dec)
        print(text)

    break


with torch.no_grad():
    print(b["input_ids"].shape)
    audio_features = wmodel.encoder(b["input_ids"].cuda())
    input_ids = b["input_ids"]
    labels = b["labels"].long()
    dec_input_ids = b["dec_input_ids"].long()

    audio_features = wmodel.encoder(input_ids.cuda())
    print(dec_input_ids)
    print(input_ids.shape, dec_input_ids.shape, audio_features.shape)
    print(audio_features.shape)
    print()
out = wmodel.decoder(dec_input_ids.cuda(), audio_features)


print(out.shape)
print(out.view(-1, out.size(-1)).shape)
print(b["labels"].view(-1).shape)


tokens = torch.argmax(out, dim=2)
for token in tokens:
    token[token == -100] = wtokenizer.eot
    text = wtokenizer.decode(token)
    print(text)


log_output_dir = "tensor_logs"
check_output_dir = "artifacts"

train_name = "whisper"
train_id = "00001"

cfg = Config()

Path(log_output_dir).mkdir(exist_ok=True)
Path(check_output_dir).mkdir(exist_ok=True)

tflogger = TensorBoardLogger(
    save_dir=log_output_dir,
    name=train_name,
    version=train_id
)

checkpoint_callback = ModelCheckpoint(
    dirpath=f"{check_output_dir}/checkpoint",
    filename="checkpoint-{epoch:04d}",
    save_top_k=-1
)

callback_list = [checkpoint_callback,
                 LearningRateMonitor(logging_interval="epoch")]
model = WhisperModelModule(
    cfg, MODEL, LANG, train_dataset, eval_dataset)

trainer = Trainer(
    precision='16-mixed',
    accelerator=DEVICE,
    max_epochs=cfg.num_train_epochs,
    accumulate_grad_batches=cfg.gradient_accumulation_steps,
    logger=tflogger,
    callbacks=callback_list
)

trainer.fit(model)
