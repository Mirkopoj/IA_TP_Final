import torch
import whisper
import evaluate

from comun import SAMPLE_RATE, Config, WhisperModelModule
from comun import WhisperDataCollatorWithPadding, SpeechDataset
from comun import eval_dataset, LANG, CHANNELS

woptions = whisper.DecodingOptions(language=LANG, without_timestamps=True)
wtokenizer = whisper.tokenizer.get_tokenizer(
    True, language=LANG, task=woptions.task)

checkpoint_path = "artifacts/checkpoint/checkpoint-epoch=0009.ckpt"

state_dict = torch.load(checkpoint_path)
print(state_dict.keys())
state_dict = state_dict['state_dict']

cfg = Config()

whisper_model = WhisperModelModule(cfg)
whisper_model.load_state_dict(state_dict)

dataset = SpeechDataset(eval_dataset,
                        wtokenizer, SAMPLE_RATE, CHANNELS)
loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, collate_fn=WhisperDataCollatorWithPadding())

refs = []
res = []
for b in loader:
    input_ids = b["input_ids"].half().cuda()
    labels = b["labels"].long().cuda()
    with torch.no_grad():
        results = whisper_model.model.decode(input_ids, woptions)
        for r in results:
            res.append(r.text)
        for lab in labels:
            lab[lab == -100] = wtokenizer.eot
            ref = wtokenizer.decode(lab)
            refs.append(ref)

cer_metrics = evaluate.load("cer")
cer_metrics.compute(references=refs, predictions=res)

for k, v in zip(refs, res):
    print("-"*10)
    print(k)
    print(v)
