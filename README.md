# audio-caption

Training code for fine-tuning whisper model for audio captioning. Currently focusing on speech style descriptions.

# Training

If you are familiar with hydra or huggingface Trainer, check .yaml files in configs folder. main configs are `trainer_config`, `data_config`, `model_config`

To fine-tuning whisper-base on dataset https://huggingface.co/datasets/mitermix/audiosnippets, using sdpa attention backend. start with command below

```
accelerate launch train.py model_config.model_name="openai/whisper-base" model_config.attn_implementation="sdpa" data_config.name="mitermix/audiosnippets" data_config.audio_key="mp3" data_config.caption_key="caption"
```

Currently, Huggingface streaming dataset is supported. 


## Pretrained models

Model Link: https://huggingface.co/seastar105/whisper-base-emo-speech-caption

Train commands

```
accelerate launch train.py model_config.model_name="openai/whisper-base" model_config.attn_implementation=flash_attention_2 data_config.name="mitermix/audiosnippets" data_config.audio_key="mp3" data_config.caption_key="caption" trainer_config.bf16=true trainer_config.dataloader_num_workers=16 trainer_config.per_device_train_batch_size=128 trainer_config.max_steps=50000 ++trainer_config.push_to_hub=true ++trainer_config.hub_model_id='seastar105/whisper-base-emo-speech-caption' ++trainer_config.save_total_limit=5 ++trainer_config.save_steps=5000 ++trainer_config.dataloader_prefetch_factor=4 ++trainer_config.warmup_steps=5000
```

# Inference
To train caption task, Whisper's translate special token is used, and language is fixed on english. You can generate some caption using pretrained model.
```
from transformers import WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperTokenizer
from datasets import load_dataset
import librosa

tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-base", language="en", task="translate")
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-base")
model = WhisperForConditionalGeneration.from_pretrained("seastar105/whisper-base-emo-speech-caption")

dataset = load_dataset("seastar105/emo_speech_caption_test", split="train", streaming=True)
sample = next(iter(dataset))

audio = sample["audio"]["array"]
sample_rate = sample["audio"]["sampling_rate"]

if sample_rate != 16000:
    audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)

input_features = feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_features
pred = model.generate(input_features=input_features, do_sample=True, num_beams=5, temperature=0.7)
print(tokenizer.batch_decode(pred, skip_special_tokens=True))

['This recording contains a female speaker, with a clear and articulate voice, talking about black people and their']
```

 # TODOs
 - [ ] support webdataset
 - [ ] support evaluation metrics(e.g. BLEU, ROUGE-L, METEOR, BERTScore, CIDEr-D, SPICE, CLAPScore)
