from dataclasses import dataclass
from typing import Any, Dict, List, Union

import librosa
import torch
from transformers import WhisperFeatureExtractor, WhisperTokenizer


class WhisperAudioCaptionProcessor:
    def __init__(self, model_name: str):
        # use translate task as captioning task now
        self.tokenizer = WhisperTokenizer.from_pretrained(model_name, language="en", task="translate")
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)

    def __call__(self, item):
        caption = item["json"]["text"]
        audio = item["flac"]["array"]
        sr = item["flac"]["sampling_rate"]
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        input_features = self.feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_features[0]
        input_ids = self.tokenizer(caption).input_ids
        return {"input_features": input_features, "labels": input_ids}


class WhisperAudioCaptionTestProcessor:
    def __init__(self, model_name: str):
        # use translate task as captioning task now
        self.tokenizer = WhisperTokenizer.from_pretrained(model_name, language="en", task="translate")
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)

    def __call__(self, item):
        caption = item["caption"]
        audio = item["audio"]["array"]
        sr = item["audio"]["sampling_rate"]
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        input_features = self.feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_features[0]
        input_ids = self.tokenizer(caption).input_ids
        return {"input_features": input_features, "labels": input_ids}


# from https://huggingface.co/blog/fine-tune-whisper
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels

        return batch
