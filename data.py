from torch.utils.data import Dataset
from transformers import WhisperProcessor
from transformers import WhisperTokenizer, WhisperFeatureExtractor
from datasets import load_dataset
import librosa
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch


class WhisperAudioCaptionProcessor:
    def __init__(self, model_name: str):
        # use translate task as captioning task now
        self.tokenizer = WhisperTokenizer.from_pretrained(model_name, task="translate")
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    
    def __call__(self, item):
        caption = item["json"]["text"]
        audio = item["flac"]["array"]
        sr = item["flac"]["sampling_rate"]
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        
        input_features = self.feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_features[0]
        input_ids = self.tokenizer(caption).input_ids
        return {
            "input_features": input_features,
            "labels": input_ids
        }

# from https://huggingface.co/blog/fine-tune-whisper
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

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

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        
        return batch
