from torch.utils.data import Dataset
from transformers import WhisperTokenizer, WhisperFeatureExtractor
from datasets import load_dataset
import librosa


class WhisperAudioCaptionProcessor:
    def __init__(self, model_name: str, caption_token: str = "<|caption|>"):
        self.tokenizer = WhisperTokenizer.from_pretrained(model_name)
        
        # add caption token to tokenizer if not present
        self.caption_token = caption_token
        
        if self.caption_token not in self.tokenizer.get_vocab():
            self.tokenizer.add_tokens(self.caption_token, special_tokens=True)
        
        self.prefix = "<|startoftranscript|>" + self.caption_token
        self.postfix = self.tokenizer.eos_token
        
        self.feat_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    
    def __call__(self, item):
        caption = item["json"]["text"]
        audio = item["flac"]["array"]
        sr = item["flac"]["sampling_rate"]
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        
        input_features = self.feat_extractor(audio, sampling_rate=16000, return_tensors="pt").input_features[0]
        input_ids = self.tokenizer(self.prefix + caption + self.postfix, return_tensors="pt", add_special_tokens=False).input_ids
        return {
            "input_features": input_features,
            "input_ids": input_ids.squeeze(0)
        }


class WhisperAudioCaptionCollator:
    def __init__(self, processor: WhisperAudioCaptionProcessor):
        self.processor = processor
    
    def __call__(self, items):
        input_features = [{"input_features": item["input_features"]} for item in items]
        batch = self.processor.feat_extractor.pad(input_features, return_tensors="pt")
        
        input_ids = [{"input_ids": item["input_ids"]} for item in items]
        padded_input_ids = self.processor.tokenizer.pad(input_ids, return_tensors="pt")
        batch["decoder_input_ids"] = padded_input_ids["input_ids"]
        batch["decoder_attention_mask"] = padded_input_ids["attention_mask"]
        
        labels = padded_input_ids["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        return batch


class WhisperAudioCaptionDataset(Dataset):
    def __init__(self, model_name: str, dataset_name: str, split: str, streaming: bool = False, caption_token: str = "<|caption|>"):
        super().__init__()
        
        self.dataset = load_dataset(dataset_name, split=split, streaming=streaming)
        self.tokenizer = WhisperTokenizer.from_pretrained(model_name)
        
        # add caption token to tokenizer if not present
        self.caption_token = caption_token
        
        if self.caption_token not in self.tokenizer.get_vocab():
            self.tokenizer.add_tokens(self.caption_token, special_tokens=True)
        
        self.prefix = "<|startoftranscript|>" + self.caption_token
        self.postfix = self.tokenizer.eos_token
        
        self.feat_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    
    def __getitem__(self, index):
        item = self.dataset[index]