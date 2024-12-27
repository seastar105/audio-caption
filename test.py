import argparse

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import WhisperForConditionalGeneration
from tqdm.auto import tqdm

from data import DataCollatorSpeechSeq2SeqWithPadding, WhisperAudioCaptionTestProcessor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, required=True)
    # parser.add_argument("--judge_model", type=str, required=True, choices=["openai", "gemini"])
    # parser.add_argument("--judge_mode", type=str, required=True, choices=["use-audio", "text-only"])
    parser.add_argument("--num_dataloader_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)

    args = parser.parse_args()

    processor = WhisperAudioCaptionTestProcessor("openai/whisper-base")
    tokenizer = processor.tokenizer
    collator = DataCollatorSpeechSeq2SeqWithPadding(processor)
    dataset = (
        load_dataset("seastar105/emo_speech_caption_test", split="train", streaming=True)
        .map(processor, batched=False)
        .select_columns(["input_features", "labels"])
    )

    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    results = []
    
    temperature = 1.0
    num_beams = 5
    max_length = 256
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_dataloader_workers, collate_fn=collator)
    
    model = model.to(device)
    
    for batch in tqdm(dataloader, desc="Inference"):
        gt_captions = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
        input_features = batch["input_features"].to(device)
        
        outputs = model.generate(
            input_features,
            do_sample=False,
            temperature=temperature,
            num_beams=num_beams,
            max_length=max_length,
        )
        for i, output in enumerate(outputs):
            results.append({"gt_caption": gt_captions[i], "pred_caption": tokenizer.decode(output, skip_special_tokens=True)})

    print(results)
