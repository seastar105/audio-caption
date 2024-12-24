from datasets import load_dataset
from data import WhisperAudioCaptionProcessor, WhisperAudioCaptionCollator
from transformers import WhisperForConditionalGeneration
from torch.utils.data import DataLoader
import torch
from tqdm.auto import tqdm


if __name__ == "__main__":
    processor = WhisperAudioCaptionProcessor("openai/whisper-tiny")
    collator = WhisperAudioCaptionCollator(processor)
    dataset = load_dataset("krishnakalyan3/emo_webds_2", split="train", streaming=True).map(processor, batched=False).select_columns(["input_features", "input_ids"])
    
    num_workers = 4
    batch_size = 8
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator, num_workers=num_workers)
    
    whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    tokenzier = processor.tokenizer

    if whisper_model.get_input_embeddings().num_embeddings < len(tokenzier):
        whisper_model.resize_token_embeddings(len(tokenzier))
    
    num_steps = 5000
    optimizer = torch.optim.AdamW(whisper_model.parameters(), lr=1e-5, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-5, total_steps=num_steps)
    device = torch.device("mps")
    
    whisper_model = whisper_model.to(device)
    whisper_model.train()
    losses = []
    num_intervals = 10
    
    for step, batch in tqdm(enumerate(dataloader), total=num_steps):
        if (step + 1) % num_intervals == 0:
            print(f"Step {step + 1}, Loss: {sum(losses[-num_intervals:]) / num_intervals}, Learning Rate: {scheduler.get_last_lr()[0]}")
        if step >= num_steps:
            break
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        outputs = whisper_model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        losses.append(loss.item())
