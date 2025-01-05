import json

import hydra
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from transformers import (
    Trainer,
    TrainingArguments,
    WhisperConfig,
    WhisperForConditionalGeneration,
)

from data import DataCollatorSpeechSeq2SeqWithPadding, WhisperAudioCaptionProcessor


@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(config: DictConfig):
    model_config = config.model_config
    data_config = config.data_config

    # Load model
    whisper_config = WhisperConfig.from_pretrained(model_config.model_name)
    whisper_config._attn_implementation = model_config.attn_implementation
    whisper_model = WhisperForConditionalGeneration.from_pretrained(model_config.model_name, config=whisper_config)

    # use translate task as captioning task
    whisper_model.generation_config.task = "translate"
    whisper_model.generation_config.language = "en"
    whisper_model.generation_config.forced_decoder_ids = None

    # Load the dataset
    processor = WhisperAudioCaptionProcessor(
        model_config.tokenizer_name, data_config.audio_key, data_config.caption_key
    )
    collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    dataset = load_dataset(data_config.name, split="train", streaming=True).map(processor, batched=False)

    # Load trainer
    trainer_config = OmegaConf.to_container(config.trainer_config, resolve=True)
    print(json.dumps(trainer_config, indent=2))
    trainer = Trainer(
        model=whisper_model, args=TrainingArguments(**trainer_config), data_collator=collator, train_dataset=dataset
    )
    trainer.train()


if __name__ == "__main__":
    main()
