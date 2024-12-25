from datasets import load_dataset
from data import WhisperAudioCaptionProcessor, DataCollatorSpeechSeq2SeqWithPadding
from transformers import WhisperForConditionalGeneration, WhisperConfig
import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments


@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(config: DictConfig):
    model_config = config.model_config
    data_config = config.data_config
    
    # Load model
    whisper_config = WhisperConfig.from_pretrained(model_config.name)
    whisper_config._attn_implementation = model_config.attn_implementation
    whisper_model = WhisperForConditionalGeneration.from_pretrained(model_config.name, config=whisper_config)
    
    # use translate task as captioning task
    whisper_model.generation_config.task = "translate"
    whisper_model.generation_config.forced_decoder_ids = None
    
    # Load the dataset
    processor = WhisperAudioCaptionProcessor(model_config.name)
    collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor, decoder_start_token_id=whisper_model.config.decoder_start_token_id)
    
    dataset = load_dataset(data_config.name, split="train", streaming=True).map(processor, batched=False)
    
    # Load trainer
    trainer_config = OmegaConf.to_container(config.trainer_config, resolve=True)
    trainer = Seq2SeqTrainer(
        model=whisper_model,
        args=Seq2SeqTrainingArguments(**trainer_config),
        data_collator=collator,
        train_dataset=dataset
    )
    trainer.train()

if __name__ == "__main__":
    main()
