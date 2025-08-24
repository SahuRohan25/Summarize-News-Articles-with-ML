from dataclasses import dataclass
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq,
                          Seq2SeqTrainingArguments, Seq2SeqTrainer)

@dataclass
class Config:
    model_name: str = "facebook/bart-large-cnn"
    max_input_tokens: int = 1024
    max_summary_tokens: int = 128
    lr: float = 3e-5
    batch_size: int = 4
    epochs: int = 3

def preprocess_function(examples, tokenizer: AutoTokenizer, cfg: Config):
    inputs = examples["article"]
    targets = examples["summary"]
    model_inputs = tokenizer(inputs, max_length=cfg.max_input_tokens, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=cfg.max_summary_tokens, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def train(dataset_path: str, out_dir: str, cfg: Config = Config()):
    raw = load_dataset("json", data_files={"train": dataset_path, "validation": dataset_path})
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name)
    collator = DataCollatorForSeq2Seq(tokenizer=tok, model=model)

    tokenized = raw.map(lambda ex: preprocess_function(ex, tok, cfg), batched=True,
                        remove_columns=raw["train"].column_names)

    args = Seq2SeqTrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        learning_rate=cfg.lr,
        num_train_epochs=cfg.epochs,
        predict_with_generate=True,
        logging_steps=50,
        evaluation_strategy="steps",
        save_steps=500,
        fp16=True,
        report_to=[]
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=collator,
        tokenizer=tok,
    )
    trainer.train()
    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)
    return out_dir
