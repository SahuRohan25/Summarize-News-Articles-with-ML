from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

class Summarizer:
    def __init__(self, model_name: str = "facebook/bart-large-cnn", device: int = -1,
                 max_input_tokens: int = 1024, max_summary_tokens: int = 128):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.pipe = pipeline("summarization", model=self.model, tokenizer=self.tokenizer, device=device)
        self.max_input_tokens = max_input_tokens
        self.max_summary_tokens = max_summary_tokens

    def __call__(self, text: str) -> str:
        out = self.pipe(text, truncation=True, max_length=self.max_summary_tokens,
                        min_length=max(32, self.max_summary_tokens//2), do_sample=False)
        return out[0]["summary_text"].strip()
