import os
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import uvicorn

class Item(BaseModel):
    text: str

app = FastAPI(title="News Summarizer API")

_model_cache = {}

def get_pipe(model_name: str):
    if model_name not in _model_cache:
        _model_cache[model_name] = pipeline("summarization", model=model_name)
    return _model_cache[model_name]

@app.post("/summarize")
async def summarize(item: Item, model: str = "facebook/bart-large-cnn"):
    pipe = get_pipe(model)
    out = pipe(item.text, truncation=True, max_length=128, min_length=48, do_sample=False)[0]
    return {"summary": out["summary_text"]}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
