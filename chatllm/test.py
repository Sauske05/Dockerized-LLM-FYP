from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import AsyncGenerator, Iterator
import asyncio
from llama_cpp import Llama
import uvicorn

# Model configuration
local_model_dir = "./Qwen_gguf/deepseek-r1-distill-qwen-1.5b-q4_0.gguf"
app = FastAPI(title = 'Chatbot API')

class RequestData(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    stop: list = []

# Initialize the model
llm = Llama(
    model_path=local_model_dir,
    n_ctx=2048,  # Context window size
    n_threads=4  # Number of CPU threads to use
    #n_gpu_layers=-1
)

@app.post("/generate")
async def generate_text(request: RequestData):
    async def token_generator() -> AsyncGenerator[str, None]:
        # Create a generator from the sync iterator
        def sync_generator() -> Iterator[str]:
            response = llm(
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=request.stop,
                stream=True
            )
            
            for chunk in response:
                if "choices" in chunk and len(chunk["choices"]) > 0:
                    token = chunk["choices"][0]["text"]
                    yield token
        
        # Convert sync generator to async generator
        for token in sync_generator():
            yield token
            # Small delay to prevent overwhelming the client
            await asyncio.sleep(0.01)

    return StreamingResponse(token_generator(), media_type="text/plain")

@app.get("/health")
async def health_check():
    return {"status": "ok", "model": local_model_dir}

if __name__ == "__main__":
    uvicorn.run("test:app", host="0.0.0.0", port=8080, reload=True)