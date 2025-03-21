import asyncio
from fastapi  import FastAPI, Body, WebSocket
from typing import Annotated, AsyncGenerator, Iterator
from contextlib import asynccontextmanager
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextStreamer
import torch
import uvicorn
from pydantic import BaseModel
from bert_model.configure import *
from bert_model.tokenizer import *
from bert_model.model import *
from fastapi.responses import StreamingResponse
from llama_cpp import Llama

recommendation_local_model_dir = "./Qwen_gguf/deepseek-r1-distill-qwen-1.5b-q4_0.gguf"
chat_local_model_dir = "./chat_model/llama-3.2_4bit.gguf"

# Initialize the model
chat_lllm = Llama(
    model_path=recommendation_local_model_dir,
    n_ctx=2048,  # Context window size
    n_threads=4  # Number of CPU threads to use
    #n_gpu_layers=-1
)
recommendation_llm = Llama(
    model_path=recommendation_local_model_dir,
    n_ctx=2048,  # Context window size
    n_threads=4  # Number of CPU threads to use
    #n_gpu_layers=-1
)

def load_bert():
    model = SentimentModel(config()['h'], config()['d_model'], config()['d_ff'], config()['labels'])
    model.load_state_dict(torch.load('./bert_model/model_state_dict.pth', weights_only=True))
    return model

async def bert_inference(input_text, bert_model):
    label_dict = {0 : 'Anxiety', 1 : 'Depression', 2 : 'Normal', 3 : 'Suicidal', 4 : 'Personality disorder'}
    tokenizer_obj = Tokenizer()
    tokenized_input = tokenizer_obj.tokenize([input_text], 100)
    input_ids = tokenized_input['input_ids'].unsqueeze(0)
    print(input_ids.size())
    input_mask_ids = tokenized_input['attention_mask'].unsqueeze(0)
    print(input_mask_ids.size())
    input_mask = input_mask_ids.transpose(-1,-2)
    input_attn_mask = ((input_mask @ input_mask.transpose(-1,-2)).unsqueeze(1))
    bert_model.eval()
    model_pred = bert_model(input_ids, input_attn_mask)
    model_idx = torch.argmax(model_pred[0]).item() #torch.argmax(model_pred.squeeze())
    if model_idx in label_dict.keys():
        return  label_dict[model_idx]
@asynccontextmanager
async def lifespan(app:FastAPI):
    #chat_model, chat_tokenizer = load_model('./chat_model')
    app.state.chat_model = chat_lllm
    #app.state.chat_tokenizer = chat_tokenizer
    print('Chat Model Loaded Successfully!')
    recomm_model = recommendation_llm
    app.state.recomm_model = recommendation_llm
    print('Recommendation Model Loaded Successfully!')
    #bert_model = load_bert()
    #app.state.bert_model = bert_model

    yield

app = FastAPI(title = 'LLM Services', lifespan=lifespan)


async def generation(prompt, model, tokenizer, device):
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    attention_mask = inputs["attention_mask"]
    with torch.no_grad():
        output_ids = model.generate(
            inputs['input_ids'],
            max_length=500,
            no_repeat_ngram_size=1,
            top_k=5,
            temperature=0.7,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            attention_mask=attention_mask,
            top_p=0.95,
            use_cache=True,
            output_scores=True,
            return_dict_in_generate=True
        )

        generated_ids = output_ids.sequences  # Extract generated token IDs
        response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    print(response_text)
    return response_text 

class QueryRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    stop: list = []

@app.post('/chatbot')
async def response(request: QueryRequest):
    # print(f'This is the request body : {request_body}')
    # response = await generation(request_body.prompt, app.state.chat_model,app.state.chat_tokenizer, 'cuda')
    # return response
    async def token_generator() -> AsyncGenerator[str, None]:
        def sync_generator() -> Iterator[str]:
            response = app.state.chat_model(
                prompt = request.prompt,
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

class SentimentModelPydantic(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    stop: list = []

class SentimentUserText(BaseModel):
    user_text:str

async def sentiment_format_text(user_text, sentiment):
    prompt = f"""
    The user said: "{user_text}"
    The sentiment of the user is: {sentiment}.

    Based on the user's sentiment, generate 3 personalized recommendations to help the user feel better and stay 
    engaged for the day. Focus on activities that are uplifting, calming, or motivating.

    Recommendations:
    """
    return prompt

@app.post('/sentiment_analysis')
async def bert_sentiment_analysis(text_obj:SentimentUserText):
    user_text = text_obj.user_text if text_obj.user_text is not None else ''
    sentiment = await bert_inference(user_text, app.state.bert_model)
    
    #return sentiment
@app.post('/recommendation_analysis')
async def bert_recommendation(request: SentimentModelPydantic):
    async def token_generator() -> AsyncGenerator[str, None]:
        def sync_generator() -> Iterator[str]:
            response = app.state.recomm_model(
                prompt = request.prompt,
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

    
    

if __name__ == "__main__":
    uvicorn.run('inference:app', host="0.0.0.0", port=2001, reload=True) #reload = True