from fastapi  import FastAPI, Body, WebSocket
from typing import Annotated
from contextlib import asynccontextmanager
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextStreamer
import torch
import uvicorn
from pydantic import BaseModel
from bert_model.configure import *
from bert_model.tokenizer import *
from bert_model.model import *
from concurrent.futures import ThreadPoolExecutor

from fastapi.responses import StreamingResponse
import asyncio
import redis

#redis = redis_client.from_url('redis://localhost', decode_responses = True)
r = redis.Redis(host = 'localhost', port=6379, db=0)
# Load model at startup
def load_model(model_dir):
    device_map = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    #BitsandBytes Config
    use_4bit = True
    bnb_4bit_compute_dtype = 'float16'
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    bnb_4bit_quant_type = 'nf4'
    use_double_nested_quant = False
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_use_double_quant=use_double_nested_quant,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype
)
    #model = AutoModelForCausalLM()
    model = AutoModelForCausalLM.from_pretrained(model_dir, quantization_config=bnb_config, use_cache = False, device_map=device_map)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer


def load_bert():
    model = SentimentModel(config()['h'], config()['d_model'], config()['d_ff'], config()['labels'])
    model.load_state_dict(torch.load('./bert_model/model_state_dict.pth', weights_only=True))
    return model

def bert_inference(input_text, bert_model):
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
        return  input_text,label_dict[model_idx]
@asynccontextmanager
async def lifespan(app:FastAPI):
    chat_model, chat_tokenizer = load_model('./chat_model')
    app.state.chat_model = chat_model
    app.state.chat_tokenizer = chat_tokenizer
    print('Chat Model Loaded Successfully!')
    recomm_model, recomm_tokenizer = load_model('./recommendation_model')
    app.state.recomm_model = recomm_model
    app.state.recomm_tokenizer = recomm_tokenizer
    print('Recommendation Model Loaded Successfully!')
    bert_model = load_bert()
    app.state.bert_model = bert_model

    yield

app = FastAPI(title = 'Chatbot API', lifespan=lifespan)


# async def generation(prompt, model, tokenizer, device):
#     streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens = True)
#     inputs = tokenizer(prompt, return_tensors='pt').to(device)
#     attention_mask = inputs["attention_mask"]
#     with torch.no_grad():
#         model.generate(
#             inputs['input_ids'],
#             max_length=500,
#             no_repeat_ngram_size=1,
#             top_k=5,
#             temperature=0.7,
#             eos_token_id=tokenizer.eos_token_id,
#             pad_token_id=tokenizer.pad_token_id,
#             attention_mask=attention_mask,
#             top_p=0.95,
#             use_cache=True,
#             streamer = streamer

#         )

#     # generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     # return generated_text
#     for chunk in streamer.text.split():
#         yield chunk + " "

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

        


# async def generate_streaming_response(prompt: str):
#     async for chunk in generation_streaming(prompt, app.state.chat_model, app.state.chat_tokenizer, "cuda"):
#         chunk = chunk.strip()  # Remove spaces or newlines
#         if not chunk:  # Skip empty tokens
#             continue
        
#         # print(f"Pushing to Redis: {chunk}")  # Debugging print
#         # try:
#         #     await r.publish('my_channel', chunk)
#         # except r.ConnectionError as e:
#         #     print(f"Redis Connection Error: {e}")  # Handle Redis errors
        
#         yield chunk 
#     #return 'Redis Streaming'
#         #yield chunk


class QueryRequest(BaseModel):
    prompt: str
@app.post('/chatbot')
async def response(request_body: QueryRequest):
    print(f'This is the request body : {request_body}')
    response = await generation(request_body.prompt, app.state.chat_model,app.state.chat_tokenizer, 'cuda')
    return response

class SentimentModelPydantic(BaseModel):
    user_text:str


def sentiment_format_text(user_text, sentiment):
    prompt = f"""
    The user said: "{user_text}"
    The sentiment of the user is: {sentiment}.

    Based on the user's sentiment, generate 3 personalized recommendations to help the user feel better and stay 
    engaged for the day. Focus on activities that are uplifting, calming, or motivating.

    Recommendations:
    """
    return prompt
@app.post('/sentiment')
async def bert_recommendation(sentiment_obj: SentimentModelPydantic):
    user_text = sentiment_obj.user_text
    _, sentiment = bert_inference(user_text, app.state.bert_model)
    prompt = sentiment_format_text(user_text, sentiment)
    model = app.state.recomm_model
    tokenizer = app.state.recomm_tokenizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #response = generate_streaming(prompt, model, tokenizer, device)
    #return response

if __name__ == "__main__":
    uvicorn.run('inference:app', host="0.0.0.0", port=8080, reload=True)