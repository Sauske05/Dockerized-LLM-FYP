import asyncio
from fastapi  import FastAPI, Body, HTTPException, WebSocket
from typing import Annotated, AsyncGenerator, Iterator
from contextlib import asynccontextmanager
import httpx
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextStreamer
import torch
import uvicorn
from pydantic import BaseModel
from bert_model.configure import *
from bert_model.tokenizer import *
from bert_model.model import *
from fastapi.responses import StreamingResponse
from llama_cpp import Llama
from fastapi import FastAPI, Request, Depends
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
import mysql.connector
from mysql.connector import Error
import numpy as np


from sqlalchemy import Text, create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import select
recommendation_local_model_dir = "./chat_model/Llama-3.2-3B-Instruct-IQ3_M.gguf"
#chat_local_model_dir = "./chat_model/llama-3.2_4bit.gguf"
chat_local_model_dir = "./chat_model/Llama-3.2-3B-Instruct-IQ3_M.gguf"
# Initialize the model
# chat_lllm = Llama(
#     model_path=recommendation_local_model_dir,
#     n_ctx=2048,  # Context window size
#     n_threads=4  # Number of CPU threads to use
#     #n_gpu_layers=-1
# )

chat_llm = LlamaCpp(model_path = chat_local_model_dir,
                    n_ctx = 4096)
embedding_model = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

template = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful chatbot. Use the following relevant past conversation to respond:
{context}
<|eot_id|><|start_header_id|>user<|end_header_id|>
{input}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
prompt = PromptTemplate(input_variables=["context", "input"], template=template)

user_vector_stores = {}

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
    app.state.chat_model = chat_llm
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
    user_id:str

# @app.post('/chatbot')
# async def response(request: QueryRequest):
#     # print(f'This is the request body : {request_body}')
#     # response = await generation(request_body.prompt, app.state.chat_model,app.state.chat_tokenizer, 'cuda')
#     # return response
#     if request.user_id not in user_vector_stores:
#         user_vector_stores[request.user_id] = FAISS.from_texts(
#         ['Initial Empty Context'], embedding_model
#         )

#     vector_store = user_vector_stores[request.user_id]
#     docs = vector_store.similarity_search(request.prompt, k = 3)

#     context = '\n'.join([doc.page_content for doc in docs])
#     async def token_generator() -> AsyncGenerator[str, None]:
#         def sync_generator() -> Iterator[str]:
#             response = app.state.chat_model(
#                 prompt = request.prompt,
#                 max_tokens=request.max_tokens,
#                 temperature=request.temperature,
#                 top_p=request.top_p,
#                 stop=request.stop,
#                 stream=True
#             )
#             for chunk in response:
#                 if "choices" in chunk and len(chunk["choices"]) > 0:
#                     token = chunk["choices"][0]["text"]
#                     yield token
        
#         # Convert sync generator to async generator
#         for token in sync_generator():
#             yield token
#             # Small delay to prevent overwhelming the client
#             await asyncio.sleep(0.01)

#     return StreamingResponse(token_generator(), media_type="text/plain")

# Database configuration
DATABASE_URL = "mysql+pymysql://root:@localhost/mentalSathi"
engine = create_engine(DATABASE_URL, echo=True)  # echo=True for SQL query logging

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()

# Define the ChatbotChat model
class ChatbotChat(Base):
    __tablename__ = "chatbot_chat"
    id = Column(Integer, primary_key=True)
    user_id = Column(String(255), index=True)
    context = Column(Text)
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
@app.get("/chats/{chat_id}/message/")
async def get_specific_message(
    chat_id : str,
):
    print(f'Request of the chat inside :{chat_id}')
    #print(f'Request of the chat inside :{request.get('user_id')}')
    try:
        stmt = select(ChatbotChat.context).where(
            (ChatbotChat.user_id == chat_id)
        )
        session = Session(engine)

        results = [row[0] for row in session.execute(stmt).all()]
        if results:
            print(f'This is the result : {results}')
            return {"message": results}
        return {'message' : ''}
        #return {"error": "Message not found"}
    except Exception as e:
        return {"error": str(e)} 
    

@app.post('/chatbot')
async def response(request: QueryRequest):
    print('Reaches Here')
    print(request.user_id)
    print(request.prompt)
    vector_store = FAISS.from_texts([''], embedding_model)
    # Fetch the specific message from the /chats/{chat_id}/message/ endpoint
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"http://localhost:2001/chats/{request.user_id}/message/",
            params={"user_id" : request.user_id}  # Pass user_id if needed
        )
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Failed to fetch message")
        message_data = response.json()
        if "error" in message_data:
            raise HTTPException(status_code=404, detail=message_data["error"])
        previous_message = message_data["message"]
    # Ensure previous_message is a flat list of strings
    if not previous_message:  # Handle empty case
        previous_message = [""]
    elif isinstance(previous_message, str):  # Handle single string case
        previous_message = [previous_message]
    elif isinstance(previous_message, list):  # Handle list case
        # Flatten if nested and ensure all elements are strings
        previous_message = [str(item) for sublist in previous_message if sublist for item in (sublist if isinstance(sublist, list) else [sublist])]
    else:
        previous_message = [""]  # Fallback for unexpected types

    print(f'This is the previous message : {previous_message}')
    # Add the fetched message to the vector store
    print(f'This is the previous message : {previous_message}')
    vector_store.add_texts(previous_message)

    # Search for relevant :ocuments
    docs = vector_store.similarity_search(request.prompt, k=2)
    context = '\n'.join([doc.page_content for doc in docs])
    print(context)

    # Create LangChain components
    parser = StrOutputParser()
    chain = prompt | app.state.chat_model | parser


    postman_text = ''
    async def token_generator() -> AsyncGenerator[str, None]:
        async for token in chain.astream_events(
                {
                    'context': context,
                    'input': request.prompt
                },
                version='v2',
                config={
                    'max_tokens': request.max_tokens,
                    'temperature': request.temperature,
                    'top_p': request.top_p,
                }
        ):
            kind = token['event']
            if kind == 'on_chain_stream':
                chunk = token['data']['chunk']
                print(chunk, end='', flush=True)
                #postman_text += chunk
                yield chunk
                await asyncio.sleep(0.01)
                    
    #print(postman_text)
    return StreamingResponse(token_generator(), media_type='text/plain')
    # Return the complete text
    
class SentimentModelPydantic(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    stop: list = []

class SentimentUserText(BaseModel):
    user_text:str

async def sentiment_format_text(user_text, sentiment):
    # prompt = f"""
    # The user said: "{user_text}"
    # The sentiment of the user is: {sentiment}.

    # Based on the user's sentiment, generate 3 personalized recommendations to help the user feel better and stay 
    # engaged for the day. Focus on activities that are uplifting, calming, or motivating.

    # Recommendations:
    # """

    prompt = f"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful chatbot. Use the following relevant past conversation to respond:
The user said: {user_text}
The sentiment of the user is: {sentiment}.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Based on the user's sentiment, generate 3 personalized recommendations to help the user feel better and stay 
engaged for the day. Focus on activities that are uplifting, calming, or motivating.

Recommendations:
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
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