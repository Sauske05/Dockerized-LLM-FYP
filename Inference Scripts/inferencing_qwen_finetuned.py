from transformers import AutoTokenizer, TextStreamer, AutoModelForCausalLM, BitsAndBytesConfig

import torch
#bitsandbytes parameters

#Activate 4-bit precision base model loading
use_4bit = True
#Compute dtype for 4 bit base models
bnb_4bit_compute_dtype = 'float16'
#Quantization type
bnb_4bit_quant_type = 'nf4'
#Activate double quantization??
use_double_nested_quant = False
# BitsAndBytesConfig int-4 config

compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_use_double_quant=use_double_nested_quant,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype
)

model_name = "Aspect05/Qwen-1.5-Distil-FP16-Updated"

# Load the pretrained model
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, use_cache = True, device_map='cuda')
model.config.pretraining_tp = 1

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


def prompt_format(user):
    text = f''' {tokenizer.bos_token}  You are a converstional chatbot assistant. Your job is to answer queries asked by the user in an empathetic manner. Be brief with you answers. Do not try to repeat them. 
    <｜User｜> {user} <|Assistant|> '''
    return text

input = prompt_format('I am feeling anxious and I dont know why.')

input_ids = tokenizer(input, return_tensors="pt").input_ids.to('cuda')
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
model.generate(
    input_ids=input_ids,
    max_new_tokens=200,
    streamer=streamer,
    temperature=0.7,  # Adjust for creativity
    top_k=5,# Adjust for focused outputs
    eos_token_id=tokenizer.eos_token_id
)