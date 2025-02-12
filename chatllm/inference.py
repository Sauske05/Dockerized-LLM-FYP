from fastapi  import FastAPI, Body
from typing import Annotated
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import uvicorn

app = FastAPI(title = 'Chatbot API')


def load_model():
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
    model = AutoModelForCausalLM.from_pretrained('./model', quantization_config=bnb_config, use_cache = False, device_map=device_map)
    tokenizer = AutoModelForCausalLM('./model')
    return model, tokenizer

def generation(prompt, model, tokenizer, device):
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    attention_mask = inputs["attention_mask"]
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            max_length=500,
            no_repeat_ngram_size=1,
            top_k=5,
            temperature=0.7,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            attention_mask=attention_mask,
            top_p=0.95,
            use_cache=True

        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text
def format_text(user_tex):
    prompt = f"""
    You are a helpful mental health assistnat. Provide supportive answers.
    """
    return prompt
@app.post('/chatbot')
async def response(user_query: Annotated[str, Body()]):
    prompt = user_query['query']
    model, tokenizer = load_model()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    response = generation(prompt, model,tokenizer, device)
    return response

if __name__ == "__main__":
    uvicorn.run('inference:app', host="localhost", port=8080, reload=True)