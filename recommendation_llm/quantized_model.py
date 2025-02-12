from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

device_map = 'cuda' if torch.cuda.is_available() else 'cpu'

#BitsandBytes Config
use_4bit = True
bnb_4bit_compute_dtype = 'float16'
bnb_4bit_quant_type = 'nf4'
use_double_nested_quant = False

compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_use_double_quant=use_double_nested_quant,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype
)
#Saving the Model Locally for faster load time when initializing the docker containers
model_name = "meta-llama/Llama-3.2-1B-Instruct"
save_directory = './model'
def qunatize_save_model(model_name, save_directory):
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, use_cache = False, device_map=device_map)
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"


    tokenizer.save_pretrained(save_directory)
    model.save_pretrained(save_directory)

qunatize_save_model(model_name, save_directory)