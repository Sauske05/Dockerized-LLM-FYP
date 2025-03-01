from llama_cpp import Llama
local_model_dir = "./Qwen_gguf"


# llm = Llama.from_pretrained(
# 	repo_id="ggml-org/DeepSeek-R1-Distill-Qwen-1.5B-Q4_0-GGUF",
# 	filename="deepseek-r1-distill-qwen-1.5b-q4_0.gguf",
#     local_dir=local_model_dir
# )
#Loading from local dir
llm = Llama(model_path=local_model_dir)