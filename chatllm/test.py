from huggingface_hub import snapshot_download
model_id="Aspect05/Llama-3.2-3B-Instruct-Mental-Health-FP16"
snapshot_download(repo_id=model_id, local_dir="llama-3.2",
                  local_dir_use_symlinks=False, revision="main")