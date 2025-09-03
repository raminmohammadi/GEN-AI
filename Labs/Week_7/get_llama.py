import os
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM

def flush_memory():
    """Force garbage collection to release CPU memory."""
    gc.collect()

if __name__ == '__main__':
    model_id = "meta-llama/Meta-Llama-3-8B"
    cache_dir = "./llama3_cache"
    save_dir = "./local_llama3"
    access_token = ""

    try:
        # ✅ Step 1: Load and save tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, token=access_token)
        tokenizer.save_pretrained(save_dir)
        print("Tokenizer saved successfully!")

        # ✅ Step 2: Release tokenizer memory
        del tokenizer
        flush_memory()
        print("Tokenizer memory released!")

        # ✅ Step 3: Check if model is already partially downloaded
        model_files = os.listdir(cache_dir) if os.path.exists(cache_dir) else []
        if any(f.endswith(".bin") or f.endswith(".safetensors") for f in model_files):
            print("Detected partially downloaded model in cache. Resuming from cache...")

        # ✅ Step 4: Load model (resume if cached files exist)
        model = AutoModelForCausalLM.from_pretrained(
            cache_dir,  # Directly load from cache
            token=access_token
        )
        
        model.save_pretrained(save_dir)
        print("Model saved successfully!")

        # ✅ Step 5: Release model memory
        del model
        flush_memory()
        print("Model memory released!")

    except Exception as e:
        print(f"Error: {e}")

