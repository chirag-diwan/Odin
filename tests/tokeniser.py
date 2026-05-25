from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

# Download tokenizer.json from Hugging Face
tokenizer_path = hf_hub_download(
    repo_id="meta-llama/Llama-3.2-1B",
    filename="tokenizer.json"
)

# Load tokenizer
tokenizer = Tokenizer.from_file(tokenizer_path)

text = "Hello how are you my friend ?"

# Tokenize
encoded = tokenizer.encode(text)

print("Token IDs:", encoded.ids)
print("Tokens:", encoded.tokens)
