from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

tokenizer_path = hf_hub_download(
    repo_id="Qwen/Qwen2-7B-Instruct",
    filename="tokenizer.json"
)

tokenizer = Tokenizer.from_file(tokenizer_path)

text = """
    Hello How are you ?
"""

encoded = tokenizer.encode(text)

tokens = [198, 262, 21927, 2585, 525, 498, 17607, 265, 25379, 25, 330, 785, 501, 2319, 315, 279, 3910, 374, 1431, 2500, 13, 1084, 374, 264, 2244,
          5101, 369, 4143, 323, 13336, 25992, 13, 576, 3910, 374, 6188, 311, 387, 1196, 21896, 323, 4135, 311, 20876, 13, 1084, 5646]

decoded = tokenizer.decode(tokens)

print(decoded)
