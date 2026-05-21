from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file(
    "/home/chirag/.cache/huggingface/hub/models--Qwen--Qwen2-7B-Instruct/snapshots/f2826a00ceef68f0f2b946d945ecc0477ce4450c/tokenizer.json")

encoded = tokenizer.encode(
    "Hello , How are you ? are you fine ? I am absolutely fine , whay about you")
print(encoded.ids)

tokens = [39, 9011, 0, 1246, 525, 498, 30, 358, 2776, 264, 12305, 11, 358, 1513, 944, 614, 15650, 11, 714, 358, 2776, 1588, 311,
          1492, 498, 448, 894, 4755, 476, 9079, 498, 2578, 614, 13, 2585, 646, 358, 1492, 498, 3351, 30, 4710, 40, 646, 1492, 498, 448]

decoded = tokenizer.decode(tokens)

print("Model output in tokens ", tokens)
print(decoded)
