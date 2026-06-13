import json
import os

vocab_path = "M:/models/Qwen3-ASR-0.6B/vocab.json"
out_path = "M:/projects/AsrDemo/app/src/main/assets/qwen3-asr/tokens.txt"

if os.path.exists(vocab_path):
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    # Sort by value (ID)
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])

    vocab_size = 151936
    tokens = ["<unk>"] * vocab_size
    for token, id_val in sorted_vocab:
        if id_val < vocab_size:
            tokens[id_val] = token

    with open(out_path, "w", encoding="utf-8") as f:
        for t in tokens:
            # Replace newline and space for display
            display_t = t.replace("\n", "\\n").replace(" ", " ")
            f.write(display_t + "\n")
    print(f"Done. Wrote {len(tokens)} tokens to {out_path}")
else:
    print("Vocab file not found.")
