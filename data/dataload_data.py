from datasets import load_dataset


ds = load_dataset("wikitext", "wikitext-103-v1")["train"]
lines = []
for entry in ds:
    text = entry["text"]
    text = text.replace("\n", " ")  # remove newline formatting
    text = " ".join(text.split())  # remove sequences of whitespace
    lines.append(text + "\n")

f = open("data.txt", "w")
f.writelines(lines)
f.close()
