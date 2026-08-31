"""Print LLaMA-3.1-8B-Instruct chat-format token ids in the format
infra_utils.py expects."""
from transformers import AutoTokenizer

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL)

print(f"=== {MODEL} ===")
print(f"pad_token_id (default):            {tok.pad_token_id}")
print(f"eos_token_id:                      {tok.eos_token_id}")
print(f"bos_token_id:                      {tok.bos_token_id}")

# Token sequences for chat headers (mirrors Qwen3 entry: <|im_start|>{role}\n).
# LLaMA-3 format: <|start_header_id|>{role}<|end_header_id|>\n\n
for role in ("system", "user", "assistant", "reflect"):
    text = f"<|start_header_id|>{role}<|end_header_id|>\n\n"
    ids = tok.encode(text, add_special_tokens=False)
    print(f"{role:>10}: {ids}  (decoded: {tok.decode(ids)!r})")

# Also print the special token ids individually for sanity
specials = ["<|begin_of_text|>", "<|end_of_text|>", "<|start_header_id|>",
            "<|end_header_id|>", "<|eot_id|>", "<|finetune_right_pad_id|>"]
print("\n=== individual special tokens ===")
for s in specials:
    ids = tok.encode(s, add_special_tokens=False)
    print(f"{s}: {ids}")
