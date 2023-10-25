 import torch
from transformers import BartForConditionalGeneration, BartTokenizer

model_name = "facebook/bart-base"
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)

input = [
    "Long live America.",
    "I love Huggingface as it has made the ML development so easy."
]

input_tok = tokenizer(input, return_tensors='pt', padding=True)

with torch.no_grad():
    out = model(input_tok['input_ids'], labels=input_tok['input_ids'])

default_loss = out.loss

# Ensure that the ignore_index matches the pad token ID in the model's configuration
criterion = torch.nn.CrossEntropyLoss(ignore_index=model.config.pad_token_id)

# Reshape logits and input_ids before calculating the loss
logits = out.logits.view(-1, out.logits.size(-1))
input_ids = input_tok['input_ids'].view(-1)
should_be_loss = criterion(logits, input_ids)

print("Default Loss:", default_loss.item())
print("Should Be Loss:", should_be_loss.item())
