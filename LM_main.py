import math
import torch
from pytorch_transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
import json

#json_data = '{"activation_function": "gelu_new","architectures": ["GPT2LMHeadModel"],"attn_pdrop": 0.1,"bos_token_id": 50256,"embd_pdrop": 0.1,"eos_token_id": 50256,"initializer_range": 0.02,"layer_norm_epsilon": 1e-05,"model_type": "gpt2","n_ctx": 1024,"n_embd": 768,"n_head": 12,"n_layer": 12,"n_positions": 1024,"resid_pdrop": 0.1,"summary_activation": null,"summary_first_dropout": 0.1,"summary_proj_to_labels": true,"summary_type": "cls_index","summary_use_proj": true,"task_specific_params": {"text-generation": {"do_sample": true,"max_length": 50}}, "vocab_size": 50257}'
#config = json.loads(json_data)

with open('GPT2_QG/gpt2/gpt2-config.json', 'r') as f:
    config = json.load(f)

model = GPT2LMHeadModel.from_pretrained('gpt2', config = config) #force_download=True #config = config
model.eval()

# Load pre-trained model tokenizer (vocabulary)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', config = config)


def perplexity(sentence):
    tokenize_input = tokenizer.tokenize(sentence)
    tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
    loss = model(tensor_input, labels=tensor_input)
    return math.exp(loss[0].item() / len(tokenize_input))


if __name__ == '__main__':
    a = ["i wrote a book, i wrote a book, i wrote a book, i wrote a book,i wrote a book, i wrote a book.",
         "i wrote a book.",
         "i wrote a book about the life of two young people who fall in love with each other."]

    print([perplexity(i) for i in a])
