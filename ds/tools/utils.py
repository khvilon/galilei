
import torch
import numpy as np

from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name_or_path = "sberbank-ai/rugpt3small_based_on_gpt2"

tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)

model = GPT2LMHeadModel.from_pretrained(model_name_or_path, output_hidden_states=True)


def word2vec(text, layers=None): 
    text += " "
    layers = [-4, -3, -2, -1] if layers is None else layers
    encoded = tokenizer.encode_plus(text, return_tensors="pt")
    with torch.no_grad():
        output = model(**encoded)
        # Get all hidden states
        states = output.hidden_states
        # Stack and sum all requested layers
        output = torch.stack([states[i] for i in layers]).mean(dim=0).squeeze() 
        vec = output.mean(dim=0) 
        return vec
    
    


def cosine(x,y):
    return x.T@y / np.sqrt(x.T@x) / np.sqrt(y.T@y)

def strmlcmp(x,y):
    if x is None or y is None: 
        return 0.
    x = word2vec(x, layers=[-1])
    y = word2vec(y, layers=[-1])
    return cosine(x, y)
        