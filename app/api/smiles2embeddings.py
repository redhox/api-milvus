# %pip install torch transformers pandas -q
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import pandas as pd
import torch.nn as nn


class ModelChemBERTa:
    def __init__(self):
        self.chemberta = AutoModelForMaskedLM.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
        self.tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
        self.chemberta._modules["lm_head"] = nn.Identity()

    def smiles2embeding_cls(self,smiles):
        with torch.no_grad():
            encoded_input = self.tokenizer(smiles, return_tensors="pt",padding=True,truncation=True)
            model_output = self.chemberta(**encoded_input)
            embeddings_cls = model_output[0][:,0,:]
            return embeddings_cls
    def smiles2embeding_mean(self,smiles):
        with torch.no_grad():
            encoded_input = self.tokenizer(smiles, return_tensors="pt",padding=True,truncation=True)
            model_output = self.chemberta(**encoded_input)
            embeddings_mean = torch.mean(model_output[0],1)
            return embeddings_mean
    
