import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import os
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
#import plotly.express as px
from torchvision.transforms import transforms
import pickle
class BERTEmbeddingTransform(object):
    def __init__(self, bert_model, tokenizer, device='cpu'):
        bert_model.eval()
        bert_model = bert_model.to(device)
        bert_model.share_memory()
        self.bert_model = bert_model
        self.tokenizer = tokenizer
        self.device = device
    
    def __call__(self, sample):
        code_tokens=self.tokenizer.tokenize(sample)
        tokens = code_tokens
        tokens_ids=self.tokenizer.convert_tokens_to_ids(tokens)
        done_tok = torch.split(torch.tensor(tokens_ids, device=self.device), 510)
        with torch.no_grad():
            embedings = []
            for input_tok in done_tok:
                input_tok = torch.cat((torch.tensor([0], device=self.device), input_tok, torch.tensor([2], device=self.device)))
                embedings.append(self.bert_model(input_tok.clone().detach()[None,:]).last_hidden_state.mean(dim=1))
            return torch.flatten(torch.concat(embedings,dim=0).squeeze())
            
tokenizer = AutoTokenizer.from_pretrained("neulab/codebert-cpp")
BERT = AutoModel.from_pretrained("neulab/codebert-cpp", add_pooling_layer = False)
bert_transform = BERTEmbeddingTransform(BERT,tokenizer, device="cuda:4")
df = pd.read_orc("code_contests_cf_filtered_exploded_truncated.snappy.orc")
df = df.sort_values(by=["problem_url", "submission_id"])
#df.drop(labels="Unnamed: 0", axis=1, inplace=True)
df = df.reset_index(drop=True)
df.head()
slice_pt = []
for i in tqdm(range(len(df))):
    if not pd.isna(df.source_code[i]):
        try:
            slice_pt.append(bert_transform(df.source_code[i]).reshape(-1,768).mean(dim=0))
        except:
            print(i)
            break
    else:
        slice_pt.append(None)
with open("embeddings_CodeBERTcpp.pkl", 'wb') as f:
    pickle.dump(slice_pt, f)