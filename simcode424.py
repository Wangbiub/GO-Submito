import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers import models, losses
from torch.utils.data import DataLoader

df = pd.read_csv('./data424.csv')
proteins_arr = df['Protein']
id_namedef_dicts = joblib.load('./id_namedefi_dict')

def extrtactembeddingbymodel(model,Protein,des_list):
    prot_emb_dict = {}

    feature = model.encode(des_list, convert_to_tensor=True)
    # feature = feature.cpu().numpy()
    # embedding = torch.zeros((max_go, 768))
    # avg = feature.mean(0)
    # prot_emb_dict[protein] = avg
    max_go = 35
    # embedding_real = torch.stack([go_term[t.strip()] for t in item[2].split(';') if t.strip() in id_namedef_dicts.keys()], dim=0)
    embedding = torch.zeros((max_go, 768))

    embedding[:feature.shape[0], :] = feature[:max_go, :]
    # embedding =embedding.cpu().numpy()
    # if embedding_real.size(0) > max_go:
    #      feature = embedding_real[:max_go].squeeze(1)
    # else:
    #       feature[: embedding_real.size(0)] = embedding_real.squeeze(1)
    prot_emb_dict[Protein] = embedding.cpu().numpy()
    return prot_emb_dict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentenceTransformer('./output/oubiobert-base-uncased', device=device)
protein_ = []
for item in tqdm(df.values.tolist()):
    item_list = item[2].split(';')
    go_term_des_text = []
    for go_term in item_list:
        go_term = go_term.strip(' ')
        if go_term in id_namedef_dicts.keys():
            go_term_des_text.append(id_namedef_dicts[go_term])
    prot_emb_list = extrtactembeddingbymodel(model, item[0],  go_term_des_text)
    # max_go = 20
    # embedding_real = torch.stack([go_term[t.strip()] for t in item_list if t.strip() in go_term.keys()], dim=0)
    # feature = torch.zeros((max_go, 768))
    # if embedding_real.size(0) > max_go:
    #      embedding = embedding_real[:max_go].squeeze(1)
    #  else:
    #      embedding[: embedding_real.size(0)] = embedding_real.squeeze(1)
    protein_.append(prot_emb_list)
print(protein_)
column = ['proteinseq', 'code'] #列表头名称

df = pd.DataFrame(list(zip(proteins_arr, protein_)), columns=column)
joblib.dump(df, 'simcode424.JL')
print(df)