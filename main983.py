#%%
import warnings
import random
from dataset import *
from model import *
from sklearn.model_selection import  StratifiedKFold
from sklearn import metrics
from sklearn.metrics import  accuracy_score
import torch.nn as nn
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
class GOSubModel(nn.Module):
    def __init__(self):
        super(GOSubModel, self).__init__()

        self.mha =  nn.MultiheadAttention(768, 4, batch_first=True)
        self.linear = nn.Linear(768, 3)


    def forward(self, embedding):
        att_mask = torch.all(embedding == 0, -1)
        embedding = embedding.float()
        # print(embedding.shape)
        attn_output, attn_output_weights = self.mha(embedding, embedding, embedding,
                                                    key_padding_mask=att_mask, average_attn_weights=False)
        max_embeddings = torch.max(attn_output, 1)[0]
        output = self.linear(max_embeddings)
        return output

warnings.filterwarnings("ignore")

SEED = 79
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True




def protein_type(s):
    it = {b'Inner Membrane': 0, b'Matrix': 1, b'Outer membrane': 2}
    return it[s]

def GetGcc(y_pre, y_true, K=3):
    M_matrix = np.zeros((3, 3))
    a = np.zeros((3, 1))
    b = np.zeros((3, 1))
    e = np.zeros((3, 3))
    for i in range(len(y_true)):
        M_matrix[int(y_true[i])][int(y_pre[i])] += 1

    for i in range(0, 3):
        a[i] = M_matrix.sum(axis=1)[i]
        b[i] = M_matrix.sum(axis=0)[i]

    for i in range(0, 3):
        for j in range(0, 3):
            e[i][j] = a[i] * b[j] / len(y_true)
    gcc = 0.0
    for i in range(0, 3):
        for j in range(0, 3):
            if e[i][j] == 0:
                break
            else:
                gcc += (M_matrix[i][j] - e[i][j]) ** 2 / e[i][j]
    GCC = (gcc / (len(y_true) * (K - 1))) ** 0.5
    return GCC


def train(model, dataloader, optimizer, device, criterion):
    loss_all = 0
    for X, y in tqdm(dataloader):
        X, y = X.to(device), y.to(device)
        output = model(X)
        loss = criterion(output, y.long())
        model.zero_grad()
        loss.backward()
        optimizer.step()
        loss_all = loss_all + loss.item()
    return loss_all / len(dataloader)


def evaluate(model, dataloader, device):
    softmax = nn.Softmax(dim=1)
    preds = torch.Tensor()
    trues = torch.Tensor()

    with torch.no_grad():
        for X, y in tqdm(dataloader):
            X, y = X.to(device), y.to(device)
            output = model(X)  # .squeeze(1)
            out = softmax(output)

            _, predictions = torch.max(out, 1)
            preds = torch.cat((preds, predictions.cpu()), 0)
            # preds1 = torch.cat((preds1, logits1.cpu()), 0)
            # preds2 = torch.cat((preds2, logits2.cpu()), 0)
            trues = torch.cat((trues, y.view(-1, 1).cpu()), 0)

        preds, trues = preds.numpy(), trues.numpy()
    return preds, trues


# res = []
# columns = [ 'GO','Classifier', 'Mcc', 'Mcc-I', 'Mcc-M', 'Mcc-O',
#                                  'Gcc']


df = pd.read_csv('./sm983.csv')
proteins_arr = df['protein']
id_namedef_dicts = joblib.load('./id_namedefi_dict')

def extrtactembeddingbymodel(model,Protein,des_list):
    prot_emb_dict = {}

    feature = model.encode(des_list, convert_to_tensor=True)
    # feature = feature.cpu().numpy()
    # embedding = torch.zeros((max_go, 768))
    # avg = feature.mean(0)
    # prot_emb_dict[protein] = avg
    max_go = 35
    embedding = torch.zeros((max_go, 768))

    embedding[:feature.shape[0], :] = feature[:max_go, :]
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
    protein_.append(prot_emb_list)
print(protein_)
column = ['proteinseq', 'code'] #列表头名称

df = pd.DataFrame(list(zip(proteins_arr, protein_)), columns=column)
joblib.dump(df, 'simcode983.JL')
print(df)
#%%
df = joblib.load('simcode983.JL')

tag = np.loadtxt('sm983.csv', dtype=object, delimiter=',', converters={1: protein_type}, skiprows=1)[:, 1]

inputs = []
for _, row in df.iterrows():
    inputs.append(list((row[1].values()))[0])
for col in df.columns[1:2]:
    X = inputs
    y = np.array(tag, dtype=int)
    cal = []
    KF = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    folder = 0
    for train_index, test_index in KF.split( X, y):
        train_x = np.array(X)[train_index]
        test_x = np.array(X)[test_index]
        train_y = np.array(y)[train_index]
        test_y = np.array(y)[test_index]
        train_dataset = GOSubDataset(train_x, train_y)
        test_dataset = GOSubDataset(test_x, test_y)
        BATCH_SIZE = 32
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        LEARNING_RATE = 1e-3
        model = GOSubModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        N_EPOCHS = 10
        for epoch in range(N_EPOCHS):
            loss_avg = train(model, train_dataloader, optimizer, device, criterion)
            print(f"Epoch {epoch + 1}")
            print(loss_avg)
        train(model, train_dataloader, optimizer, device, criterion)

        y_pred, test_y = evaluate(model, test_dataloader, device)

        print(y_pred.shape, test_y.shape)
        # print(y_pred,test_y)
        metric_result = {
            'Folder': folder,
            # 'Acc_all': accuracy_score(test_y, y_pred),
            'Mcc_all': metrics.matthews_corrcoef(test_y, y_pred),
            # 'conf_mat': confusion_matrix(test_y, y_pred)
             'Gcc_all': GetGcc(test_y, y_pred)
        }
        # metric_result.update({
        #     'Acc_%d' % i: accuracy_score(test_y == i, y_pred == i) for i in range(3)
        # })
        metric_result.update({
            'Mcc_%d' % i: metrics.matthews_corrcoef(test_y == i, y_pred == i) for i in range(3)
        })

        # conf_mat = confusion_matrix(test_y, y_pred)
        folder = folder + 1
        cal.append(metric_result)

    row = pd.DataFrame(cal).set_index('Folder')
    row.to_csv('metricsim983.csv')
    print(row)
    print(row.mean(axis=0))
    #%%
