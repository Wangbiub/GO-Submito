#%%
import os
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = "gpu 0"
import joblib
df1 = joblib.load('data/alllist.JL')
print(df1)
#%%
from sentence transformers import SentenceTransformer , InputExample
from sentence transformers import models, losses
from torch.utils.data import Dataloader


#Define your sentence transformer model using CLS pooling
model_name ='seiya/oubiobert-base-uncased'
word_embedding_model = models.Transformer(model_name, max_seq_length=600)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

device = torch.device("cuda:0" if torch.cuda.is available() else "cpu")
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# Define a list with sentences (1k - 100k sentences)
train_sentences = df1
# Convert train sentences to sentence pairs
train_data = [InputExample(texts=[s, s]) for s in train_sentences]

# DataLoader to batch your data
train_dataloader = DataLoader(train_data, batch_size=128, shuffle=True)
#Use the denoising auto-encoder loss
train_loss = losses.MultipleNegativesRankingLoss(model)
# Call the fit method
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=100,
    show_progress_bar=True
)
model .save('output/oubiobert-base-uncased')