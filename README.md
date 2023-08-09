# GO-Submito
GO-Submito: Prediction of submitochondrial proteins localization based on Gene Ontology

The official PyTorch implementation of GO-Submito: Prediction of submitochondrial proteins localization based on Gene Ontology.
GO-Submito utilizes a fine-tuned pre-trained language model and the Multi-head Attention Mechanism to predict the localization of submitochondrial proteins.
![image](https://github.com/Wangbiub/GO-Submito/blob/main/1.png)

# requirements:
GO-Submito requires:

torch: 1.8.0

cuda: 11.3

python: 3.10
# Steps:
To effectively learn the representation of GO annotations, the GO terms encoder module fine-tunes the pre-trained BERT model using a GO-specific corpus via SimCSE. The code is showed in pre-train-simcse.py

The input GO anntenitons are encoded by a fine-tuned BERT model, converting the features into vectors. The code of the fine-tuned BERT model is showed in simcse424.py


To fuse the features, we utilize a multi-head attention mechanism to combine the input features in the dual fusion block， which is in model424.py.

The output of the multi-head attention mechanism is propagated through a linear layer, which is also included in model424.py. 
# Running Code::
main424.py
