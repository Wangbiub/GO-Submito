# GO-Submito
GO-Submito: Prediction of submitochondrial proteins localization based on Gene Ontology

The official PyTorch implementation of GO-Submito: Prediction of submitochondrial proteins localization based on Gene Ontology.
GO-Submito utilizes a fine-tuned pre-trained language model and the Multi-head Attention Mechanism to predict the localization of submitochondrial proteins.

# requirements:
GO-Submito requires:

torch: 1.8.0

cuda: 11.3

python: 3.10
# Steps:
To effectively learn the representation of GO annotations, the GO terms encoder module fine-tunes the pre-trained BERT model using a GO-specific corpus via SimCSE. The code is showed in pre-train-simcse.py

The input GO anntenitons are encoded by a fine-tuned BERT model, converting the features into vectors. The code of the fine-tuned BERT model is showed in simcse.py

In the output of the GO encoder, vectors are input into the projector module, mapping the inputs to the same dimension. The dataset.py handles the input data, in model_h.py the inputs are mapped into the same dimension.

To fuse the features, we utilize a multi-head attention mechanism to extract and combine the input features in the dual fusion block. The dual feature fusion mechanism is utilized in the construction of neural networks, which in the model_h.py.

The outputs of the networks are concatenated to obtain the final feature representation, which is propagated through the linear layer, the model_h.py contains the concatenation. The output of the model_h.py is the input of the prediction of the drug pairs in main-split.py.
# Steps:
main-py
