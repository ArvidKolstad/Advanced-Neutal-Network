---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: py_env_book
    language: python
    name: python3
---

<!-- #region id="mNB0O236MBX4" -->
# Predicting Sentiment Using a Transformer

<div style="background-color: #f0f8ff; border: 2px solid #4682b4; padding: 10px;">
<a href="https://colab.research.google.com/github/DeepTrackAI/DeepLearningCrashCourse/blob/main/Ch08_Attention/ec08_B_transformer/transformer.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
<strong>If using Colab/Kaggle:</strong> You need to uncomment the code in the cell below this one.
</div>
<!-- #endregion -->

```python
# Uncomment if using Colab/Kaggle.
# !pip install contractions datasets deeplay deeptrack spacy
```

This notebook provides you with a complete code example that predicts the sentiment of movie reviews using a transformer encoder network.


<div style="background-color: #f0f8ff; border: 2px solid #4682b4; padding: 10px;">
<strong>Note:</strong> This notebook contains the Code Example 8-B from the book  

**Deep Learning Crash Course**  
Giovanni Volpe, Benjamin Midtvedt, Jesús Pineda, Henrik Klein Moberg, Harshith Bachimanchi, Joana B. Pereira, Carlo Manzo  
No Starch Press, San Francisco (CA), 2026  
ISBN-13: 9781718503922  

[https://nostarch.com/deep-learning-crash-course](https://nostarch.com/deep-learning-crash-course)

You can find the other notebooks on the [Deep Learning Crash Course GitHub page](https://github.com/DeepTrackAI/DeepLearningCrashCourse).
</div>

<!-- #region id="WtFKPkzYMBX7" -->
## Using the IMDB Dataset

Start by downloading the Large Movie Review Dataset (often referred to as the IMDB dataset, as it’s available at https://huggingface.co/datasets/imdb). It contains 50,000 movie reviews, labeled as positive or negative. The dataset is divided into 25,000 reviews for training and 25,000 reviews for testing.

Download the IMDB dataset ...
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 363, "referenced_widgets": ["3ad650cbf3784ca3ab2ca8f8da87d90c", "b81609451e754a4fba45991044f57d0c", "e14a65b4193244c1bed4c08bb92625dd", "4665b17e4fc24e23ade2d5a8bd24f978", "8c385f7208194810973bcc17240fe60e", "45b0a15056a14a7f81aec19212591c33", "e2b7cd05a61b47979d40afcc38275388", "4f6c3606e30e473dbe2f1289759dc49e", "3d84fe64e31c4327b1c7d2d495fb9545", "386eb297368546b89270e16b0c35ed2d", "59a3a14ce0574a189f0e076f239007ab", "0e46010722304ced8e886b59f28a0e72", "dc4ef68328054e599378c13a0928b7a8", "eedf1be98e6f43f781f176637e35bb5d", "d89f7ca0fbc84a97a74c5454b3c9a234", "0beb916c3aae48338d44540694ebdb80", "a8af1c71d0cd44c58ce85417b174f6d0", "f9159b92471d4f7695cd08fe11bc8f28", "ebe98dd5ce1a4d7c8c7233cbd847c072", "4d09247bea80492a91bf6f8f032bed44", "ef076f570c934b6dab15297ee66baa1e", "27ec6c429c434e4ab91f57079b5c7cb8", "242354c2fe7c49319bdc27552cea5d19", "71021dc439214a078dca7332040fda2b", "4bced9dc6668447b892ea88794ca38e1", "7714a44741be44a69677d5c50ea1c8ac", "72bd54225d794eb39d03641aa25ab8e3", "4bb88aa1af9741a2896376de3c5bfa75", "262502aeaf424f849598956dbbab689b", "b839eecc88314639931aecec5d306c19", "a8b1cd6f35004ed38d387013ca2eda6c", "cde591240e1f425d80e9e709031fed95", "c6bf89e354274a1889ab468376152a23", "0734365b5d944f1e82115a86ec3d3d3a", "b276a83cf8f544b7bf72a6ab993d2c5f", "15456fe110204ad094756e27e350ba38", "362f76ca190f499fa98dbd8437badfc3", "859a8fe11eb34f828ca88b1c7f05cf22", "7d2c8a7b98644515b6af07ee18f658b4", "dca9b77161694c5c9e3111908fefbef6", "e09db5e694c24057b9f601012dd27d20", "e93f815568764bdd85fb1788a2fb7ec8", "3fee98fb02e64c9c9154ee83c9b38073", "39fdca3ae21e4f58a06dab573702ad66", "8a13bdd2b8a844b5ad2c37ab19d2ae0f", "2d8db0fdb1184291997e0ef15bb1760a", "08d53d39bd0945da80b1900553fdb35b", "6c954a73d07d478ba6de87654a0a4e1c", "314dd568a8fc44e58b0f9db3d44c30d0", "e59e7e61fe5049cb94233c413539d84f", "ba189e0fc5394ddd8dba8af044b6ee92", "294e1794d2c4498f9bcb2ab031b4f6d6", "260afe5078a3418cb5e5fe4ead591444", "177e79bc18314497b1c50e92669a2318", "ee56e614b8fc4f2fba9ccedcf1101af2", "0967041f27b84a709b63ae3ace9c72f8", "d99c06ddcfcc4f4ab72a1f7c4b7cf390", "3ad0e46bfd254fd39f07adc7072e8602", "8266881145b14158b26e32f35d89fb30", "216b5e512bae407dad8dd7ae490737dc", "5283932efc394be8a590f0f68c2de70a", "b3e9803505c74483837cddcccdcdb0fc", "56757d0e848f44fb9633c1ea7d33aa3d", "815f603316074d58be9cf0259042eeb6", "948d64fde28341d0a3f83c3bea096e39", "3059b28486674e48942aa8570166f561", "8be337c52fef482084429a159c765e6a", "8ea50f1dfa35408696c24466756156dd", "e5fd4c3c0ecb49ea98a4c31f6ea8efe9", "f3c408d6da6a496180649013a08beb1f", "37343e5b1a4b4f40a6b2a5c8b698ad66", "9a5ac54ea47142f0873ed478a8115eba", "8458215834c247a9934aead2adc14504", "aea5c8c2e3174bf5a8bd754ca91e6f5f", "d28282d5e2ee4043924e4b6c96d17a1c", "ee6076fa9032456a8772049287ff0e8a", "f2f220426c6d4eb5bd3c7a2b653f1b00"]} id="rjQmOsocMBX8" outputId="8e704556-3fce-41e6-b71a-d738a33a6d73"
from datasets import load_dataset

dataset = load_dataset("imdb")
```

<!-- #region id="fUA6xl8lMBX9" -->
... splitting the training and validation datasets ...
<!-- #endregion -->

```python id="qk_WJ8-OMBX9"
split = dataset["train"].train_test_split(test_size=0.2,
                                          stratify_by_column="label", seed=42)
train_dataset, val_dataset = split["train"], split["test"]
```

<!-- #region id="lw79eqHOMBX-" -->
... and print some example reviews.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 737} id="po9MlAJmMBX-" outputId="faa10ee5-ef40-4966-b8a8-235881761e62"
import numpy as np
import pandas as pd

samples = train_dataset.select(np.random.randint(0, len(train_dataset), 3))
texts, labels = samples["text"], samples["label"]

df = pd.DataFrame({"Text": texts, "Label": labels})
styled_df = df.style.set_properties(**{"text-align": "left"}).set_table_styles(
    [{"selector": "th", "props": [("text-align", "center")]}]
)
with pd.option_context("display.max_colwidth", None):
    display(styled_df)
```

<!-- #region id="zh51W8ZkMBX_" -->
### Preprocessing the Reviews

Implement a function to tokenize a sentence ...
<!-- #endregion -->

```python
import contractions, re, spacy, unicodedata

tokenizers = {"eng": spacy.blank("en"), "spa": spacy.blank("es")}

regular_expression = r"^[a-zA-Z0-9áéíóúüñÁÉÍÓÚÜÑ.,!?¡¿/:()]+$"
pattern = re.compile(unicodedata.normalize("NFC", regular_expression))

def tokenize(text, lang="eng"):
    """Tokenize text."""
    swaps = {"’": "'", "‘": "'", "“": '"', "”": '"', "´": "'", "´´": '"'}
    for old, new in swaps.items():
        text = text.replace(old, new)
    text = contractions.fix(text) if lang == "eng" else text
    tokens = tokenizers[lang](text)
    return [token.text for token in tokens if pattern.match(token.text)]
```

### Building a Vocabulary

Implement a class to represent a vocabulary ...

```python
class Vocab:
    """Vocabulary as callable dictionary."""

    def __init__(self, vocab_dict, unk_token="<unk>"):
        """Initialize vocabulary."""
        self.vocab_dict, self.unk_token = vocab_dict, unk_token
        self.default_index = vocab_dict.get(unk_token, -1)
        self.index_to_token = {idx: token for token, idx in vocab_dict.items()}

    def __call__(self, token_or_tokens):
        """Return the index(es) for given token or list of tokens."""
        if not isinstance(token_or_tokens, list):
            return self.vocab_dict.get(token_or_tokens, self.default_index)
        else:
            return [self.vocab_dict.get(token, self.default_index)
                    for token in token_or_tokens]

    def set_default_index(self, index):
        """Set default index for unknown tokens."""
        self.default_index = index

    def lookup_token(self, index_or_indices):
        """Retrieve token corresponding to given index or list of indices."""
        if not isinstance(index_or_indices, list):
            return self.index_to_token.get(int(index_or_indices),
                                           self.unk_token)
        else:
            return [self.index_to_token.get(int(index), self.unk_token)
                    for index in index_or_indices]

    def get_tokens(self):
        """Return a list of tokens ordered by their index."""
        tokens = [None] * len(self.index_to_token)
        for index, token in self.index_to_token.items():
            tokens[index] = token
        return tokens

    def __iter__(self):
        """Iterate over the tokens in the vocabulary."""
        return iter(self.vocab_dict)

    def __len__(self):
        """Return the number of tokens in the vocabulary."""
        return len(self.vocab_dict)

    def __contains__(self, token):
        """Check if a token is in the vocabulary."""
        return token in self.vocab_dict
```

... implement a function to build vocabulary from an iterator ...

```python
from collections import Counter

def build_vocab_from_iterator(iterator, specials=None, min_freq=1):
    """Build vocabulary from an iterator over tokenized sentences."""
    token_freq = Counter(token for tokens in iterator for token in tokens)
    vocab, index = {}, 0
    if specials:
        for token in specials:
            vocab[token] = index
            index += 1
    for token, freq in token_freq.items():
        if freq >= min_freq:
            vocab[token] = index
            index += 1
    return vocab
```

<!-- #region id="SoKLurvyMBYA" -->
... create a vocabulary ...
<!-- #endregion -->

```python id="Rmn3D8zZMBYA"
def imdb_iterator(dataset):
    """Iterate over the IMDB dataset."""
    for sample in dataset:
        yield tokenize(sample["text"])

vocab_dict = build_vocab_from_iterator(imdb_iterator(train_dataset),
                                       specials=["<unk>"], min_freq=10)
vocab = Vocab(vocab_dict, unk_token="<unk>")
vocab.set_default_index(vocab(vocab.unk_token))
```

<!-- #region id="4PUG6tJmMBYA" -->
... and preprocess the training, validation, and testing datasets.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 113, "referenced_widgets": ["b23da6ab42ab497b91c986891f4a62ee", "2ab58b6f3aca4efd88222164c0f351bb", "b5d0d67e26ca4b64822dbd57ea9d6dec", "6eacd464e9064b9188391829b2671c33", "b57b20e51e3246ec84b16445a2b9c2f4", "e9a06c3c330446f991a4e2b59bfa2efd", "85996379b7e2418998c1b302baf41218", "9aaccefc50794e2aae2164f3f5bdeb6c", "b82313a36e224f95a973ba0964cc66ef", "1f87e24487a443648228d8b1edbb5ebf", "cb660375fe834a6b9925db2e9e4fffb3", "f0b340d4f57b44d5a55faacc0aae65cf", "fc3f3df04e924b6ab50f4699d21aa719", "6034b7c72f9844499994b2643e240522", "60a06e9fbffb444b877d76786ed2cae3", "4624650247e943ac9d33c98d86a731ee", "56eb9b6e767f43b9ba551a5692d5ec4e", "987d39044b11452ea917674efa54d2df", "704dad7685da4a8494548b3a9b239893", "8e023624a67f40f7b686d88936a908c7", "d86f72b5a00d4dc4855df00871bfd3ed", "4313af56060d407a99d74d1f05a84ae6", "b3b96bc1127f4b83bdae451a9e1aa97a", "6fb57e234a97417e85ef5366b6a02de3", "e5c3ff1a81d646a48af0810b6502d153", "18a7583290a3479aa5358d347f5dbc6d", "ea735a6454a44a7f9834dd6ba8ef87cc", "9bc57111b0b846f99a02d85683c2bddb", "8a84682289d2438d8e4b3fdaa6e93870", "51162b91deda4ec9b7a438175aa28ed6", "9fb7664bd8464c79ba17276b2387bda0", "a011c7433d694c47864523280a3ff3a3", "6ff782c38da64450824a979b7d42bfb1"]} id="ojJAcCzMMBYA" outputId="45297cb3-cb61-4c07-c0bf-61deab817cb1"
def preprocessing(sample):
    """Preprocess a movie review."""
    sentence = sample["text"]
    tokens = tokenize(unicodedata.normalize("NFC", sentence))
    sequence_of_indices = vocab(tokens)
    sample.update({"sequences": sequence_of_indices})
    return sample

train_dataset = train_dataset.map(preprocessing)
val_dataset = val_dataset.map(preprocessing)
test_dataset = dataset["test"].map(preprocessing)
```

<!-- #region id="jCIDExb6MBYD" -->
## Defining the Data Loaders
<!-- #endregion -->

```python
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data

def collate(batch_of_sequences):
    """Prepare a batch of sequences for the model to process."""
    sequences, labels, batch_indices = [], [], []
    for batch_index, sample in enumerate(batch_of_sequences):
        sequence = torch.tensor(sample["sequences"])
        sequences.append(sequence)
        batch_indices.append(torch.ones_like(sequence, dtype=torch.long)
                             * batch_index)
        label = torch.tensor(sample["label"])
        labels.append(label)
    return Data(sequences=torch.cat(sequences),
                batch_indices=torch.cat(batch_indices),
                y=torch.tensor(labels).float())

train_dataloader = \
    DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate)
val_dataloader = \
    DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate)
test_dataloader = \
    DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate)
```

<!-- #region id="MYEdAsfBMBYB" -->
## Building a Transformer Encoder Layer

Prepare a class to implement a multi-head attention layer ...
<!-- #endregion -->

```python
import deeplay as dl

class MultiHeadAttentionLayer(dl.DeeplayModule):
    """Multi-head attention layer with masking."""

    def __init__(self, num_features, num_heads):
        """Initialize multi-head attention."""
        super().__init__()
        self.num_features, self.num_heads = num_features, num_heads
        self.head_dim = num_features // num_heads  # Must be integer

        self.Wq = dl.Layer(torch.nn.Linear, num_features, num_features)
        self.Wk = dl.Layer(torch.nn.Linear, num_features, num_features)
        self.Wv = dl.Layer(torch.nn.Linear, num_features, num_features)
        self.Wout = dl.Layer(torch.nn.Linear, num_features, num_features)

    def forward(self, in_sequence, batch_indices):
        """Apply the multi-head attention mechanism to the input sequence."""
        seq_len, embed_dim = in_sequence.shape
        Q = self.Wq(in_sequence)
        Q = Q.view(seq_len, self.num_heads, self.head_dim).permute(1, 0, 2)
        K = self.Wk(in_sequence)
        K = K.view(seq_len, self.num_heads, self.head_dim).permute(1, 0, 2)
        V = self.Wv(in_sequence)
        V = V.view(seq_len, self.num_heads, self.head_dim).permute(1, 0, 2)

        attn_scores = (torch.matmul(Q, K.transpose(-2, -1))
                       / (self.head_dim ** 0.5))

        attn_mask = torch.eq(batch_indices.unsqueeze(1),
                             batch_indices.unsqueeze(0))
        attn_mask = attn_mask.unsqueeze(0)
        attn_scores = attn_scores.masked_fill(attn_mask == False,
                                              float("-inf"))

        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.permute(1, 0, 2).contiguous()
        attn_output = attn_output.view(seq_len, self.num_features)
        return self.Wout(attn_output)
```

<!-- #region id="f60v6Gw4MBYB" -->
... and a class to implement a transformer encoder layer ...
<!-- #endregion -->

```python id="A_luVYrpMBYC"
from torch_geometric.nn.norm import LayerNorm

class TransformerEncoderLayer(dl.DeeplayModule):
    """Transformer encoder layer."""

    def __init__(self, num_features, num_heads, feedforward_dim, dropout=0.0):
        """Initialize transformer encoder layer."""
        super().__init__()

        self.self_attn = MultiHeadAttentionLayer(num_features, num_heads)
        self.attn_dropout = dl.Layer(torch.nn.Dropout, dropout)
        self.attn_skip = dl.Add()
        self.attn_norm = dl.Layer(LayerNorm, num_features, eps=1e-6)

        self.feedforward = dl.Sequential(
            dl.Layer(torch.nn.Linear, num_features, feedforward_dim),
            dl.Layer(torch.nn.ReLU),
            dl.Layer(torch.nn.Linear, feedforward_dim, num_features),
        )
        self.feedforward_dropout = dl.Layer(torch.nn.Dropout, dropout)
        self.feedforward_skip = dl.Add()
        self.feedforward_norm = dl.Layer(LayerNorm, num_features, eps=1e-6)

    def forward(self, in_sequence, batch_indices):
        """Refine sequence via attention and feedforward layers."""
        attns = self.self_attn(in_sequence, batch_indices)
        attns = self.attn_dropout(attns)
        attns = self.attn_skip(in_sequence, attns)
        attns = self.attn_norm(attns, batch_indices)

        out_sequence = self.feedforward(attns)
        out_sequence = self.feedforward_dropout(out_sequence)
        out_sequence = self.feedforward_skip(attns, out_sequence)
        out_sequence = self.feedforward_norm(out_sequence, batch_indices)

        return out_sequence
```

<!-- #region id="3K0WhlqCMBYC" -->
## Building a Transformer Encoder Model

Build a class to implement a transformer encoder model ...
<!-- #endregion -->

```python id="xHuCs0QfMBYC"
class TransformerEncoderModel(dl.DeeplayModule):
    """Transformer encoder model."""

    def __init__(self, vocab_size, num_features, num_heads, feedforward_dim,
                 num_layers, out_dim, dropout=0.0):
        """Initialize transformer encoder model."""
        super().__init__()
        self.num_features = num_features

        self.embedding = dl.Layer(torch.nn.Embedding, vocab_size, num_features)

        self.pos_encoder = dl.IndexedPositionalEmbedding(num_features)
        self.pos_encoder.dropout.configure(p=dropout)

        self.transformer_block = dl.LayerList()
        for _ in range(num_layers):
            self.transformer_block.append(TransformerEncoderLayer(
                    num_features, num_heads, feedforward_dim, dropout=dropout,
            ))

        self.out_block = dl.Sequential(
            dl.Layer(torch.nn.Dropout, dropout),
            dl.Layer(torch.nn.Linear, num_features, num_features // 2),
            dl.Layer(torch.nn.ReLU),
            dl.Layer(torch.nn.Linear, num_features // 2, out_dim),
            dl.Layer(torch.nn.Sigmoid),
        )

    def forward(self, dict):
        """Predict sentiment of movie reviews."""
        in_sequence, batch_indices = dict["sequences"], dict["batch_indices"]

        embeddings = self.embedding(in_sequence) * self.num_features ** 0.5
        pos_embeddings = self.pos_encoder(embeddings, batch_indices)

        out_sequence = pos_embeddings
        for transformer_layer in self.transformer_block:
            out_sequence = transformer_layer(out_sequence, batch_indices)

        batch_size = torch.max(batch_indices) + 1
        aggregates = torch.zeros(batch_size, self.num_features,
                                 device=out_sequence.device)
        for batch_index in torch.unique(batch_indices):
            mask = batch_indices == batch_index
            aggregates[batch_index] = out_sequence[mask].mean(dim=0)

        pred_sentiment = self.out_block(aggregates).squeeze()
        return pred_sentiment
```

<!-- #region id="tNh6P0EpMBYC" -->
... instantiate the transformer encoder model ...
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="3XnTCyXCMBYC" outputId="917bf406-4930-482d-c4ab-7faa2ed6ef64"
model = TransformerEncoderModel(
    vocab_size=len(vocab), num_features=300, num_heads=12, feedforward_dim=512,
    num_layers=4, out_dim=1, dropout=0.1,
).create()
```

... and print it out.

```python
print(model)
```

<!-- #region id="p2j2pIP2MBYD" -->
## Loading Pretrained Embeddings

Download the GloVe embeddings ...
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="c4K8AuxHMBYD" outputId="dc49c293-a566-40cd-a823-ead3e4e67757"
import os
from torchvision.datasets.utils import download_url, extract_archive

glove_folder = ".glove_cache"
if not os.path.exists(glove_folder):
    os.makedirs(glove_folder, exist_ok=True)
    url = "https://nlp.stanford.edu/data/glove.42B.300d.zip"
    download_url(url, glove_folder)
    zip_filepath = os.path.join(glove_folder, "glove.42B.300d.zip")
    extract_archive(zip_filepath, glove_folder)
    os.remove(zip_filepath)
```

... implement a function to load the GloVe embeddings ...

```python
def load_glove_embeddings(glove_file):
    """Load GloVe embeddings."""
    glove_embeddings = {}
    with open(glove_file, "r", encoding="utf-8") as file:
        for line in file:
            values = line.split()
            word = values[0]
            glove_embeddings[word] = np.round(
                np.asarray(values[1:], dtype="float32"), decimals=6,
            )
    return glove_embeddings
```

... implement a function to get GloVe embeddings for a vocabulary ...

```python
def get_glove_embeddings(vocab, glove_embeddings, embed_dim):
    """Get GloVe embeddings for a vocabulary."""
    embeddings = torch.zeros((len(vocab), embed_dim), dtype=torch.float32)
    for i, token in enumerate(vocab):
        embedding = glove_embeddings.get(token)
        if embedding is None:
            embedding = glove_embeddings.get(token.lower())
        if embedding is not None:
            embeddings[i] = torch.tensor(embedding, dtype=torch.float32)
    return embeddings
```

... ad add the GloVe pretrained embeddings.

```python
glove_file = os.path.join(glove_folder, "glove.42B.300d.txt")
glove_embed, embed_dim = load_glove_embeddings(glove_file), 300

model.embedding.weight.data = \
    get_glove_embeddings(vocab.get_tokens(), glove_embed, embed_dim)
model.embedding.weight.requires_grad = False
```

<!-- #region id="eKnDsq7hMBYD" -->
## Training the Model

Compile the model ...
<!-- #endregion -->

```python
classifier = dl.BinaryClassifier(
    model=model, optimizer=dl.AdamW(lr=1e-4),
).create()
```

<!-- #region id="_yYlg2_6MBYD" -->
... and train it.
<!-- #endregion -->

```python
trainer = dl.Trainer(max_epochs=5, accelerator="cpu")  ###
trainer.fit(classifier, train_dataloader, val_dataloader)
```

<!-- #region id="TNPYwlY5MBYE" -->
## Evaluating the Trained Model

Test the trained model ... ...
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 129, "referenced_widgets": ["cda70c4356194ea295635ea57738f22a", "d9c377f40e084edfba414ae5e4edaf07"]} id="itF4Q1kfMBYE" outputId="3ef22fa5-5af2-424d-ca33-f70c97e909f3"
test_results = trainer.test(classifier, test_dataloader)
```

<!-- #region id="h8-NHO3WMBYE" -->
... and display the model’s prediction on some reviews.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 380} id="NMuYkOT2MBYE" outputId="cd0ad688-1c8e-4e9a-d802-0a71d6a67e34"
import random

classifier.model.eval()

texts, labels, predictions = [], [], []
for idx in random.sample(range(len(test_dataset)), 3):
    sample = test_dataset[idx]
    input_sequence = torch.tensor(vocab(tokenize(sample["text"]))).long()
    test_input = {
        "sequences": input_sequence,
        "batch_indices": torch.zeros_like(input_sequence, dtype=torch.long),
    }
    probability = classifier.model(test_input)
    prediction = probability > 0.5

    texts.append(sample["text"])
    labels.append(sample["label"])
    predictions.append(prediction.item() * 1)

df = pd.DataFrame({"text": texts, "label": labels, "prediction": predictions})
styled_df = df.style.set_properties(**{"text-align": "left"}).set_table_styles(
    [{"selector": "th", "props": [("text-align", "center")]}]
)
with pd.option_context("display.max_colwidth", None):
    display(styled_df)
```
