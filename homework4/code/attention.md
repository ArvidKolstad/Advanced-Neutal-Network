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

# Understanding Attention

<div style="background-color: #f0f8ff; border: 2px solid #4682b4; padding: 10px;">
<a href="https://colab.research.google.com/github/DeepTrackAI/DeepLearningCrashCourse/blob/main/Ch08_Attention/ec08_1_attention/attention.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
<strong>If using Colab/Kaggle:</strong> You need to uncomment the code in the cell below this one.
</div>

```python
# !pip install contractions deeplay spacy  # Uncomment if using Colab/Kaggle.
```

This notebook provides a complete code example that demonstrates how the attention mechanism works.


<div style="background-color: #f0f8ff; border: 2px solid #4682b4; padding: 10px;">
<strong>Note:</strong> This notebook contains the Code Example 8-1 from the book  

**Deep Learning Crash Course**  
Giovanni Volpe, Benjamin Midtvedt, Jesús Pineda, Henrik Klein Moberg, Harshith Bachimanchi, Joana B. Pereira, Carlo Manzo  
No Starch Press, San Francisco (CA), 2026  
ISBN-13: 9781718503922  

[https://nostarch.com/deep-learning-crash-course](https://nostarch.com/deep-learning-crash-course)

You can find the other notebooks on the [Deep Learning Crash Course GitHub page](https://github.com/DeepTrackAI/DeepLearningCrashCourse).
</div>


## Obtaining a Matrix Representation of Sentences

Start by writing two sentences ...

```python
query_sentence = "The teacher praised a student because she improved"
key_sentence = "A student asked the teacher to help her improve"
```

... tokenize the two sentences ...

```python
query_tokens = query_sentence.split()
key_tokens = key_sentence.split()
```

... print the tokens ...

```python
print(query_tokens)
print(key_tokens)
```

... download the GloVe embeddings ...

```python
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

... calculate the embeddings of the query and key sentences ...

```python
import numpy as np
import torch

glove_file = os.path.join(glove_folder, "glove.42B.300d.txt")
glove_embed, embed_dim = load_glove_embeddings(glove_file), 300

query_embeddings = get_glove_embeddings(query_tokens, glove_embed, embed_dim)
key_embeddings = get_glove_embeddings(key_tokens, glove_embed, embed_dim)
```

... and print them out.

```python
print(query_embeddings)
print(key_embeddings)
```

## Implementing Dot-Product Attention


Implement a class to perform the dot-product attention ...

```python
import deeplay as dl

class DotProductAttention(dl.DeeplayModule):
    """Dot-product attention."""

    def __init__(self):
        """Initialize dot-product attention."""
        super().__init__()

    def forward(self, queries, keys, values):
        """Calculate dot-product attention."""
        attn_scores = (torch.matmul(queries, keys.transpose(-2, -1))
                       / (keys.size(-1) ** 0.5))
        attn_matrix = torch.nn.functional.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_matrix, values)
        return attn_output, attn_matrix
```

... calculate attention matrix and attention output ...

```python
attention = DotProductAttention()
attn_output, attn_matrix = attention(
    queries=query_embeddings, keys=key_embeddings, values=key_embeddings,
)
```

... implement a function to plot an attention matrix ...

```python
from matplotlib import pyplot as plt
from matplotlib.ticker import FixedLocator

def plot_attention(query_tokens, key_tokens, attn_matrix):
    """Plot attention."""
    fig, ax = plt.subplots()
    cax = ax.matshow(attn_matrix, cmap="Greens")
    fig.colorbar(cax)
    ax.xaxis.set_major_locator(FixedLocator(range(len(key_tokens))))
    ax.yaxis.set_major_locator(FixedLocator(range(len(query_tokens))))
    ax.set_xticklabels(key_tokens, rotation=90)
    ax.set_yticklabels(query_tokens)
    plt.show()
```

... and use it to plot the attention matrix.

```python
plot_attention(query_tokens, key_tokens, attn_matrix.detach().squeeze())
```

## Making the Attention Mechanism Trainable


Implement a class to perform a trainable dot-product attention ...

```python
class TrainableAttention(dl.DeeplayModule):
    """Trainable dot-product attention."""

    def __init__(self, num_in_features=300, num_out_features=256):
        """Initialize dot-product attention."""
        super().__init__()
        self.Wq = torch.nn.Linear(num_in_features, num_out_features)
        self.Wk = torch.nn.Linear(num_in_features, num_out_features)
        self.Wv = torch.nn.Linear(num_in_features, num_out_features)

    def forward(self, queries, keys, values):
        """Calculate dot-product attention with linear transformations."""
        Q, K, V = self.Wq(queries), self.Wk(keys), self.Wv(values)
        attn_scores = (torch.matmul(Q, K.transpose(-2, -1))
                       / (K.size(-1) ** 0.5))
        attn_matrix = torch.nn.functional.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_matrix, V)
        return attn_output, attn_matrix
```

... calculate it ...

```python
trainable_attention = TrainableAttention()
trainable_attn_output, trainable_attn_matrix = trainable_attention(
    queries=query_embeddings, keys=key_embeddings, values=key_embeddings,
)
```

... and plot the resulting attention matrix.

```python
plot_attention(query_tokens, key_tokens,
               trainable_attn_matrix.detach().squeeze())
```

## Implementing Additive Attention


Implement a class to perform additive attention ...

```python
class AdditiveAttention(dl.DeeplayModule):
    """Additive attention."""

    def __init__(self, num_in_features=300, num_out_features=256):
        """Initialize additive attention."""
        super().__init__()
        self.Wq = torch.nn.Linear(num_in_features, num_out_features)
        self.Wk = torch.nn.Linear(num_in_features, num_out_features)
        self.Ws = torch.nn.Linear(num_out_features, 1)

    def forward(self, queries, keys, values):
        """Calculate additive attention."""
        Q, K = self.Wq(queries), self.Wk(keys)
        attn_scores = self.Ws(
            torch.tanh(Q.unsqueeze(-2) + K.unsqueeze(-3))
        ).squeeze(-1)
        attn_matrix = torch.nn.functional.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_matrix, values)
        return attn_output, attn_matrix
```

... calculate it ...

```python
additive_attention = AdditiveAttention()
additive_attn_output, additive_attn_matrix = additive_attention(
    queries=query_embeddings, keys=key_embeddings, values=key_embeddings,
)
```

... and plot the resulting attention matrix.

```python
plot_attention(query_tokens, key_tokens, 
               additive_attn_matrix.detach().squeeze())
```
