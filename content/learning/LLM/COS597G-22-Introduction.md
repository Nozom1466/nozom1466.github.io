---
title: 'COS597G 22 Introduction'
date: 2024-11-15T22:37:35+08:00
slug:
summary:
description:
cover:
    image:
    alt:
    caption:
    relative: false
showtoc: true
draft: false
tags: ['LLM', 'COS597G:2022']
categories:
---

> [Homepage](https://www.cs.princeton.edu/courses/archive/fall22/cos597G/)

## Human Language Understanding & Reasoning
Introductory reading authored by [Christopher D. Manning](https://nlp.stanford.edu/~manning/). 

### Brief introduction of NLP history
The NLP history is divided into four sections, running from the middle of last century to 2 years ago. NLP starts with machine translation in Cold War 1950 - 1969, when researchers on both sides sought to develop systems capable of translating the scientific output of the other nations. The NLP system provided little more the word-level lookups and some simple principle-based mechanisms. 

The second era was from 1970 to 1992 and systems were able to deal with syntax and reference in human language. The new generation of hand-built systems had a clear separation between declarative linguistic knowledge and its procedural processing and which benefited from the development of a range of more modern linguistic theories. 

NLP dramatically changed in the third era, 1993 - 2012 because of the emergence of digital text. At the beginning, researchers tend to extract certain model from a large corpus of data by counting certain facts. Early attempts to learn language structure from text collections were fairly unsuccessful, which led most of the field to concentrate on constructing annotated linguistic resources. Supervised machine learning dominates NLP techniques.

The last era features with deep learning and growing artificial intelligence methods. Word & sentence embedding went viral. From 2013 to 2018, deep learning promotes the advantages of embedding thus leading NLP techiques to vector spaces. In 2018, very large scale self-supervised neural network succeeded in learning an enormous amount of knowledge by simly exposed to large contexts. Representative tasks are net word prediction and filling masked words or phrases.


### Now-dominant neural network
Since 2018, the dominant neural network model for NLP applications has been the transformer neural network. The dominant idea is one of attention, by which a representation at a position is computed as a weighted combination of representations from other positions. Masked word prediction turns out to be very powerful because it is universal: every form of linguistic and world knowledge, from sentence structure, word connotations, and facts about the world, help one to do this task better. As a result, these models assemble a broad general knowledge of the language and world to which they are exposed. 


### What can we do with LPLMs?
Multilingual machine translation trained on all languages simutaneously; for other tasks like QA, sentiment classification, NER and fluent text generation, LPLMs turns out to be the best solution.


### Prospects
What's the meaning in contexts? The dominant approach to describing meaning is a denotational semantics approach or a theory of reference: the meaning of a word, phrase, or sentence is the set of objects or situations in the world that it describes. This contrasts with the simple distributional semantics (or use theory of meaning) of modern empirical work in NLP, whereby the meaning of a word is simply a description of the contexts in which it appears. Manning claims that meaning arises from understanding the network of connections between a linguistic form and other things, whether they be objects in the world or other linguistic forms. Using this definition whereby understanding meaning consists of understanding networks of connections of linguistic forms, there can be no doubt that pretrained language models learn meanings. As well as word meanings, they learn much about the world.

One of the exciting prospects is learning from multi modal data, such as vision, robotics, knowledge graphs, bioinformatics, and multimodal data. Manning also mentions external database as the source of model while he still addresses the importance of multi-modal learning. 

We will witness the comming of foundation models, with its specializations handling most information processing and analysis tasks. There might be concerns of risks that foundation models are controlled by several powerful and influencial groups and somehow it will
be difficult to tell if models are safe to use in particular contexts because the models and their training data are so large. Manning believes in the limitation of models while also gives postive comments on their utility and foresees the future that models are widly deployed. 



## Attention is All You Need 
Transformer architecture is firstly introduced in this work.

### Before Reading
*Attention is All You Need* is well known for its contribution of Transformer architecture and therefore probably be seen as the inception of LLM era. 14k citations well demonstrate its significance. Authors are from Google Brain team or UofT. All of the authors shared the same contribution. Paper was accepted by NIPS 2017.


### Introduction
RNNs established SOTA approaches in sequence modeling while fell short of efficiency because of its non-parallizable computation. Previous attention mechanisms attempted to solving the problem yet only in few cases or by combining RNNs. The author proposed Transformer architecture to deal with parallelization by relying entirely on an attention mechanism.


### Background
Previous attempts on reducing sequential computation is to use convolutional neural networks. Though Convs proved efficient, the operations to relate signals from different positions grows either linearly or logarithmically. Transformer coinstrains the number of operations to constant and counteract resolution cost by applying Multi-head Attention. Self-attention performs well on a wide range of tasks and Transformer relies on self-attention without combining with RNNs.


### Model Architecture
Key concepts: encoder-decoder structure, stacked self-attention, fully connected layers

{{< figure src="/COS597G-introduction/transformer_architecture.png" width="350" caption="The Transformer - model architecture." align="center">}}

1. Encoder
Encoder is composed of 6 identical stacked layers which contains 2 sub-layers. Residual connection are employed around each of the two sub-layers, followed by layer normalization. Sub-layers are multi-head self-attention layer and a simple, position-wise fully connected feed-forward network. Output dimension @d_{\text{model}} = 512@

2. Decoder
Decoder is composed of 6 identical stacked layers. Self-attention sub-layer is modified by applying mask, which prevent the model from attending to subsquent positions.

3. Attention
An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a *compatibility function* of the query with the corresponding key.

4. Scaled Dot-Product Attention
Two common attention functions: additive attention and dot-product attention. The difference is *compatibility function*. Additive attention use a feed-forward network with a single hidden layer while dot-product attention use *Query* and *Key*, which resembles to attention mechanism introduced in the paper. The two are similar in complexity but the latter could be optimized by matrix multiplicatoin, thus being more space-efficient in practice.

Scaled Dot-Product Attention:

$$
    \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^{T}}{\sqrt{d_k}}\right)V
$$

For larger values of @d_k@, result of @QK^T@ is going to be large, which falls into @\text{softmax}@ regions where its has extremely small gradients. Therefore, scaling is added. Notice that the scaling factor is the dimension of *Key*: @1 / \sqrt{d_k}@


#### Multi-Head Attention
Instead of performing a single attention function with @d_{\text{model}}@-dimensional keys, values and queries, it is better to linearly project the queries, keys and values @h@ times with different, learned linear projections to @d_k@, @d_k@ and @d_v@ dimensions, respectively. Computation is carried out in parallel and output values are concatenated and once again projected.

{{< figure src="/COS597G-introduction/dot-product-attention.png" width="500" caption="(left) Scaled Dot-Product Attention. (right) Multi-Head Attention consists of several attention layers running in parallel." align="center">}}

Perform linear output transformation after concatenation.

$$
    \text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_i)W^O
$$

where @\text{head}_i = \text{Attention}(QW_i^Q, KW_i^{K}, VW_i^{V})@. Dimensions (Sequence length @n@): @K \in \mathbb{R}^{n \times d_{\text{model}}}, W_{i}^K \in \mathbb{R}^{d_{\text{model}}\times d_k},@ @ W^O \in \mathbb{R}^{hd_v \times d_{\text{model}}}@. In the paper @h = 8, d_k = d_v = d_{\text{model}} / h = 64@. @d_{\text{model}}@ could be seen as dimension of embedding.


#### Applications of Attention
In deocoder, query comes from last layer while key and values comes from encoder. Each position in the encoder can attend to all positions in the previous layer of the encoder. Mask is implemented inside of scaled dot-product attention by masking out (setting to −∞) all values in the input of the softmax which correspond to illegal connections. 


#### Position-wise FFN
FFN consists of two linear transformation with ReLU activation n between.

$$
    \text{FFN} = \text{ReLU}(xW_1 +b_1)W_2 + b_2
$$

Dimensions: inner dimension 2048, output dimension 512.


#### Embeddings and Softmax
The implemetation share the same weight matrix between the two embedding layers and the pre-softmax linear transformation. In the embedding layers, we multiply those weights by @\sqrt{d_{\text{model}}}@.


#### Positional Embedding
Sine and cosine functions:

$$
\begin{aligned}
    PE_{(pos, 2i)} &= \sin(pos / 10000^{2i / d_{\text{model}}}) \\
    PE_{(pos, 2i + 1)} &= \cos(pos / 10000^{2i / d_{\text{model}}})
\end{aligned}
$$
where @pos@ is the position and @i@ is the dimension. The author also tries positional embeddings, which yields similar results. But sin/cos PE could extrapolate to sequence longer inputs.


#### Why Self-Attention
1. Total computational complexity per layer
   
|Type|Complexity per Layer|
|:--|:--:|
|Self-Attention|@O(n^2 \cdot d)@|
|Self-Attention(restricted)|@O(r \cdot n \cdot d)@|
|Recurrent|@O(n \cdot d^2)@|
|Conv|@O(k \cdot n \cdot d^2)@|

where @d@ is the representation dimension and @n@ is the length of sequence. Self-attention does not perform well when @n > d@ compared with RNNs. To improve computational performance for tasks involving very long sequences, self-attention could be restricted to considering only a neighborhood of size @r@ in the input sequence centered around the respective output position.

2. The amount of computation that can be parallelized, as measured by the minimum number of sequential operations required.
3. The path length between long-range dependencies in the network.
4. More interpretable models by investigating attention values.


### Training
Experiments on machine translation: WMT 2014 English-German/French. 8xP100 GPU, Adam optimizaer, regularization: residual dropout & label smoothing. Metrics: BLEU, Trainig cost(FLOPs)

#### Results
BLEU: EN-DE 28.4 EN-FR 41.8
[English constituency parsing](https://paperswithcode.com/task/constituency-parsing) for checking ability of task generalization.

#### Ablation
1. Number of attention heads
2. Attention key size @d_k@
3. Model size
4. Other positional embeddings


## Blog Post: The Illustrated Transformer
### Visualizing QK
In encoder,  @QK@ matrix mutiplication could be seen as quries multiplying keys, then get summation.

{{< figure src="https://jalammar.github.io/images/t/self-attention-output.png" width="400" caption="Visualization of Encoder self-attention." align="center">}}


### Encoder-Decoder Attention
Detailed version of encoder-decoder architecture:

{{< figure src="https://jalammar.github.io/images/t/transformer_resideual_layer_norm_3.png" width="600" caption="Visualization of encoder-decoder architecture" align="center">}}

What happened between encoder and decoder: encoder-decoder architecture gif:

{{< figure src="https://jalammar.github.io/images/t/transformer_decoding_1.gif" width="600" caption="Encoder-decoder attention" align="center">}}


### Position Embedding Visualization

{{< figure src="https://jalammar.github.io/images/t/attention-is-all-you-need-positional-encoding.png" width="500" caption="Position Embedding Visualization" align="center">}}
x-axis: Embedding dimension; y-axis: Token position.


### About model outputs
Because the model produces the outputs one at a time, we can assume that the model is selecting the word with the highest probability from that probability distribution and throwing away the rest. That’s one way to do it (called greedy decoding). Parameter `temperature` can affect output as well by adding a scaling factor: @\text{softmax(x / T)}@.



## References
[1] C. D. Manning, "Human Language Understanding & Reasoning," journal-article, 2022. [Online]. Available: https://www.amacad.org/sites/default/files/publication/downloads/Daedalus_Sp22_09_Manning.pdf
[2] A. Vaswani et al., "Attention Is All You Need," arXiv.org, Jun. 12, 2017. https://arxiv.org/abs/1706.03762
[3] J. Alammar, "The Illustrated Transformer." https://jalammar.github.io/illustrated-transformer/






