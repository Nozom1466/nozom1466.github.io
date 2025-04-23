---
title: 'COS597G 22 Encoder Decoder Models'
date: 2025-04-22T23:45:33+08:00
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
tags: []
categories:
---

[Homepage](https://www.cs.princeton.edu/courses/archive/fall22/cos597G/)


For BART, T5, mT5 and AlexaTM 20B


## BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension

Encoder-decoder model was not actually a popular architecture between 2019 and 2020. As you can observe from Figure 1, during 2019-2020 slot, lots of tech companies bid on Encoder-only models, including BERT, RoBERTa, ALBERT. Encoder-decoder and decoder-only models are not yet well explored. But if we further look at the end of branches, Encoder-decoder models still take a place in model zoo, such as Flan series. 

{{< figure src="https://media.licdn.com/dms/image/v2/D4D22AQFNeG_WIY7WOQ/feedshare-shrink_1280/feedshare-shrink_1280/0/1683649150911?e=1746057600&v=beta&t=J7YdPUI60vDiuhDTK7UtMVQrfHrLeVddhkINUVwM3gs" width="800" caption="Figure 1. LLMs evolutionary tree." align="center">}}


### Contributions
1. **Combination of bidirectional encider and autoregressive decoder**: BERT(bi-directional) + GPT(uni-direct auto-regressive)
2. **Pretraining with better noising approaches**: shuffling + in-filling scheme


### Approach
- **Architecture**: BART use seq2seq Transformer architecture. Mofidications: 
1. ReLUs are modified as GeLUs, with initialization from @\mathcal{N}(0, 0.02)@.
2. 6 layers in encoder and 12 layers in decoder (following BERT)
3. Remove Feed-Forward network before word prediction
- **Pretraining**: Trained by corrupting documents and then optimizing a reconstriction loss. Corruptions are introduced in Figure 2.

{{< figure src="https://myblog-1316371247.cos.ap-shanghai.myqcloud.com/myblog/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-04-23%20013045.png" width="800" caption="Figure 2: Transformations for noising the input that we experiment with. These transformations can be composed." align="center">}}

- **Finetuning**: Representations produced by BART are used in Seq/token classification, seq generationa and MT tasks. In MT, the author replace BART's encoder embedding layer with a new randomly initialized encoder, as illustarted in Figure 3.

{{< figure src="https://myblog-1316371247.cos.ap-shanghai.myqcloud.com/myblog/20250423013642552.png" width="1000" caption="Figure 3. Fine tuning BART for classification and translation." align="center">}}


### Pretraining Objective Comparison
- **Models**: LM(GPT), Permuted LM(XLNet), Masked LM(BERT), Multitask Masked LM(UniLM), Masked seq2seq(MASS)
- **Tasks**: 
1. SQuAD(context + question -> context span)
2. MNLI(con + q -> relation)
3. ELI5, XSum, CNN/DM(con + q -> abstraction)
4. ConvAI2(dialogue gen)
- **Insights**:
1. Performance of pre-training methods varies significantly across tasks
2. Token masking is crucial
3. Left-to-right pre-training improves generation
4. Bidirectional encoders are crucial for SQuAD
5. The pre-training objective is not the only important factor
6. Pure language models perform best on ELI5
7. BART achieves the most consistently strong performance.


### Large-scale Pre-training Experiments
Pretraining with large batchsize (8000 a batch. 500k steps following RoBERTa): 
- **Discriminative Tasks**: BART’s improvements on generation tasks do not come at the expense of classification performance.
- **Generation Tasks**: Summarization, Dialogue and Abstrctive QA.


## Exploring the Limits of Transfer Learning with a Unified  Text-to-Text Transformer

This work explores the landscape of transfer learning techniques for NLP by introducing a unified framework that converts all text-based language problems into a text-to-text format. And the model proposed in this paper, known as T5, is trained on unified text2text framework, where all of outputs are regarded as texts. ("3.8", "5")


### Motivation
Previously the training in ML is amenable to downstream learning tasks and the knowledge required for the learning is learned as part of auxiliary task.(word vectors). Later the scheme shifted to pretraining on data-rich tasks using unlabeled data (like Common Crawl project). The work focus on the understanding of burgeoning transfer learning techniques. 

Basic idea: treat every text processing problem as a “text-to-text” problem, i.e. taking text as input and producing new text as output.



### Architecture of T5 Model
To summarize, T5 is roughly equivalent to the original Transformer with the exception of 
1. **removing the Layer Norm bias**：activations are only rescaled and no additive bias is applied.
2. **placing the layer normalization outside the residual path**
3. **using a different position embedding scheme.**: We use a simplified form of position embeddings where each “embedding” is simply a scalar that is added to the corresponding logit used for computing the attention weights. For efficiency, we also share the position embedding parameters across all layers in our model.

> Check out [Transformers without Normalization](https://arxiv.org/abs/2503.10622) from meta.... Layer Norm is replaced by @tanh(\cdot)@，


### Dataset - The Colossal Clean Crawled Corpus
Adopted from Common Crawl dataset, using cleaning up techniques:
- end in .
- page > 3 sentences; lines >= 5 words
- remove bad words
- lorem ipsum placeholder removed
- curly bracket removed
- citation markers removed
- policy & cookies removed
- leave only one in 3-span sentence sliding window
- pages not written in English


### Downstream Tasks

> Indeed an in-depth tech report ...

750 GB dataset
- machine translation: WMT
- question answering: SQuAD
- abstractive summarization: CNN/Daily Mail
- text classification: GLUE, SuperGLUE


### Baselines
- **Goal**: Pre-train a standard Transformer using a simple denoising objective and then separately fine-tune on each of our downstream tasks.
- **Model**: encoder and decoder are each similar in size and configuration to a BERT_BASE. Each has 12 blocks, FF layer @d_{ff}=3072@, @d_{kv} = 64@, multi-attention with 12 heads. Dropout prob 0.1.
- **Training**: As all tasks are regarded as t2t, loss f: [teacher forcing](https://blog.csdn.net/qq_30219017/article/details/89090690) and cross-entropy loss. AdaFactor as optimizor. Learning rate schedule: inverse square root (@1 / \sqrt{\max{(n, k)}}@), with n as iteration index and k as number of warm-up steps.
- **Vocabulary**: SentencePiece to encode text as WordPiece tokens.
- **Unsupervised Objective**: (Denoising) An objective that randomly samples and then drops out 15% of tokens in the input sequence, as shown in Figure 4.

{{< figure src="https://miro.medium.com/v2/resize:fit:988/format:webp/1*9yFICqDlfprn-I_VZ5RHgw.png" width="800" caption="Figure 4.Schematic of the objective we use in our baseline model. In this example, we process the sentence “Thank you for inviting me to your party last week.” The words “for”, “inviting” and “last” (marked with an ×) are randomly chosen for corruption. Each consecutive span of corrupted tokens is replaced by a sentinel token (shown as <X> and <Y>) that is unique over the example. Since “for” and “inviting” occur consecutively, they are replaced by a single sentinel <X>. The output sequence then consists of the dropped-out spans, delimited by the sentinel tokens used to replace them in the input plus a final sentinel token <Z>." align="center">}}


> The following sections are discussing model performance from Architectures, Unsupervised Objectives, Pre-training Datasets, Training strategy and Scaling.


### Architectures
Another classification other than encoder/decoder: look into attention mask adopted by the model. There are 3 types of mask patterns: Fulli-visible, Casual and Casual with prefix, as shown in Figure 5.

{{< figure src="https://blog.codescv.com/images/t5-masks.png" width="800" caption="Figure 5. Matrices representing different attention mask patterns. The input and output of the self-attention mechanism are denoted x and y respectively. A dark cell at row i and column j indicates that the self-attention mechanism is allowed to attend to input element j at output timestep i. A light cell indicates that the self-attention mechanism is not allowed to attend to the corresponding i and j combination. Left: A fully-visible mask allows the self-attention mechanism to attend to the full input at every output timestep. Middle: A causal mask prevents the ith output element from depending on any input elements from “the future”. Right: Causal masking with a prefix allows the self-attention mechanism to use fully-visible masking on a portion of the input sequence." align="center">}}


- **Standard Encoder-Decoder architecture**: All-visible mask(Encoder) + Casual mask(Decoder)
- **Language Model**: Casual mask, Language models are typically used for compression or sequence generation
- **Prefix LM**: Casual with Prefix mask, could be considered as encoder+decoder combined. Architectures as shown in Figure 6. 
Prefix LMs resembles BERT for classification tasks when you feeding the model with tasks like: `I hate pigeons. hypothesis: My feelings towards pigeons are filled with animosity. target:` (Of course! Didn't see why the author are adding this paragraph ...). Attention masking seems to be an interesting topic, which involves many other memory-efficient or computing-efficient methods (paged attention for vllm?).


{{< figure src="https://dkharazi.github.io/88459ae93dd4af11a69cab297fec5dbd/t5training.png" width="800" caption="Figure 6. Schematics of the Transformer architecture variants we consider. In this diagram, blocks represent elements of a sequence and lines represent attention visibility. Different colored groups of blocks indicate different Transformer layer stacks. Dark grey lines correspond to fully-visible masking and light grey lines correspond to causal masking. We use “.” to denote a special end-of-sequence token that represents the end of a prediction. The input and output sequences are represented as x and y respectively. Left: A standard encoder-decoder architecture uses fullyvisible masking in the encoder and the encoder-decoder attention, with causal masking in the decoder. Middle: A language model consists of a single Transformer layer stack and is fed the concatenation of the input and target, using a causal mask throughout. Right: Adding a prefix to a language model corresponds to allowing fully-visible masking over the input." align="center">}}

Here the authors mentioned criterions in selecting models, condidering these models are in different architectures and parameters. We suppose two models are equivalent if they have the same parameter @P@ or the same computational cost @C@. Consider an *encoder-decoder model* with @L + L@ layers, @P + P@ parameters and a *language model*(decoder) with @2L@ layers and @2P@ parameters. The parameters are approximately the same for these models but the **computation cost** of *language model* is approx. twice of that in *encoder-decoder* model. Because the latter has to deal with both input squence and output sequence but the former deal with inputs and outputs separately. (has lots to do with sequence length ...). Theresfore, they select:
- e-d, L + L -> 2P, M flops
- e-d,  shared params -> P, M flops
- e-d, L/2 + L/2 -> P, M/2 flops
- d, L -> P, M flops
- d, prefix -> P, M flops
where e-d for encoder and decoder and L for layers, P for parameters, M for computational cost. 


- **Results (for different architecture)**: Denoising task (metioned in previous section) and language modeling task (predicting the whole sentence for language model and predicting the second half of the sentence given the first half). Sharing the params across e-d performed very well and halfing params hurts the performace. 




### Unsupervised Objectives
Examples of common unsupervised objectives (Figure 7). Models are fisrt pretrained based on these unsupervised objectives then evaluated on downstream tasks (GLUE, CNNDM, SQuAD, SGLUE, EnDe, EnFr and EnRo). This section extends in: 3 common unsupervised objectives -> variants of BERT objective (MLM) -> exploration of corruption rate -> exploration of corrupting spans.


{{< figure src="https://myblog-1316371247.cos.ap-shanghai.myqcloud.com/myblog/20250423235343386.png" width="1000" caption="Figure 7. Examples of inputs and targets produced by some of the unsupervised objectives we consider applied to the input text “Thank you for inviting me to your party last week .” Note that all of our objectives process tokenized text. For this particular sentence, all words were mapped to a single token by our vocabulary. We write (original text) as a target to denote that the model is tasked with reconstructing the entire input text. <M> denotes a shared mask token and <X>, <Y>, and <Z> denote sentinel tokens that are assigned unique token IDs. The BERT-style objective (second row) includes a corruption where some tokens are replaced by a random token ID; we show this via the greyed-out word apple." align="center">}}


- **High Level Approaches**: Tha author evaluated 3 types of objectives: Prefix language modeling, BERT-syle(MLM) and deshuffuling (as illustarted in Figure 7). BERT objective stands out (signifigantly over Deshuffling).
- **Variants of BERT Objective**: Purpose: better performance and better efficiency. 
    - *VARIANT 1: MASS-style*, reconstruct the original uncorrupted sequence(4th in Figure 7); 
    - *VARIANT 2: Unique mask token*, predict token prefixed by special token(5th in Figure 7).; 
    - *VARIANT 3: Drop Corrupted Tokens*, concatenate predicted tokens(6th in Figure 7). All these variants performs similarly. 
    - Notice that performace of dropping corrupted tokens fluctuates on several metrics. Dropping is still attractive because it *reduces the input length thus making training faster*.
- **Corruption Rate**: Limited effect on performance. Larger corruption rate -> more inference time.
- **Corrupting Spans**: BERT mask follows i.i.d., masking tokens independently. While in some cases we need consecutive corruption. Number of corruption span and span lengths are determined by parameters. Again, this trick has limited effect on downstream task performance. While span corruption slightly speeds up training because it produces shorter sequence on average.

- **Conclusions**: 
1. Denoising objectives(BERT MLM) outperforms language modeling and deshuffling.
2. Choosing among the denoising objectives we considered here should mainly be done **according to their computational cost**(since similar approaches yields slight improvements). It may be fortuitous to explore entirely different ways of leveraging unlabeled data.


> For conclusion 2, the paper seems only to explore BERT variants. What about the other two?


### Pretraining Data set