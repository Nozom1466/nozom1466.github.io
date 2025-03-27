---
title: 'COS597G 22 Encoder Only Models'
date: 2025-02-13T15:10:32+08:00
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

[Homepage](https://www.cs.princeton.edu/courses/archive/fall22/cos597G/)

## (ELMo) Deep contextualized word representations

### Before Reading
Authors are from [AI2](https://allenai.org/) and [UW](https://www.cs.washington.edu/). Citation 16115 (until 11/25/2024). Paper accepted by NAACL 2018, nominated as Best Paper. Paper introduced a embedding by stacking embeddings from bidirectional LSTMs.


### Motivation
ELMo aims to find better embeddings for NLP tasks. Previous methods proposed word vectors, which are encoded in static strategy and failed to deal with words with various meanings in different contexts (Training multiple representation for 1 word partly solved the problem but it is not feasible when it comes to evolving meanings). Improvements are using subword meaning and bidirectional LSTM to encode contexts around the target word. ELMo embeddings are based on biLSTM hidden representations. Previous work also claims that layers from different depth encode meanings of different levels. ELMo takes it into account in hidden representation concatenation. 



### ELMo: Embedidngs from LM
ELMo is built on biLM respresentations. BiLM gives prediction of token @t_k@ by combining forward and backward LM. Log likelihood of token @t_k@ is given by:

$$
    \begin{aligned} & \sum_{k=1}^N\left(\log p\left(t_k \mid t_1, \ldots, t_{k-1} ; \Theta_x, \vec{\Theta}_{L S T M}, \Theta_s\right)\right. \\ & \left.\quad+\log p\left(t_k \mid t_{k+1}, \ldots, t_N ; \Theta_x, \overleftarrow{\Theta}_{L S T M}, \Theta_s\right)\right)\end{aligned}
$$

where @N@ is the number of tokens, @\Theta@ denotes parameters, with subscript @x@ as token representations and @s@ as softmax layer. Note that parameters of forward and backward LM are separately maintained.

ELMo is the combination of intermediate respresentation in biLSTMs where representation set with @L@-layer biLM is given by:

$$
    \begin{aligned} R_k & =\left\{\mathbf{x}_k^{L M}, \overrightarrow{\mathbf{h}}_{k, j}^{L M}, \overleftarrow{\mathbf{h}}_{k, j}^{L M} \mid j=1, \ldots, L\right\} \\ & =\left\{\mathbf{h}_{k, j}^{L M} \mid j=0, \ldots, L\right\}\end{aligned}
$$

where @\mathbf{h}_{k, 0}^{L M}@ is the token layer and @\mathbf{h}_{k, j}^{L M} = [\overrightarrow{\mathbf{h}}_{k, j}^{L M}; \overleftarrow{\mathbf{h}}_{k, j}^{L M}]@. Basically, EMLo concatenates hidden representation of forward and backward LSTM model by layer. Architecture shown in Figure 1.


{{< figure src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*oi2vJesp7L1ElKnQ1nQQUg.png" width="600" caption="Fig. 1 ELMo architecture (illustration form BERT)" align="center">}}


Collapsed ELMo representations are used in downstream NLP tasks. The author adds scale parameters @\gamma^{\text{task}}@ and softmax-normalized weights @s^{\text{task}}@ for different layers:

$$
    \mathbf{E L M o}_k^{\text {task }}=E\left(R_k ; \Theta^{\text {task }}\right)=\gamma^{\text {task }} \sum_{j=0}^L s_j^{\text {task }} \mathbf{h}_{k, j}^{L M}.
$$

ELMo vector could either be added in inputs for enhanced representation @[x_k;\mathbf{ELMo}_k]@ or be concatenated with output @[h_k;\mathbf{ELMo}_k]@.

For computational requirements, the author cuts hidden dimensions to half (to 512) and incorporate residual connections from the first to second layer. `CNN-BIG-LSTM` trained for 10 epochs yield 39.7 on average forward and backward perplexity, with 9.7 increase compared with forward `CMM-BIG-LSTM`


### Experiments
Tasks and datasets & mectrics:
- QA: SQuAD, F1
- Textual entailment: SNLI, accuracy
- Semantic role labeling: SRL, F1 (OntoNotes)
- Conference resolution: OntoNotes coreference annotation, avg. F1
- NER: Reuters RCV1 corpus, accuracy
- SST-5: , F1

Adding ELMo representations yields SOTA results, as illustrated in Figure 2.

{{< figure src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*aQb-FD7F33Z_svGr4jTrfg.png" width="400" caption="Fig. 2 Results by adding ELMo across 6 tasks." align="center">}}

- **Where to add ELMo?**: The author add the representation in the lowest layer in this paper yet claims that some tasks may prefer adding representation in the output of the layer.

- **Differences between layers**: for tasks like Word Sense Disambiguation, last layer is better than the first layer probably because of semantic meanings in final layer. However, for tasks like POS Tagging, as structural information is needed, the first layer outperforms the last layer.

- **Efficiency in sampling**:  In the SRL case, the ELMo model with 1% of the training set has about the same F1 as the baseline model with 10% of the training set. Faster convergence by adding "offsets" to vectors in high dimension space, which helps model be optimized towards optimal points efficiently?



## Improving Language Understanding by Generative Pre-Training

### Before reading
Authors are from OpenAI (Ilya Sutskever!). Cited by 11755 (30/11/2024). 
> How time flies ... 6 years passed and OpenAI has grown into a renowned tech company with $3.4 billion annual revenue. GPT chat bot is well known by people around the globe and everyone can enjoy part of the bonus that AI continues to bring to our society. But Ilya and many other founders left OpenAI with a growing concern about LLM safety; AGI is coming yet it seems like an illusion given the poor performance of current LLM bots. Well, just embrace the changes and move forward and grow up with AI.

The paper introduced a semi-supervised approach by combining pre-training and fine-tuning. Authors also introduced a task-specific input adaption startegy for fine-tuning.


### Motivation
Training on labeled data has received successful results on NLP tasks, while using unlabeled data is challenging. The optimization objective is unclear and there is no consensus on effective way of transfer learning with learnt representations.


### Framework
There are two stages of training procedure: unsupervised pretraining and supervised fine-tuning. For fine-tuning, the paper introduces a task-agnostic approach to better adapt learnt representations to spcific tasks.

- **Unsupervised pre-training**
Classic Transformer Decoder next word prediction with multi-head attention, FFN ... 

Next word prediction objective is given by
$$
    L_1(\mathcal{U})=\sum_i \log P\left(u_i \mid u_{i-k}, \ldots, u_{i-1} ; \Theta\right)
$$

$$
    \begin{aligned}
        h_0 &= UW_e + W_p \\
        h_l &= \text{transformer_block}(h_{l - 1}), \forall \in [1, n] \\
        P(u) &= \text{softmax}(h_nW_e^{T})
    \end{aligned}
$$
where @\mathcal{U = \{u_1, \dots, u_{i - 1}\}}@ are unsupervised tokens, parameters @\Theta@, @W_e@ token embedding matrix, @W_p@ position embedding matrix. 


- **Supervised fine-tuning**
We get labeled dataset @\mathcal{C}@ in supervised fine-tuning, in which input tokens @x^{i}, i \in [1, m]@ are labeled with @y@. @y@ prediction is formulated as 
$$
    P\left(y \mid x^1, \ldots, x^m\right)=\operatorname{softmax}\left(h_l^m W_y\right).
$$

Objective is given by
$$
    L_2(\mathcal{C})=\sum_{(x, y)} \log P\left(y \mid x^1, \ldots, x^m\right).
$$

In order to improve generalization and to speed up convergence, the objective is given by
$$
    L_3(\mathcal{C}) = L_2(\mathcal{C}) + \lambda \cdot L_1(\mathcal{C}).
$$


- **Task-specific input transformations**
The paper also introduced a task-specific strategy in fine-tuning so as to aviod making extensive changes to the model architecture across tasks. Startegy is illustrated in Figure 3.

{{< figure src="https://miro.medium.com/v2/resize:fit:4800/format:webp/1*ME_kS-46o8zsF2q-Lt1caA.png" width="700" caption="Fig. 3: (left) Transformer architecture and training objectives used in this work. (right) Input transformations for fine-tuning on different tasks. We convert all structured inputs into tokensequences to be processed by our pre-trained model, followed by a linear+softmax layer." align="center">}}

Sounds like structured prompt input.


### Experiments

- **Setups**: For pre-training, the paper use BooksCorpus dataset & 1B Word Benchmark (used by ELMo), because both datasets contains long and contigious contexts. For fine-tuning, parameters are learning rate 6.25e-5, batchsize 32, linear learning rate decay with 0.2% training warm up. 


- **Results**: 4 downstream NLP tasks in fine-tuning: Natural Language Inference (recognizing textual entailment), Question answering and commonsense reasoning, Semantic Similarity, Classification. The approach achieved SOTA in 9 out of 12 datasets and works well on both small and large datasets

- **Analysis**
1. Impact of the number of transferred layers on overall performance: all layers are useful and each layer adds approx. 9% of performance increase on datasets RACE and Mutlti NLI.
2. Zero-shot performance of pretraining models on NLP tasks: performance steadily increases as over pretraining, which suggests that  generative pretraining supports the learning of a wide variety of task relevant functionality.
3. Ablation studies: 
   1. Performance without auxuliary LM objective @L_1(\mathcal{C})@:larger dataset benefit from @L_1(\mathcal{C})@ while smaller dataset not.
   2. Importance of using Transformer: the author compares Transformer with LSTM using the same framework. Transformer outperforms LLMs on most tasks.
   3. Importance of pretraining: performance drops when the model is trained directly on tasks without pertraining 




## BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

### Before Reading
Authors are from Google AI Language. Citation 120133 (30/11/2024). Paper accepted by NAACL 2019, awarded with Best Long Paper.

The paper introduces BERT, a bidirectional pretraining method using Transformer. The representations are learnt from left to right and right to left, which provides better representation.


### Motivation
There are two strategies in applying pre-trained language representations: feature-based methods (ELMo) and fine-tuning (OpenAI GPT). However, the current approached limited power of representation bacause of the nature of learning from left to right. BERT uses bidirectional training and applied two novel pretraining objectives, namely MLM and NSP, to get pretraininig representations of both token-level and sentence-level. BERT also reduce the need of task-specific archituctures.


### BERT
- **Training:** There are two steps of BERT: pre-training and fine-tuning. During pre-training, BERT utilize two objectives to get pre-training representations. Fine-tuning is firstly initialized with pre-trained parameters and all of the parameters are fine-tuned.

- **Architecture:** BERT is basically a multi-layer bidirectional Transformer encoder, which is different from GPT constrained by left-to-right nature. (In my opinion, bidirectional is mostly illustrated by attention mechanism in encoder)

- **Input/Output Representations:** In BERT, the input sequence might be a single sentence or a pack of two sentences. The first token is always [CLS] and seperation token between two sentences is [SEP]. For two sentences, the author add learned segment embedding @E_A, E_B@ to mark tokens in two sentences @A@ and @B@. The final input is the summation of token, segment embedding and position embedding, as illustrated in Figure 4.

{{< figure src="https://upload.wikimedia.org/wikipedia/commons/6/65/BERT_input_embeddings.png" width="700" caption="Fig. 4 BERT input representation. The input embeddings are the sum of the token embeddings, the segmentation embeddings and the position embeddings." align="center">}}

**Pre-training:**
1. Masked LM (MLM): Because of bi-directional feature of BERT, each word would see itself, therefore we need a new pre-training objective. Inspired by Cloze task, the author decided to randomly mask out 15% tokens in each sequence by replacing it with token [MASK]. However, since token [MASK] will not appear in fine-tuning stage after pre-training, there is a mismatch between pre-training and fine-tuning tokens. The authoer then elaborates on the detailed approach of setting [MASK]: within 15% tokens, 80% tokens are replaced by [MASK], 10% tokens are replaced by a random token and the rest stay unchanged.
2. Next Sentence Prediction (NSP): A simple binarized task of deciding whether the sentence @B@ is the next sentence of sentence @A@ in sentence pair @[A, B]@. BERT thus learned sentence-level information.

> Pre-training data: document-level literature with long contexts, such as BooksCorpus and English Wikipedia in order ot extract contiguous sequences.


**Fine-tuning:** Plug in the taskspecific inputs and outputs into BERT and finetune all the parameters end-to-end. Sentence @[A, B]@ could be interpreted as different meanings like QA,  hypothesis-premise pairs etc..



### Experiments
Tested on four different tasks: 
1. GLUE: last hidden state + classification weights + softmax
2. SQuAD v1.1: QA pairs, predict on the answer span index @[i, j]@
3. SQuAD v2.0: The answer probably does not exists in contexts. No answer -> span from [CLS] to [CLS]. The rule of SQuAD also applies.
4. SWAG: Given a sentence, the task is to choose the most plausible continuation among four choices.

- **Ablations**
1. Effect of Pre-training tasks: experiments on No NSP, LTR(left2right)&No NSP
   1. removing NSP hurts performance significantly on QNLI, MNLI, and SQuAD 1.1
   2. The LTR model performs worse than the MLM model on all tasks, with large drops on MRPC and SQuAD.
2. Effect of Model Size: 
   1. scaling to extreme model sizes also leads to large improvements on very small scale tasks, provided that the model has been sufficiently pre-trained. 
   2. We hypothesize that when the model is fine-tuned directly on the downstream tasks and *uses only a very small number of randomly initialized additional parameters*, the task-specific models can benefit from the larger, more expressive pre-trained representations even when downstream task data is very small.
3. Ablation w/o fine-tuning 
   1. Pre-computed representations lower down the costs
   2. BERT is effective for both finetuning and feature-based approaches (only pre-training).







## RoBERTa: A Robustly Optimized BERT Pretraining Approach

### Before Reading
17176 citation up to 01/09/2025. It's a follow-up work of BERT, in which the author introduced better settings for BERT model training.


### Motivation
The RoBERTa proposed an improved receipe for training BERT models. Main settings are training time, size of batches, elimination of NSP training, longer sequence and dynamic masking patterns. The results showed improved performance on metrics in BERT.


### Training Settings of BERT
BERT is optimized with Adam with @\beta_1 = 0.9, \beta_2 = 0.999, \epsilon=1e-6@ and @L_2@ weight decay of @0.01@ (warm up 10000 steps to 1e-4 and linearly decayed). Dropout rate 0.1 on all layers. GELU activation. Models trained for 1000000 updates with 256 as batchsize and max-length 512. Models are trained with mixed precision floating point, 8xV100.

Training data includes BOOKCORPUS, CC-NEWS, OPENWEBTEXT and STORIES.

### Training Analysis
**Dynamic masking and static masking**: To avoid using the same mask for each epoch, the training data were duplicated 10 times and were masked with different ways for each epoch. This was introduced in BERT and called static masking. While for dynamic masking, masking patterns are generated every time we feed a sequence to the model. And ... as the results presented, we indeed see the increase though being marginal.


**Next Sentence Prediction**: NSP loss was questioned by replication experiments. The authors found:
1. Using individual sentences hurts performance on downstream tasks, which we hypothesize is because the model is not able to learn long-range dependencies.
2. Removing the NSP loss matches or slightly improves downstream task performance.
3. Restricting sequences to come from a single document performs slightly better than packing sequences from multiple documents


**Training with large batches**: Training with large batches improves perplexity for the masked language modeling objective, as well as end-task accuracy. Large batches are also easier to parallelize via distributed data parallel training.


- **Text Encoding**: Train BERT model using Byte-Pair Encoding.


### RoBERTa Training settings
Three points: Dynamic masking, trained with FULL-SENTENCES dataset without NSP loss, large mini-batches and byte-level BPE. More settings revolves around data used for pretraining and number of passes through the data.

The author further conbimed three datasets for training (160GB) and trained the model from 100K to 500K steps.


### Evaluations
Models are evaluated on GLUE, SQuAD and RACE.

- **GLUE**
There are 2 types of tasks: single-task and emsembled task in GLUE. RoBERTa was finetuned for single task on each training dataset based on pretrained model. And for ensembled task, RoBERTa did not depend on multi-task finetuning. Instead, for RTE, STS and MRPC, the model was fine-tuned on MNLI single-task model.

- **SQuAD**
RoBERTa finetuned only on SQuAD training data without data augmentation like previous works. The single RoBERTa model outperforms all but one of the single model submissions, and is the top scoring system among those that do not rely on data augmentation.


- **RACE**
Each candidate answer was concatenated with the corresponding question and passage. The total length is at most 512 tokens.




## ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators
### Before Reading
Accepted by ICLR 2020. Authors are from Stanford and Google Brain (Manning!). Citation 4424 up to 01/23/2025.

Major improvements on pretraining with MASK (Masked language modeling task, MLM) in BERT. Instead of training with fixed [MASK] token, ELECTRA predicts whether the word is replaced by generator or not, which means all tokens in the input will be considered as prediction objectives.

### Motivation
In MLM task, only 15% of the tokens are learnt by the model as the number of masked tokens are limited. In this work, the author proposed a pretraining task *replaced token detection*, in which the model learns to distinguish real input tokens from generated replacements. The side-product of this setting is that it solves mismatch between training and testing in BERT. (Remember 15%-80%-10%-10%?) Note that the model is seen as a generator that predicts the original identity of corrupted tokens. Moreover, the ELECTRA also features with compute-efficiency and parameter-efficiency in pretraining stage.


### Method
There are two NNs in this work, namely *Generator* and *Discriminator*. The generator is in charge of putting mask on input sentence and generating corrupted sentence by replacing masks with other words. The discriminator then tries to distinguish which word in the corrupted sentence is replaced by Generator. In Generator mask words @m_i@, which follows uniform distribution:
$$
    m_i \sim \operatorname{unif}\{1, n\} \text{ for } i=1 \text{ to } k \quad \mathbf{x}^{\text {masked }}=\operatorname{REPLACE}(\mathbf{x}, \mathbf{m},[ \text{MASK} ])
$$
where @\text{REPLACE(\mathbf{x}, \mathbf{m}, p)}@ means replace masked elements in @\mathbf{x}@ with p using @\mathbf{m}@ as mask. @h@ are hidden representations and @e@ are embeddings of generator encoder. Then the masked elements @m_i@ are replaced with new words @\hat x_i@, which follows the distribution given by softmax normalization:
$$
\begin{aligned}
    \hat{x}_i &\sim p_G\left(x_i \mid \mathbf{x}^{\text { masked }}\right)\text{ for } i \in \mathbf{m} \\
    p_G\left(x_t \mid \mathbf{x}\right)&=\exp \left(e\left(x_t\right)^T h_G(\mathbf{x})_t\right) / \sum_{x^{\prime}} \exp \left(e\left(x^{\prime}\right)^T h_G(\mathbf{x})_t\right) \\
    \mathbf{x}^{\text {corrupt }}&=\operatorname{REPLACE}(\mathbf{x}, \mathbf{m}, \hat{\mathbf{x}}) 
\end{aligned}
$$

The corrupted input @\mathbf{x}^{\text{corrupt}}@ is the input of Discriminator, which tries to distinguish the word replaced by Generator. The possibility for each word is given by:

$$
    D(\mathbf{x}^{\text {corrupt}}, t) = \text{sigmoid}(w^{T}h_{D}(\mathbf{x}^{\text {corrupt }})_t)
$$
The loss function is the combination of MLM task and discrimination task. Figure 4 illustrates ELECTRA using an example.

{{< figure src="https://ar5iv.labs.arxiv.org/html/2207.08141/assets/f222222.png" width="500" caption="Fig 4. An overview of replaced token detection. The generator can be any model that producesan output distribution over tokens, but we usually use a small masked language model that is trainedjointly with the discriminator. Although the models are structured like in a GAN, we train thegenerator with maximum likelihood rather than adversarially due to the difficulty of applying GANsto text. After pre-training, we throw out the generator and only fine-tune the discriminator (the ELECTRA model) on downstream tasks." align="center">}}

$$
\begin{aligned}
    &\min _{\theta_G, \theta_D} \sum_{\mathbf{x} \in \mathcal{X}} \mathcal{L}_{\mathrm{MLM}}\left(\mathbf{x}, \theta_G\right)+\lambda \mathcal{L}_{\text {Disc }}\left(\mathbf{x}, \theta_D\right) \\
    & \mathcal{L}_{\mathrm{MLM}}\left(\mathbf{x}, \theta_G\right)=\mathbb{E}\left(\sum_{i \in \mathbf{m}}-\log p_G\left(x_i \mid \mathbf{x}^{\text {masked }}\right)\right) \\ 
    & \mathcal{L}_{\mathrm{Disc}}\left(\mathbf{x}, \theta_D\right)=\mathbb{E}\left(\sum_{t=1}^n-\mathbb{1}\left(x_t^{\mathrm{corrupt}}=x_t\right) \log D\left(\mathbf{x}^{\text {corrupt }}, t\right)-\mathbb{1}\left(x_t^{\text {corrupt }}f \neq x_t\right) \log \left(1-D\left(\mathbf{x}^{\text {corrupt }}, t\right)\right)\right)
\end{aligned}
$$
> Disc loss: cross-entropy loss for discrimination.

- **Difference with GAN** 
1. If the generated token happens to ben correct, the token will be considered "real" instead of "fake".
2. The generator is trained with maximum likelihood rather than being trained adversarially to fool the
discriminator. (Ad training is challenging because of it is impossible to backpropergate through sampling from generator.)


### Experiments
Datasets are GLUE, SQuAD. EM and F1 scores.

- **Model extension**: Some techiques used in initialization and training.
1. Weight sharing: share all/parts of parameters between generator and discriminator
2. Smaller generators: large models generates challenging tasks for discriminator, and sometimes being too hard to answer by discriminator. Smaller generator works effectively. (small model here: keep some of params in generator constant without updating in BP)
3. Training algorithms: Two stage procedure: training MLM task for n steps; initialize params in discriminator using params in trained generator. Train the discriminator with generator frozen. 

The author also discusses about the small and large ELECTRA models using weaker training hyperparameters. Further discussions are about the efficiency of ELECTRA. Thr author designed three variations to test token learning in ELECTRA. Results suggest that a large amount of ELECTRA’s improvement can be attributed to learning from all tokens and a smaller amount can be attributed to alleviating the pre-train fine-tune mismatch. (btw proves 10% random replacement in BERT is insufficient to solve the issue).




### References
[1] Tsang, S. (2022, January 8). Review — ELMO: Deep Contextualized Word Representations. Medium. https://sh-tsang.medium.com/review-elmo-deep-contextualized-word-representations-8eb1e58cd25c
[2] Peters, M. E., Neumann, M., Iyyer, M., Gardner, M., Clark, C., Lee, K., & Zettlemoyer, L. (2018, February 15). Deep contextualized word representations. arXiv.org. https://arxiv.org/abs/1802.05365
[3] Radford, A. (2018). Improving language understanding by generative pre-training.
[4] Devlin, J., Chang, M., Lee, K., & Toutanova, K. (2018, October 11). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv.org. https://arxiv.org/abs/1810.04805
[5] Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L., & Stoyanov, V. (2019, July 26). RoBERTa: A Robustly Optimized BERT Pretraining Approach. arXiv.org. https://arxiv.org/abs/1907.11692
[6] Clark, K., Luong, M., Le, Q., V., & Manning, C. D. (2020, March 23). ELECTRA: Pre-training text encoders as discriminators rather than generators. arXiv.org. https:// arxiv.org/abs/2003.10555