# Pretrained Language Representations for Text Understanding: A Weakly-Supervised Perspective

<https://yumeng5.github.io/kdd23-tutorial/>

# Session 1 - Pretrained Language Models (Yu Meng)

- Outline
    - Categorization by Architecture
- Categorization by Architecture
    - if GPT and LLaMA are decoder only models - why are they still good at NLU and classification tasks (which encoder-only or encoder/decoder models are supposedly good at)
- GPT-Style Pretraining: Text Generation
    - A lot of downstream tasks like classification can be converted to generation tasks (this answers my above question)
- Why LLMs?
    - scaling up language models induces emergent abilities
    - "Emergent": not present in smaller models but in larger models
- Are decoder-only models faster to train?
    - Is that why they are more popular? 
    - Are emergent properties also appearing in encoder-only or encoder-decoder models?
        - Not clear
    - Decoder-only models are good at text generation where these emergent properties are coming out
    - Encoder-only cannot generate next token
    - PaLM is encoder-decoder
- Encoder-Only Models (Bidreictional) like BERT
    - MLM - masked language modeling
    - Next sentence prediction
    - Variants of BERT
- Encoder-Decoder models (T5, BART)
    - T5:
        - Pretraining: Mask out spans of texts; generate the original spans
    - BART [slide]

## Training and Deployment

- Deployment of Pretrained Language Models
    - Fine-tuning
        - linear layer for standard fine-tuning
        - MLM layer for prompt-based fine-tuning
    - Standard Fine-Tuning (of encoder models)
        - add task specific layers on top of embeddings
- Prompt-based fine-tuning (decoder and encoder)
    - not adding any new layers
    - Task descriptions are created to convert training examples to cloze questions
    - Prompt-Based Zero-Shot Inference
        - knowledge can be extracted from PLMs through cloze patterns
        - can serve as knowledge bases but retrieved answers are not guaranteed to be accurate
    - In-Context Learning: Few-Shot Inference
    - Instruction-tuning
- Parameter-Efficient Tuning of PLMs
    - Strategies:
        - Adapter
        - Prefix Tuning
        - Low-Rank Adaption
    - Prefix Tuning
    - Low-Rank Adaption (LoRA)
        - inject trainable low-rank matrices into transformer layers to approximate the weight updates
        - can be used with quantization techniques (QLoRA)

## Extending Language Models for Text-Rich Networks

- Homogenous & Hetereogenous Text-Rich Networks
    - Homogenous: nodes/edges are single-typed
    - Hetereo: multi-typed
- Edgeformers: Learning on Homogenous Networks
    - See paper
    - Learning node and edge representations with virtual node tokens
- Heterformer: Learning on Heterogenous Networks
    - Only built on BERT, haven't tried decoder-only models like LLaMA
    - Use virtual neighbor tokens inside each Transformer layer for text encoding
    - Fuse representations of each node's text-rich neighbors, textless neighbors, and its own context via attention
    - Validation
        - link prediction
- Pretraining on text-rich networks
    - Text understanding could depend on network structures!
        - "Hershey's should have some similarity with the chocolate from "Ferrero" based on network structures
    - Patton
        - Two pretraining objectives:
            - Network-contextualized masked language modeling (NMLM)
            - Masked node prediction (MNP)
        - Performance studies

# Session 2 - Text Representation Enhanced Topic Discovery (Jiaxin Huang)
- Outline
    - Traditional Topic Models
    - Embedding-based Discriminative Topic Mining
    - Topic Discovery with PLMs
- Topic Modeling: Introduction
    - word distribution
    - topic distribution
    - LDA
        - each document is represented by a mixture of words
        - How to learn latent variables?
        - Issues: LDA is completely supervised, cannot take user supervision. What if a user is interested in a topic that is not in the corpus?
    - Supervised LDA
        - Allow users to provide document annotations/labels
    - Seeded LDA
        - several seed words for each topic
- Discriminative Topic Modeling
    - Speaker proposes this method: Given a text corpus and a set of category names, retrieve a set of terms that exclusively belong to each category
    - Difference from topic modeling
        - requires a set of user provided category names
- Discriminative Topic Mining via CaTE: Category Name-guided Embedding (2020)
    - Text Generation Modeling
        1. Topic Assignment
        2. Global Context
    - Introduce a scalar value: specificity - how specific a word is
    - Case study: Effect of distributional specificity 
        - cool showing the coarse-to-fine specificity on NYT dataset [slide]
- Hierarchical Topic Modeling 
    - JoSH Text Embedding - new type of embedding
        - <https://github.com/yumeng5/JoSH>
        - hyperbolic methods - are these present in word2vec and glove?
- Topic Discovery with PLMs (Pre-trained Language Models)
    - Challenges [slide]
    - TopClus: Paper 2022 
    - SeedTopicMine: Paper 2023
    - EvMine: Paper 2022
- Event-detection
    - Real-world events are naturally organized in a hierarchical structure
    - Big events vs. small events
    - Key-event detection given a news corpus on a theme
        - Key-event: non-overlapping document clusters that not necessarily exhaust the corpus
    - We introduce the idea of temporal term frequency - inverse time frequency [slide]
    - Graph based method to combine textual and temporal information
    - Examples
        - 2019 Hong Kong Protest
        - Ebola outbreak
- It'd be interesting to apply these methods TopClus, SeedTopicMine and EvMine to new datasets like
    - Lex Friedman transcripts
    - Sermons
    - Chat transcripts
    - Company reviews (Glassdoor)
    - Company survey responses (free text)

# Session 3 - Part III: Weakly-Supervised Text Classification 
<https://yumeng5.github.io/files/kdd23-tutorial/Part3.pdf>

- Single-label vs. Multi-label
    - Papers: probably multi-label if it's categornization
- Flat vs. Hierarchical 
    - Flat: All labels are at the same granularity (reviews)
    - Hierarchical: Representing their parent-child relationship
        - paper topic classification, e.g. `cs.CL` is a child of `cs` for arXiv papers
- NLU benchmarks
    - 6 out of 7 NLU tasks in the GLUE benchmark can be cast as a text classification problem
- Weakly-supervised text classification
    - Motivation: 
        - Supervised text classification models (especially recent deep neural models) rely on a significant number of manually labeled training documents to achieve good performance.
        - Collecting such training data is usually expensive and time-consuming. In some domains (e.g., scientific papers), annotations must be acquired from domain experts, which incurs additional cost.
    - Text classification without massive human-annotated training data
        - Keyword-level weak supervision: category names or a few relevant keywords
        - Document-level weak supervision: a small set of labeled docs
    - General Ideas (3)
        - Joint representation learning: Put words, labels, and documents into the same latent space using embedding learning or pre-trained language models
        - Pseudo training data generation
            - Retrieve some unlabeled documents or synthesize some artificial documents using text embeddings or contextualized representations
            - Give them pseudo labels to train a text classifier
        - Transfer the knowledge of pre-trained language models to classification tasks
- Example - WeSTClass
    - Embed all words (including label names and keywords) into the same space 
    - Pseudo document generation: generate pseudo documents from seeds
        - are these coherent documents?
    - Self-training: train deep neural nets (CNN, RNN) with bootstrapping   
- ConWea[ACL’20], LOTClass[EMNLP’20], X-Class[NAACL’21], PromptClass [arXiv’23]
- What abilities to LM have?
    - word disambiguation 
    - next token prediction
    - directly use BERT to encode the whole document
- ConWea 
    - User-provided seed words may be ambiguous
- LOTClass
    - find topic words based on label names: overcome the low semantic coverage of label names
    - use language models to predict what words can replace label names
    - interchangeable words are likely to have similar meanings
    - [slide] example using sports
        - sports is not at a topic-indicative position
    - Context-free matching of topic words is inaccurate 
        - “Sports” does not always imply the topic "sports"
    - Achieve around 90% accuracy on four benchmark datasets by only using at most 3 words (1 in most cases) per class as the label name
- How Powerful Are Vanilla BERT Representations in Category Prediction?
    - An average of BERT representations of all tokens in a sentence/document preserves domain information well [1].
- X-Class: Class-Oriented BERT Representations
- PromptClass: Prompt-based Fine-tuning for Text Classification


## Weakly-supervised structure-enhanced text classification
 - Taxonomy-enhanced:TaxoClass[NAACL’21]
 - Metadata-enhanced:MICoL[WWW’22],MAPLE[WWW’23]
 - Taxonomies for multi-label text classification are often big. 
    - Amazon Product Catalog: x10^4 categories
    - MeSH Taxonomy (for medical papers): x10^4 categories 
    - Microsoft Academic Taxonomy: x10^5 labels
    - Relevance model: BERT/RoBERTa fine-tuned on the NLI task
        - NLI task is to determine whether a hypothesis is entailed by a premise
        - **Treat the document as the premise, can you infer the hypothesis?**
    - How to use the taxonomy?
        - shrink the label search space with top-down exploration
    - Doesn't this take a long time? 
        On average how many forward passes are you doing through BERT?
    - Do you have any methods to learn a taxonomy in an automated session?
- Metadata-enhanced:MICoL[WWW’22],MAPLE[WWW’23]
    - Metadata is prevalent in many text sources
    - Contrastive learning [1]: Instead of training the model to know “what is what” (e.g., relevant (document, label) pairs), train it to know “what is similar with what” (e.g., similar (document, document) pairs)
    - Using metadata to define similar (document, document) pairs.
        - ideas of meta-path and meta-graph
- MAPLE: Constructing a Cross-Field Benchmark
    - 19 scientific fields
    - 11.9 million papers
- MAPLE: A Cross-Field Cross-Model Study
    - In the 19 fields, using the 3 classifiers, we empirically study if adding metadata (i.e., venues, authors, and references) can be helpful.
    - The effect of metadata tends to be similar in two fields that belong to the same high-level scientific area [1]. For example, Biology and Medicine are both life sciences, and the effects of venues, authors, and references are largely aligned in the two fields.

## Weakly-supervised NLU

- Zero-shot: SuperGen[NeurIPS’22], ZeroGen[EMNLP’22]
- Few-shot: FewGen [ICML’23]
- Combine a generator to generate training data and pass that training data with a classifier
- FewGen: Augmentation-Enhanced Few-Shot Learning
    - Tune a generative PLM (GPT-like) on the small few-shot training set using prefix-tuning 
    - Use the tuned PLM to create novel training data
    - Fine-tune a classification PLM on both the few-shot and synthetic training sets

# Session 4 - Language Models for Knowledge Base Construction 

- Span Detection
    - Phrase Mining
    - Statistics-based models (TopMine, SegPhrase, AutoPhrase)
    - UCPhrase: Unsupervised Context-aware Quality Phrase Tagging [KDD’21]
- UCPhrase
    - Unsupervised 
    - AttentionMap
        - Extract knowledge directly from a pre-trained language model
        - the attention map of a sentence vividly visualizes its inner structure
    - Phrase tagging as Image Classification
    - Quantitative Evaluation
- Phrase-aware Unsupervised Constituency Parsing [ACL’2022]

## Entity Typing

- Few-shot Entity Typing - Automatic Label Interpretation and Generating New Instance for Entity typing [KDD’22]
- Zero-shot Entity Typing - ONTOTYPE: Ontology-Guided Annotation-Free Fine-Grained Entity Typing [Not accepted by ACL`2022]

## OntoType: Ontology-Guided Entity Typing
- Zero-shot entity typing: Assigns fine-grained semantic types to entities without any annotations
    - Ex. Sammy Sosa [Person/Player] got a standing ovation at Wrigley Field [Location/Building/Stadium] 
- Challenges of weak supervision based on masked language model (MLM) prompting
    - A prompt generates a set of tokens, some likely vague or inaccurate, leading to erroneous typing
    - Not incorporate the rich structural information in a given, fine-grained type ontology 
- OntoType: Ontology-guided, Annotation-Free, Fine-Grained Entity Typing
    - Ensemble multiple MLM prompting results to generate a set of type candidates
    - Progressively refine type resolution, from coarse to fine, following the type ontology,
under the local context with a natural language inference model
- OntoType: Outperforms the SoTA zero-shot fine-grained entity typing methods
- TODO: We could use this to build a skills parser

### Steps

1. Candidate Type Generation
    - if all the steps down the taxonomy you get a clear vote, then you are more confident in the prediction
2. High-level Type Alignment
3. Fine-grained type resolution
    - Progressively refine type resolution, from coarse to fine, following the type ontology
    - Type ontology used at every step

### Candidate Type Generation

- Four Hearst Patterns

Main point: Pretrained LLM + Taxonomy/Ontology - you can probably beat the supervised benchmarks 

## Relation and Event Extraction

- Open-Vocabulary Argument Role Prediction
    - Free text --> Pass in categories to extract --> Downstream task argument extraction 
    - Yizhu Jiao, Sha Li, Yiqing Xie, Ming Zhong, Heng Ji and Jiawei Han “Open-Vocabulary Argument Role Prediction for Event Extraction”, EMNLP’22
    - Framework for RolePred - (Argument Role Prediction)

# Session 5 - Advanced Topics
- https://yumeng5.github.io/files/kdd23-tutorial/Part5.pdf
- https://github.com/mickeysjm/awesome-taxonomy