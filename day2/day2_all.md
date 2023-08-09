# Takeaways from Day 2

- Ed Chi had my favorite line from the day:
    - Humans + Search --> Superhuman
    - LLMS + Tools --> Super LLMS
    - Humans + Super LLM --> Super super humans??
- Reaffirmed the LLM space is moving very quickly. There are areas of research that if not explored in the next year or so, it will be too late to make a meaningful contribution.
- Learned some new methodologies:
    - LLMs: Prompt Tuning, Mixture of Experts
    - Causal ML: Double Machine Learning (DML), many packages to do Causal ML like CausalML, EconML and UpliftML
- Two groups in an A/B test may not be sufficient, need to account for 4 groups

# LLM Workshop: Foundations and Applications in Large-scale AI Models - Pre-training, Fine-tuning, and Prompt-based Learning

The website for this workshop is here: [https://llm-ai.github.io/llmai/](https://llm-ai.github.io/llmai/).

## LLM Revolution: Implications from Chatbots and Tool-Use to Reasoning - Ed Chi

Ed Chi from Google gave this great talk.

### 2016 - Functions that Deep Neural Network Can Learn
- Pixels --> Lion
- Audio --> Audio to text
- Text --> Text (translation)
- Pixels --> Caption

### Chatbots
- Not just transactional
- We want chatbots to be contextual
- Personalized assistants for everyone

### Lambda --> Bard (Brought to You by Ed's Team)
- They wanted to publish Lambda in the form of Bard, but there were difficulties

### Large Language Models (LLM)
- Large knowledge base
- What is a plan to read 20 books a year? Reaches into the LLM to come up with a real plan
- Genesis of captions --> not too far to be able to generate text

### Programming
- Coding is less about coding, more about data
- Data Science (DS) is going to be a bigger part of software development

### Retrieval Augmentation: Leveraging External Knowledge
- Factuality trigger
- Open-book Generative QA
- RETRO: Retrieval-augmented generative model
- Questions:
  - How big does the LLM need to be?
  - How big does the external knowledge base need to be?
  - Fruitful Line of Research

### Multi-modality output (not just text, could be images)
- Image retrieval
- Image input --> Generate captions

### Humans and LLMs with Tools
- Humans + Search --> Superhuman
- LLMS + Tools --> Super LLMS
- Humans + Super LLM --> Super super humans??

### Future Challenges
- Responsibility and Safety
- Factuality, Grounding, and Attribution
- Human <-> AI Content Loop and Ecosystem
- Personalization and User Memory

### Keynote
- Ed is going to give the keynote tomorrow
- You can interrogate a model for why it made a decision or prediction
- Area: Self-critique, self-reflection (next year or so)
- 3-5 year research topics:
  - Hallucinations / Bias in areas where the LLM has not been trained
  - Relationship between hallucinations and safety

## Large-scale AI Model Research at Google Pre-training, Fine-tuning, and Prompt-based Learning

Tania Bedrax-Weiss from Google gave this talk.

### Mixture of Experts Models
- How to route the question to the right expert, right experts

### Conditional Computation
- COLT5 Transformer layer
- Scales to longer context
- Early exit
- Per step confidence thresholds

### Multi-modal Work
- Imagen - diffusion model
  - [Imagen Research Google](https://imagen.research.google/)
- Parti - autoregressive model
  - [Parti Research Google](https://sites.research.google/parti/)

### Imagen: Technical Details
- ViT-VQGAN as image tokenizer
  - What's an image tokenizer? See: https://keras.io/examples/vision/token_learner/
- Autoregressively generate images in a similar way that LLMs generate text
- Can generate text reliably - spell words out unlike other models

### Pali
- Image to text
- State of the art text captioning model

### Spotlight
- Screenshots / user interfaces - understand what are the actions that a user can perform
- Execute commands in the user interface

### PLay: Parametrically Condition Layout Generation Using Guidelines
- Fine-tuning
- Prompt Tuning
  - Look at this more

### How do you handle ambiguity in an answer?
  - LLMs are very eager to give an answer
  - Types
    - Use multiple prompts to get different types of answers. This is my answer. Can you generate other answers?
    - Diversity objectives


## Retrieval-Augmented Multimodal Language Modeling

Paper: [https://arxiv.org/abs/2211.12561](https://arxiv.org/abs/2211.12561)

Recent multimodal models such as DALL-E and CM3 have achieved remarkable progress in text-to-image and image-to-text generation. However, these models store all learned knowledge (e.g., the appearance of the Eiffel Tower) in the model parameters, requiring increasingly larger models and training data to capture more knowledge. To integrate knowledge in a more scalable and modular way, we propose a retrieval-augmented multimodal model, which enables a base multimodal model (generator) to refer to relevant text and images fetched by a retriever from external memory (e.g., documents on the web). Specifically, for the retriever, we use a pretrained CLIP, and for the generator, we train a CM3 Transformer on the LAION dataset. Our resulting model, named Retrieval-Augmented CM3 (RA-CM3), is the first multimodal model that can retrieve and generate both text and images. We show that RA-CM3 significantly outperforms baseline multimodal models such as DALL-E and CM3 on both image and caption generation tasks (12 FID and 17 CIDEr improvements on MS-COCO), while requiring much less compute for training (<30% of DALL-E). Moreover, we show that RA-CM3 exhibits novel capabilities, such as faithful image generation and multimodal in-context learning (e.g., image generation from demonstrations).

- Develop a retrieval-augmented multimodal model, a first of it's kind
- The generator uses retrieved items for generation too
- Retrieval augmented training - helped a lot

## In-Context Learning User Simulators for Task-Oriented Dialog Systems

- Code: [https://github.com/telepathylabsai/prompt-based-user-simulator](https://github.com/telepathylabsai/prompt-based-user-simulator)
- Paper: [https://arxiv.org/abs/2306.00774](https://arxiv.org/abs/2306.00774)

This paper presents a novel application of large language models in user simulation for task-oriented dialog systems, specifically focusing on an in-context learning approach. By harnessing the power of these models, the proposed approach generates diverse utterances based on user goals and limited dialog examples. Unlike traditional simulators, this method eliminates the need for labor-intensive rule definition or extensive annotated data, making it more efficient and accessible. Additionally, an error analysis of the interaction between the user simulator and dialog system uncovers common mistakes, providing valuable insights into areas that require improvement. Our implementation is available at this https URL.

- Rule based systems are still more accurate. However they mainly understand happy paths of a dialog system.
- These LLM based approaches can explore unexpected behavior of users

## Challenges in post-training quantization of Vision Transformers

Paper: [https://research.ibm.com/publications/challenges-in-post-training-quantization-of-vision-transformers](https://research.ibm.com/publications/challenges-in-post-training-quantization-of-vision-transformers)

Vision Transformers recently showed outstanding performance in computer vision tasks. However, those models are compute and memory intensive that require accelerators with a large amount of memory like NVIDIA A100 graphic processing unit for training and even for inference. Post-training quantization is an appealing compression method, as it does not require retraining the models and labels to tune the model. In this paper, we look in depth at multiple models in terms of size, architecture, and training procedure and provide guidelines on how to quantize the model to an 8-bit integer, both weights and activations. We perform a well-rounded study on the effects of quantization and sensitivity to the quantization error. Moreover, we show that applying mixed-data precision quantization works well for most vision transformer models achieving up to 90% compression ratio within a 2% top-1 accuracy drop. This kind of quantization offers a trade-off between memory, compute, and performance of the models that are deployable with the current software and hardware stack.

- There's a difference between Static vs Dynamic Quantization 
- Larger models are supposed to be easier to quantize, but not the case here
- Signal to noise quantization ratio - SNQR 
- Partial Quantization: Some models that lost accuracy during dynamic quant, regained during 90% quant

## Generalization in Graph Neural Networks: Improved PAC-Bayesian Bounds on Graph Diffusion

Paper: [https://proceedings.mlr.press/v206/ju23a/ju23a.pdf](https://proceedings.mlr.press/v206/ju23a/ju23a.pdf)

Graph neural networks are widely used tools for graph prediction tasks. Motivated by their empirical performance, prior works have developed generalization bounds for graph neural networks, which scale with graph structures in terms of the maximum degree. In this paper, we present generalization bounds that instead scale with the largest singular value of the graph neural network’s feature diffusion matrix. These bounds are numerically much smaller than prior bounds for real-world graphs. We also construct a lower bound of the generalization gap that matches our upper bound asymptotically. To achieve these results, we analyze a unified model that includes prior works’ settings (i.e., convolutional and message-passing networks) and new settings (i.e., graph isomorphism networks). Our key idea is to measure the stability of graph neural networks against noise perturbations using Hessians. Empirically, we find that Hessian-based measurements correlate with observed generalization gaps of graph neural networks accurately; Optimizing noise stability properties for fine-tuning pretrained graph neural networks also improves the test performance on several graph-level classification tasks.

- Overfitting if there's an imbalance between pretraining data and finetuning data size
- Generalization gap
    - Not just cross validation loss
    - More detailed understanding - what networks are causing the overfitting
    - Generalization gap - measures the gap between training/test losses

## NLP Research in the Era of LLMs - Unleashing the Potential of LLMs through Task and Data Engineering

Shafiq Joty gave this talk: https://raihanjoty.github.io/

### Background: Data Engineering
- Hold the code fixed and invite research to improve the data (Andrew Ng)

### Background: Rise of Task Engineering
- Multi-task models with task prompts
- Trained with many different instructions
- Mentions prompt tuning again (soft tokens) ???

### Background: Task Engineering
### LLM Lifecycle
### **XGen LLM**: June 2023
- [GitHub Link](https://github.com/salesforce/xgen)
- Goal is to outperform LLaMA1

### Instructed tuned
- Instructional data: WizardLM. [Paper Link](https://arxiv.org/abs/2304.12244)

### What does WizardLM do exactly in advancing the SoTA?
- [Details on WizardLM](https://arxiv.org/abs/2304.12244)
- Training large language models (LLMs) with open-domain instruction following data brings colossal success. However, manually creating such instruction data is very time-consuming and labor-intensive. Moreover, humans may struggle to produce high-complexity instructions. In this paper, we show an avenue for creating large amounts of instruction data with varying levels of complexity using LLM instead of humans. Starting with an initial set of instructions, we use our proposed Evol-Instruct to rewrite them step by step into more complex instructions. Then, we mix all generated instruction data to fine-tune LLaMA. We call the resulting model WizardLM. Human evaluations on a complexity-balanced test bed and Vicuna's testset show that instructions from Evol-Instruct are superior to human-created ones. By analyzing the human evaluation results of the high complexity part, we demonstrate that outputs from our WizardLM are preferred to outputs from OpenAI ChatGPT. In GPT-4 automatic evaluation, WizardLM achieves more than 90\% capacity of ChatGPT on 17 out of 29 skills. Even though WizardLM still lags behind ChatGPT in some aspects, our findings suggest that fine-tuning with AI-evolved instructions is a promising direction for enhancing LLMs. Our code and data are public at this https URL

- **Verify and Edit CoT** - Self-consistency
- Knowledge adapting framework
- Language diversity prompting
- Standard vs Personalized Distillation from LLMs

## Modular Large Language Model and Principle-Driven alignment with Minimal Human Supervision

Yikang Shen from IBM gave this talk.

### Foundation model types

#### Challenges of LLM
- **Efficiency**
- **Extendability**
- **Flexibility**

### ModuleFormer - Learning Modular LLM from Uncurated Data
- Previous modular models were based on already labeled data

### Mod-Squad - designing a mixture of experts as modular multi-task learners
- Can select the right experts for a task
- Experts can share knowledge!?

### Dromedary - efficiently teach AI to follow a given set of principles
- [GitHub Link for Dromedary](https://github.com/IBM/Dromedary)
- **Principle Engraving** -
- **Verbose Cloning** - refining the model to produce in-depth and detailed response
- 300 lines of annotations
- Kind of similar to Evol-Instruct/WizardLM to produce annotations to fine-tune a model

## AutoHint: Automatic Prompt Optimization with Hint Generation

Paper: [https://arxiv.org/pdf/2307.07415.pdf](https://arxiv.org/pdf/2307.07415.pdf)

This paper presents AutoHint, a novel framework for automatic prompt engineering and optimization for Large Language Models (LLM). While LLMs have demonstrated remarkable ability in achieving high-quality annotation in various tasks, the key to applying this ability to specific tasks lies in developing high-quality prompts. Thus we propose a framework to inherit the merits of both in-context learning and zero-shot learning by incorporating enriched instructions derived from input-output demonstrations to optimize original prompt. We refer to the enrichment as the Hint and propose a framework to automatically generate the hint from labeled data. More concretely, starting from an initial prompt, our method first instructs a LLM to deduce new hints for selected samples from incorrect predictions, and then summarizes from per-sample hints and adds the results back to the initial prompt to form a new, enriched instruction. The proposed method is evaluated on the BIG-Bench Instruction Induction dataset for both zero-shot and few-short prompts, where experiments demonstrate our method is able to significantly boost accuracy for multiple tasks

# Causal Inference Workshop: Causal Inference and Machine Learning in Practice

The website for this workshop is here: https://causal-machine-learning.github.io/kdd2023-workshop/

## COG: Creative Optimality Gap for Video Advertising

Raif Rustamov from Amazon gave this invited talk.

### Video ads motivation
- How does a particular video affect shopper experience?

### Goal
- Driven by explicit hypotheses tied to quantifying value of the video

### Approach - Creative Optimality Gap (COG)
- If we were to replace the video of class 0 to video of class 1, what would be the improvement in the outcome for the ad?
- **Uplift or Heterogenous Treatment Effect modeling**

### Benefits
- Differentiated at the level of video features vs. global ATE
  - **ATE** - average treatment effect - videos are good
  - **ITE** - individual treatment effect - noisy
  - **HTE** - heterogeneous treatment effect - in the middle, denoising
- Handle cold start ads

### Preliminaries
- **Treatment indicator (T)**
- **Video features**
  - Computed using e.g. video embeddings
  - Can contain non
- **Ad features**
  - Contains non-video related features like price, product category
  - Used as confounder/matching variables
- **Outcome = Y**

### COG Modeling
- **Step 1**
- **Step 2**
- **Step 3** -
  - Used interpretable models in this step, why?

### COG Modeling: Guardrails
#### Bias
- Bias comes from G model, comes from regularization or not enough capacity in the model
- Bias is not constant but varies in the Z space
- Double ML?

#### Uncertainty/Variance

### Solution
- Conservative COG = lower bound of confidence interval


## The Value of Last-Mile Delivery in Online Retail	

Ruomeng Cui from Emory gave this talk.

### Cainiao - Chinese Company
- Alibaba's logistics platform
- Largest logistics platform in China
- If there are differences in preferences, there is an opportunity for optimization

### Use Causal ML: Estimating ITE
- **Data:** Post-treatment data Q4 2021

### Models
- Partial Linear DML
- First-difference DML
- Others

### Account for Knapsnack 
- Tau does not capture economic efficiency
- Need to account for how much capacity a customer is using. A customer going from 0 to 1 unit sales is much more valuable than a customer going from 19 to 20 units sold because the latter is not using much capacity.

## Leveraging Causal Uplift Modeling for Budget Constrained Benefits Allocation

Dmitri Goldenberg from Booking.com gave this talk. It was a very good talk with virtually no words on his slides.

## Ensemble Method for Estimating Individualized Treatment Effects	Kevin Wu Han, Han Wu (Stanford)

- Paper: [https://arxiv.org/abs/2202.12445](https://arxiv.org/abs/2202.12445)
- Ensemble methods almost always perform a validation-set model selection based method!

## A Scalable and Debiased Approach to Dynamic Pricing with Causal Machine Learning and Optimization

- Heard the term double machine learning for the second time which caused me to do to learn what it is.

## An IPW-based Unbiased Ranking Metric in Two-sided Markets	Keisho Oh, Naoki Nishimura (Recruit Co), Minje Sung, Ken Kobayashi, Kazuhide Nakata (Tokyo Institute of Technology)

In two-sided markets like job-matching or dating-apps, need to use an unbiased ranking metric which they propose in their paper.

## Unit Selection Based on Counterfactual Logic	

This was an invited talk by Ang Li about this paper: [https://ftp.cs.ucla.edu/pub/stat_ser/r488.pdf](https://ftp.cs.ucla.edu/pub/stat_ser/r488.pdf).

My main takeaway was dividing a population into a typical A/B test where one group receives a treatment and the other group is the control is too simplistic. There are actually 4 groups we should be concerned about: 

- Complier: Individuals who would respond positively if treated and negatively if not treated.
-  Always-taker: Individuals who always respond positively no matter whether they are treated or not.
- Never-taker: Individuals who always respond negatively no matter whether they are treated or not.
- Defier: Individuals who would respond negatively if treated and positively if not treated.

Along with a benefit vector that assigns a positive or negative value to each of these 4 groups, we can use this to select the best treatment for each individual.

Ang also used the Pfizer Covid vaccine as a motivating example for why these 4 groups should be accounted for.

## Towards Automating the Causal Machine Learning Pipeline	Vasilis Syrgkanis (Stanford/EconML)

- A large variety of causal estimands that arise in complex static and longitudinal data analysis can be automatically de-biased when regularized machine learning algorithms are used to estimate nuisance models
- Estimation of the de-biasing term itself can be performed with generic machine learning
- Experimental results using neural nets and random forests for automated de-biasing provide examples superior performance to plug-in approaches and to prior automatically debasing approaches based solely on linear models