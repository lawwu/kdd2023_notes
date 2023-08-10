# Llama 2: Open Foundation and Fine-Tuned Chat Models - Vedanuj Goswami (Meta FAIR)

- Pretraining
- Finetuning (SFT and RHLF)
- Safety

## Pretraining

- 2T tokens, 40% more than LLaMA1 
- 1.5x-7x more compute than LLaMA1
- grouped query attention
- scaling training beyond 2K GPUs
- GQA was only used for the 34B and 70B models
- More compute and longer training - don't see model saturation yet. Can train for model to see better performance.
  - Training optimal: Given a given amount of compute
  - Inference optimal: more in the regime of inference optimal
- Long context pretraining (2k to 4k)
  - context length to use in pretraining is determined by the pretraining data distribution
  - when you have longer context length, the kv-cache increases a lot and it becomes infeasible to have larger batches
  - GQA - 8 heads vs. 1 head in MHA (multi-head attention)
- Parallelism

## Pretrained Model Evaluation
  - on par or better with open source models
  - still gap with closed source models

## Fine-tuning

- SFT or Instruction tuning - similar to the point Denny made in his talk, this allows you to encode instructions/prompts within the model itself
  - 3rd party datasets lack diversity and quality
  - focus on fewer but clean instruction-tuning data for higher quality models
  - collected about 27k samples
  - SFT model output often matched or outperformed human annotated data. So they shifted to getting human preference data with the remaining budget.
- Human preference data
  - 1.4M samples collected
  - Compared to other open source datasets like Anthropic, OpenAI, StackExchange etc., their samples were longer
- Reward Modeling - want the model to helpful and harmless
  - usually these rewards are in conflict so they decided to train two separate models
- Iterative Finetuning with RLHF
  - 5 total iterative loops
  - Two approaches: Proximal Policy Optimization, Rejection sampling. The 5th iterative loop was done with PPO AND rejection sampling.
- Favorable comparison win-rate vs ChatGPT on helpfulness and safety
- HumanEval results compared to other open source models on single turn and multi-turn prompts

## Safety

- Impact of Safety RHLF - improvement in safety without sacrificing helpfulness
- Context Distillation
  - Generate safety pre-prompts using various adjectives associated with safe behavior like responsible, respectful, wise
  - prefix a safety pre-prompt to adversarial prompts
- Safety Evaluation
  - Multi-turn evaluation generates more unsafe responses

## Some interesting observations

- dynamic rescaling of temperature contingent upon the context for creative vs. factual prompts. For factual prompts, model provides same response in spite of rising temperature
- models show emergent behavior to organize knowledge by time
- tool usage also emerges

## Q&A

- How did you ensure the training was stable? The training curves are smooth?
  - Depends on calibrating the learning rate and the quality of the data. Bad batches of data lead to spikes.
- How to make these models learn continually?
  - Inspired by Denny's talk, if we can improve it's reasoning capabilities, then it may just need a few prompts to generalize to new unseen tasks.
- What's your opinion on these two steps of SFT vs. RHLF?
  - Once models get to a certain size, the model gets better than humans at generating SFT-like responses
  - It's easier for humans to compare two responses than to generate a response (gives the model an idea of human preferences)
- How long did it take to train your models?
  - 45 days to train 70B on 2,000 GPUs
