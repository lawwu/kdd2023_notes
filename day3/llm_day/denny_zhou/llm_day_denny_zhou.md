# Takeaways

- The thing that is missing from traditional ML is reasoning. Humans can learn from a few examples because humans can reason. ML requires thousands of examples.
- Created a toy example "last-letter-concatenation" task that is difficult for ML and LLMs but trivial for humans. From this example came Chain-of-thought prompting, Self-consistency and least-to-most prompting
- Key idea of instruction-tuning: making a big prompt by combining prompts from different tasks, and then using it for any task. The big prompt is typically too large for one context window so these instruction pairs are used to fine-tune the model through instruction-tuning where then the instructions are encoded as weights in the model!
- LMs as tool makers: save cost while more accurate

# Teach language models to reason (Denny Zhou (Google DeepMind))

  - Leads the Reasoning team at Google DeepMind
  - What do you expect from AI?
  - My little expectation on AI
  - Does ML meet this expectation?
    - Semi-supervised learning
    - etc.
  - What is missing in ML?
    - Reasoning 
    - Humans can learn from a few examples because humans can reason
  - Teach LLMs to reason like we teach kids
  - Toy problem: concatenate the last letter of each word
    - Elon Musk --> nk
    - Bill Gates --> ls
    - Solve it by machine learning
    - Solve it by LLMs
  - Training an LLM
    - You can think of training an LLM like training a parrot to mimic human language
    - Few-shot prompting for last-letter concatenation
      - give it the two examples and one input to get the answer
- Why we created the last-letter-concatenation task?
  - Make ML fail
  - Make few-shot prompting
  - But trivial for humans
- Chain of though prompting
  - Adding "thought" before "answer"
  - End of 2021 - discovered this
- One demonstration is enough, just like humans
- Standard few-shot prompting vs. Chain-of-thought prompting
- Can LLMs solve math word problems?
- Apply CoT to any task
  - all tasks can be solved by CoT without machine learning
  - 100x-1000x data efficient than supervised SoTA in literature - only need 1-2 examples!
- Multilingual CoT
- Apply CoT to solve BIG-Bench Hard
- "Thought" does NOT have to be "step by step"
- Self-consistency decoding - greatly improves chain-of-thought decoding
  - prompt a language model using example chains of thought
  - sample from the LLM decoder to generate a diverse set of reasoning paths
  - choose the most consistent answer using the majority vote
  - crushed GSM8K SoTA with only 8 examples
- How many more examples are needed for fine-tuning to be comparable to CoT + SC?
  - From original paper: two additional orders of magnitude of training data to reach an 80% solve rate
  - But with only 8 examples, the model can be taught reasoning
- Solve high school math problems
  - Fine-tuning PaLM with math data
  - SC + CoT solves 50%
  - non-math grad students solve 40%
- Motivation to SC decoding
  - Answer in the greedy output from CoT DOES NOT EQUAL the most likely answer
  - SC leads to the most likely answer
    - The full probability of each answer is computed by summing over all reasoning paths (marginalize over the reasoning paths)
    - Implementation via sampling: sample multiple times, then choose the most common answer
  - Implications from the probabilistic explanation

## Least-to-most prompting

Enables easy-to-hard generalization

- CoT fails to generalize to harder problems
- Least-to-most prompting = Planning + Reasoning 
  1. Decompose a complex problem into a list of easier sub-problems
  2. Sequentially solve these sub-problems
- Solve math word problems by least-to-most prompting 
- How does a LLM plan and rank tasks easy to hard?
- Can use this for common sense reasoning
- Can use this for solving math word problems.
  - Did Aristotle use a laptop?
  - Are chinchillas cold-blooded?
- Last-letter task generalization (still not perfect)
- SCAN (compositional generalization): text to actions
  - 100% accuracy using least-to-most
- CFQ (compositional generalization): text to code
  - Using 1% of the data, crush SoTA results
- Post on compositional generalization: <https://ai.googleblog.com/2020/03/measuring-compositional-generalization.html>

## LLMs for Code

How to generate high-quality code?

- Teaching LLMs to Self-Debug
  - self-debug to generate higher quality code
- Large Language Models as Tool Makers
  - Paper: <https://arxiv.org/abs/2305.17126>
  - One way:
    - Reduce serving costs using distillation or quantization
    - For most models we use a small model
    - For some models we use a large model
  - Use a few instances to make a tool --> reuse the tool to solve similar instances
    - Analogy of deep learning libraries - talented programmers develop pytorch, practitioners `import torch` to solve their problems

## Common big prompt for any task?

Yes!

- Key idea: making a big prompt by combining prompts from different tasks, and then using it for any task
- Magic: any task: including tasks which are not even seen
- Implementation 

## Instruction Tuning

- Enable zero-shot prompting in any task
- Pioneered by FLAN and T0
- Store the large prompts in model weights
- Example of parsing all names from this message, then sort and add Quoc Le to the list.
- Why does this work?
  - What I cannot create, I do not understand - Feynmand
  - We know how to create LLMs but do not know how they work
- Emergent properties
  - All these are emergent properties
  - Emergent properties are discovered, not designed by LLM builders
- How to make parrots intelligent? Scaling up!
- Toward understanding in-context learning
  - Paper: [What learning algorithm is in-context learning? Investigations with linear models
](https://arxiv.org/abs/2211.15661)
  - learned models are encoded in activations!

## Summary
- Chain-of-thought prompting: <question, rationale, answer>
- Self-consistency: solve multiple times and choose the most common answer
- Least-to-most prompting: decompose to easier subproblems LLMs self-debug: generate much better code by debugging
- LLMs as tool makers: save cost while more accurate
- Instruction finetuning: mixing up exemplars to enable zero-shot

- These ideas are trivial if LLMs are humans
