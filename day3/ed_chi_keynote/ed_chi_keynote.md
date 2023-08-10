Notes from Ed Chi's keynote at KDD 2023, 2023-08-08

# Takeaways

- 8-year tech cycles led to breakthroughs in web, search, mobile & AI (1991, 1999, 2007, 2016). We're overdue for another breakthrough but LLMs will bring about the next one.
- We had all the pieces for LLMs back in 2016 (deep learning models that could caption images)
- Outlines the timeline of Large Language Models. The future: LLMs augmented with tools, humans with the tool-assisted LLMs will lead to augmented human intelligenceO
- Key Idea: Teaching LLMs the way we teach kids led to key-breakthroughs like Chain-of-Thought Prompting, Self-consistency decoding, and Least-to-most Prompting
- LLMs+Reasoning Key Ideas:
    - Chain-of-thought prompting: ‹question, explanation, answer›
    - Self-consistency: solve multiple times and choose most common answer
    - Least-to-most prompting: decompose and solve subproblems
    - Instruction finetuning: teach LLMs to follow instructions
- LLMs augmented with tools are capable of reasoning via language


# Notes 

# 8 year cycles
- 1991 - invention of web browser
- 1999 - invention of Google and search
  - indexed the world's books
- 2007 - invention of mobile phone
- 2016: Functions that Deep Neural Network Can Learn
  - Text: Humans can be augmented by statistical machine translations
  - Image --> Caption: Put in an image and it would generate salient captions. This was an incredible breakthrough since no specific instructions were given but accurate captions were **generated**.
  - How many people recognized in 2016 that there was going to be a generative AI cycle.

- By 2016, image recognition by machines/AI was already super-human, 3% error rate vs 5% error rate
- What changed about deep learning to make it a useful method?
- Getting comfortable with revolutions: as humans we couldn't fly 100 years ago. But now, we get delayed 30 minutes and get annoyed.
  - "Any sufficiently advanced technology is indistinguishable from magic." - Arthur C Clarke
  - "Any commonplace technology is not magic!" - Ed Chi's corollary

#  LLM-based chatbots and asssistants 

- Timeline of how we got here
  - 2014: sequence to sequence learning with neural networks
  - 2017: Attention is all you need
  - 2020: Towards a Human-like Open Domain Chatbot
  - 2022-01: LaMDA: Language Model for Dialogue Applications
  - 2022-01: Chain-of-Thought Prompting Elicits Reasoning in LLMs
  - 2022-02: Finetuned Language Models are Zero-shot learners
  - 2022-04: PaLM: Scaling Language Models with Pathways

- Multi-task language models
- LaMDA
  - Large model, up to 137B parameters
  - Fine-tuned for sensibility, specificity, interestingness, safety, factuality
  - Foreshadowed Bard

- Possible soon: Everyone can have their own personal assistant that is not merely transactional but understands contexts.
- Bard
  - With Bard, we realized LLMs could begin to do planning, "Help me design a plan to read 20 books in a year"
  - No longer just a cute assistant but they could be helpful assistants in a wide variety of topics
  - Improved multilingual understanding, understanding idioms. Explain idioms and why they can be misunderstood.
  - Coding capabilities: more surprising to Ed. 
    - explain JAX, explain the code within the google/github repo
    - can you fix the code with a bug and add line by line comments in Korean
    - Learning about Double Machine Learning
  
- Insight (Data + Data Efficiency is the key to Conversational AI)
  - Pre-training (1B examples, 1T tokens)
  - Fine-tuning (10k examples)
  - Prompting (1 example)
    - small changes, micro-prompting can lead to big changes in the text that is generated. there is a parallel to human language where small changes to what you say can elicit different responses, e.g. adding "please" and "thank you"

# Tool-use: Retrieval-Augmentation and Multi-modality in LLMs

- Limitations of LLMs
- Retrieval-Augmentation: Leveraging External Knowledge
  - TODO: insert image
  - Humans learned how to use tools like Google, how to craft the right search query to get the right results
  - We're teaching an LLM how to use tools like Google
- RETRO: Retrieval-augmented generative model. 
  - the generator processes the question and the retrieved docs/passages separately
- Multi-modality output
  - Query generation may call an image service or an image generation service
  - Image input and output
- Coming Soon: Tool-use App Integration
  - 1st party tools from Google
  - 3rd party tools: Adobe Firefly to generate images, Uber, shopping service

# Augmented Intelligence

- Human intelligence is augmented by search (needed two revolutions: mobile and search)
- LLMs + search/tools --> Super LLMs
- Humans + Super LLMs --> Super super humans?!

# Reasoning

## Human Intelligence vs Machine Learning 

- Denny Zhou: Differences between how humans and machines learn.
- Humans learn from only a few examples while machine learning needs tons of labeled data to train a model
- Attempts to fill the gap
TODO: See slide

## Can we teach LLMs like we teach kids?

- hypothesis on how to improve reasoning

### Chain-of-Thought = "explanation" + "answer"

- 2022 paper
- Breakthrough capabilities: Reasoning tasks
  - Standard prompting
  - Chain of thought prompting - adding an example of an explanation/answer to the prompt allows the LLM to replicate that reasoning capabilities

### Self-consistency decoding

- Paper: Self-Consistency Improves Chain of Thought Reasoning in Language Models
- if you create diverse reasoning paths with different temperatures and then took a majority vote, this would lead to better performance

### Least-to-most Prompting

- Paper: Least-to-Most...
- Decompose question into subquestions
- Sequentially solve subquestions

### Instruction Fine-tuning: Enables zero-shot prompting

- ML: Data --> Prediction
- Instruction fine-tuning: Instruction --> Answer
- FLAN Instruction tuning
  - Fine-tunes model on a large set of varied instructions that use a simple and intuitive description of the task
  - instruction tuning improves performance on unseen tasks only for models for certain sizes (68B)

### FLAN2: Fine-tune with 1800+ tasks, bigger models

### Reasoning Summary

- Chain-of-thought prompting: ‹question, explanation, answer›
- Self-consistency: solve multiple times and choose most common answer
- Least-to-most prompting: decompose and solve subproblems
- Instruction finetuning: teach LLMs to follow instructions

# Future Challenges for LLMs

Interesting he made a comment, thanking OpenAI for releasing ChatGPT because that was the impetus for Google's LaMDA team being able to release Bard into the marketplace. As an outsider, it's fascinating thinking about what those internal discussions must've been like.

## Responsibility and Safety

- Constitutional AI - instruction tuning is one way to help keep LLMs safe

## Factuality, Grounding, and Attribution
- Retrofit Attribution using Research and Revision (RARR): Aligning generated answers with passages in source documents.
- Given a generated passage x, uses a three-stage approach to annotate x with retrieved evidences e

## Human <-> AI Content Loop and Ecosystem
- Many reasons for identifying machine-generated content - e.g. avoid loops of using LLM-generated examples for future LLMs
- Humans are not good at detection LLM-generated content (Ironic from a Turing test POV)

## Personalization and User Memory

- We want the LLM experience personalized to you, understand your needs, and respect your privacy
- Serving efficiency
  - Low rank models
  - If he was a PhD student, he would invest in solving this problem. Can be very profitable making models more efficient.
- Pigeonhole problem: avoiding over memorization of your preferences

# Conclusion

- History tells us we were due for a revolution and LLMs first applications are chatbots because LLMs understanding conversational context better
- Just like humans are empowered by tools and reasoning capabilities, **LLMs are now augmented with tools and capable of reasoning via language**

# Q&A

- How large do foundation models need to be?
  - Base models need to be sufficiently large or "double-digit billions". At this size, it is on the order of 1 cent per query
  - Costs are continuing to drop
- What are your thoughts about security/privacy with LLMs?
  - This is important because you don't want user preferences bleeding into other users
  - Differential privacy techniques can be applied
  - Localize some parameter changes in a separate tower that can be added to the base model
- When will LLMs develop causal reasoning capabilities?
  - One derivation from the Chain of Thought paper is that LLMs can do causal reasoning. The LLMs can explain themselves. 
- As data scientists, we are used to building specialized models and we have control over the parameters. With LLMs, the paradigm has shifted where we have to trust these pre-trained models. What are your thoughts?
  - Our ability to trust an LLM is going to take a human flavor. You build trust over time with a human. But a human usually cannot fully explain how they came to that decision.
  - You can interrogate these models for how it came up with these answers.
- How do we continue to be able to train models on human-generated data? 
  - Watermarking
    - for images is easier
    - for text - there is more sparsity
    - more pessimistic about watermarking being the solution to this problem
  - Have not heard a single proposal that is promising
  - Turing Award for work here...
- What abilities will humans retain that are superior than LLMs?
  - For the next few years at least, humans have a physical body
  - Reasoning capabilities: still some gaps between humans and LLMs
  - From the other side, machines never get tired
  - Remember 
  - Humans have a spiritual component. A machine or LLM will never have a spiritual component or a soul.
- Thoughts on evaluation of these LLMs? How can we differentiate between models?
  - In developing LaMDA and Bard, how do we build an evaluation framework to take the technology.
  - Defining three categories: quality, factuality/groundedness, safety/persona. In these 3 areas, developed sub-metrics in all of these. Spent a lot of money and time on each of these areas. 
    - Creativity: writing a poem, summarizing documents