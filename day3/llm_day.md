Notes from Large Language Model Day with sessions spanning from 10am-5:30pm on 2023-08-08.

<https://bigmodel.ai/llmday-kdd23/>

# Opening Remarks: Jie Tang (Tsinghua University)

- Opportunity to KDD: Unified model for all data (text, DB, image, multimedia)
- Opportunity to Academia: 
  - Before we categorize our research vertically
  - Next, we may need to reorganize our research horizontally
- Opportunity to startup:
  - LLM offers a chance to implement AGI
  - Vision of a new startup is not necessary to answer "how", it is more about "why"?

# From Documents to Dialogues: How LLMs are Shaping the Future of Work - Jaime Teevan (Microsoft)

How we share and disseminate knowledge

- Also going to mirror's Ed's talk where Jaime is going to share about a history.
- How the future of work is going to change and how the history of academic research
- Pursuit of GDP Growth [slide]
- "The Current Moment"
- The search engine was the first AI-scale application. Enabled by the cloud.
  - Worked at Infoseek in the early days
  - Had 4 kids during this time (oldest was born in 2004, oldest in 2008)
  - In those 4 years: Both cloud, edge devices (mobile) and social networks (create information at scale) came into existence. Mechanical Turk was also created. The combination of those things created ImageNet, which led to much development.
  - Have to be able to make use of fragmented time and attention. Her focus of research. Micro-productivity and how people work with AI.
- Worked at Microsoft Research after doctorate
  - Information retrieval
  - Worked with Satya Nadella as his technical advisor
- Now here role is to think about how to drive disruption, bring research into products
  - in the midst of this was the pandemic/COVID.
- AI is more of an internal disruption than an external one
  - Spotify: "Today has been a great year in AI"
- Fall 2022: Meeting with Sam Altman getting to try out GPT-4 for the first time. Had to drive to campus to try it out because it was so secretive.
  - Your job is to get this into all of Microsoft's products
  - Driving home, had to pull over and scream because it was so amazing

## Enterprise Grade AI
taking the world's best language models and making them work for the enterprise involves these 4 things:
  - Global Scale
  - Grounded in your Data
  - Trustworthy
  - Embedded in Existing Workflows
- Global Scale: Multilingual
  - shipped M365 copilot in all tier 1 languages because of language model's ability to translate
  - translating through EN improves performance for some languages but not for others
    - high resource languages tend to do better without needing to go through English
    - non-latin scripts tend to have additional challenges
- Global Scale: Efficiency
- **Just like how everyone uses the internet, everyone will use language models**
- Global Scale: Sustainable
  - Underwater data centers are a thing!
- Grounded in your data: RAG [slide]
  - search over knowledge and new types of data: documents, chat history, application context
  - grounded in the context:
    - word document
    - chat history
    - transcript
- Grounded in your data: Private by design [slide]
- Trustworthy: Differential privacy
  - Better privacy/utility trade-off: similar accuracy for private vs. non-private fine-tuning
  - Privacy episilon?
- Trustworthy: Measuring privacy
  - Combining DP and PII scrubbing effictively eliminates the risk of privacy
- Trustworthy: Responsible
  - requires layers of mitgation
  - LLMs foreground new challenges
    - Hallucination and errors
    - Jailbreaks and prompt injection
    - Harmful content and code
    - Manipulation, human-like behavior
  - Enterprise vs. consumer context
- Embedding in Existing Workflows: ODSL
  - Office Domain Specific Language
  - Excel is Turing Complete!
    - As soon as you can translate natural language into a DSL, you unlock a lot of value
    - Excel + LLMs

## Knowledge in Conversations

- Grounding vs. Grounded
  - After grounding, we create grounded content
  - In the LLM context: the first few prompts are the process of grounding
  - Microsoft is a document company - LLMs can extract knowledge from those documents. 
    - Not just facts
    - Style
    - Structure
- Collective Intelligence - how to capture all the knowledge across a community like KDD and synthesize it
  - Analogy: maps. Structured data. Allows us to navigate the world and create geopolitical boundaries. Google Maps has centralized a lot of this knowledge. 
  - Chat Log Analysis
    - Come up with Prompting Do's and Don'ts
- Prompt Support: Creating the LLM "Ribbon"
  - support learning: identify and surface tips in situ

## Lead like a Scientist

## Q&A
- What do you foresee about the future of remote work?
  - Research shows in person work is valuable
  - Language models may help people access fast real-time institutional knowledge 
- Once we have products built on top of LLMs, how do you see the future of development?
  - A lot of this will be emergent behavior
  - With Twitter/X, they didn't invent hashtags
- What do you see the challenges of ethics by design?
  - ?
- How do you audit a model whether you are meeting those metrics?
  - For Microsoft products: there are metrics tied to each guideline in their Responsible AI guidelines
- Can you comment on the future of LLMs in the context of open source? As scientists, how can we validate models that are closed?
  - Human parity has been our measure for so long, what's next?
  - This is something we have to figure out.
  - Source selection for RAG
  - Model selection for efficiency is important
- How do you think about security at Microsoft?
   - Red-teaming for security and ethics
   - Interesting: a lot of models are used in red-teaming efforts.

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
- Implementation: Too 

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

## Smmmary
[slide]

- These ideas are trivial if LLMs are humans

# Llama 2: Open Foundation and Fine-Tuned Chat Models - Vedanuj Goswami (Meta FAIR): 

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

# From GLM-130B to ChatGLM - Peng Zhang (Zhipu AI)

- teaching machines to think like humans
- all-in on LLMs, 400 people working on this
- Zhipu's GLM models vs OpenAI GPT models
  - GLM (auto-regressive blank-filling) vs. GPT (generative pre-training)
  - Tokens: a mix of English and Chinese models
- General Language Model (GLM)
- Training stability
  - Tradeoff between stability and efficiency

# The large language model renaissance: paradigms and challenges - Jason Wei (OpenAI)

- Timeline
  - 2018 - BERT
  - 2023 - ChatGPT - ask in natural language
  - Contrast between the two is night and day
- What is legacy thinking? How can we adapt to the new way of thinking?
- Outline
  - Scaling Laws
  - Emergent abilities
  - Reasoning via prompting
  - How they work technically and how it affects AI work.
- Scaling Laws
  - Palm2 more than 1 mole (6e23) of flops
  - Scaling is hard and was not obvious at the time
   - Technical challenges
   - Psychological challenges
  - Scaling predictability improves performance ("scaling laws")
    - Test Loss vs. Compute
    - Increase compute --> loss is supposed to go down smoothly
    - You should expect to get a better language model as you scale compute
    - Spans 7 orders of magnitude
  - Scaling laws: certain metrics can be very predictable
  - Changes in the nature of AI work: scaling laws
    - 5 years ago: Many individual or small-scale projects, bottom-up research culture, run the code once; then submit to NeurIPS
    - Technical paradigm shift (b/c/ training the best models require scaling compute and data)
    - Now: Teams usually have dozens of people, everyone works together toward one focused goal (top down set?), tooling and infra matter a lot (increased value in being a good software engineering)
  - 202 downstream tasks in BIG-Bench
    - Smoothly increasing (29%) - small tasks
    - Flat (22%)
    - Inverse scaling (2.5%)
    - Not correlated with scale (13%)
    - Emergent abilities (33%) - flat for awhile
- Emergence in science
  - emergence is a qualitative change that arises from quantitative changes (aka phase shifts)
  - popularized by this 1972 piece by Nobel-Prize winning physicist
    - with a bit of uranium
    - with a bit of calcium...
  - emergence in large language models
    - an ability is emergent if it is not present in smaller models, but is present in larger models
  - Emergence in few-shot prompting: examples
    - performance is flat for small models
    - performance spikes to well above-random for large models
  - Emergence in prompting: example
    - Simple translation task
  - Implications of Emergence
    - there is an area of emergent abilities that can be "unlocked" with larger models
  - 3 implications of Emergence
    - Unpredictable
    - Unintentional - not specified by trainer of the model
    - One model, many-tasks
    - Suggested further reading: Emergent deception and emergent optimization
  - Changes in the nature of AI work: emergent abilities
    - 5 years ago
      - a few benchmarks for many years (CIFAR, ImageNet)
      - easy to rank models
      - task-specific architectures, data and protocols
    - Technical paradigm shift: a single model performs many tasks without the tasks being explicitly specified at pre-training
    - Now
      - Need to create new benchmarks all the time
      - Hard to decide if one model is universally better
      - Create general technology; relatively easy to pivot


## Reasoning 

- What is the difference between human intelligence and machine learning?
- Chain-of-thought prompting
- CoT itself requires scaling
  - For small models, CoT hurts performance
  - For large models, CoT helps performance
  - Why?
- Least-to-most prompting
  - the potential of problem decomposition
    - write a research proposal about the best approaches for aligning a super-intelligent artificial intelligence
- Changes in the nature of AI work: reasoning via prompting
  - 5 years ago:
    - Type 1 tasks: easy to evaluate, debug models
    - Task specification via training data and protocols
    - Black magic of AI = hyperparameter tuning
  - Technical paradigm shift: LLMs can perform multi-step reasoning via prompting
  - Now
    - Type 2 tasks: harder to evaluate / debug models
    - Task specification via natural language prompt
    - Black magic of AI = prompt engineering
  - Some potential research directions
    - Evaluation
    - Factuality 
    - Multimodality
    - Tool use
    - Super-alignment

## Q&A

- Can LLMs be used for game theory, forecasting?
  - Humans are bad at forecasting, but it may be possible for LLMs to have this behavior?
- What are your thoughts on serving models cheaper?
  - Chinchilla scaling laws - use more data to train for less data.
- What are your thoughts on models scaling and getting improved performance?
  - Based on current scaling laws we probably won't see model saturation for awhile
  - Open source community
- What about emergent abilities that are harmful that we can't see? Is it a game of cat and mouse?
  - Great blogpost on emergent deception
  - These are things that he's looking into on the "super-alignment" team
- How many papers do you read per day? How do you keep up?
  - I don't read any papers per day.
  - Fortunately works with a lot of people who work in AI
  - Uses Twitter to find what's interesting
- What about tabular data? Are LLMs good for using tabular data?
  - ChatGPT already is pretty good working with tabular data (Code Interpreter)
- Can you say anything about GPT5?
  - No.
- How did you come up with chain-of-thought?
  - Really interested in meditation. Stream of consciousness was a thing in meditation. LLMs can produce stream of consciousness. Asked if LLMs could solve word math problems and he applied stream of consciousness. Originally thought it could be called stream-of-thought but sounded informal so he called it chain-of-thought.

## Thoughts

- Jason talked about how LLMs will affect AI Research (his forte), but how does LLMs affect any profession?

# Paradigm Shifts in the Era of LLMs: Opportunities and Challenges in Academia, Industry, and Society.

- Panelists: Ed Chi (Google DeepMind), Vedanuj Goswami (Meta FAIR), Jaime Teevan (Microsoft), Vy Vo (Intel Labs), Denny Zhou (Google DeepMind)
- What's one perspective on LLMs you have?
  - Denny Zhou: Happy that many people know the importance of reasoning. Reasoning is all you need.
  - Vy Vo: Comparing human brains vs. LLMs. Comparative intelligence approaches will be more important going forward.
  - Ed: Has a background in HCI and cognitive science
  - Vedanuj: On the panel because he helped training LLaMA2 models. 
  - Jaime: On the panel because she brings organizational diversity. Knowledge is increasingly embedded in conversations. How do we setup people to have successful conversations?
- Some researchers in certain fields fear their fields will be eliminated because of LLMs. What fields will disappear?
  - Lawrence: There's an analog to work/jobs
  - Jaime: Should be excited to reimagine how research is conducted. How knowledge is disseminated. We have an amazing new tool on our hands now. 
  - Ed: People working on methods are more likely to be out of a job, people working on data are more likely to be safe. 40-50% of questions touched on knowledge graphs and injecting them into LLMs - this community spent years on turning unstructured data into a knowledge graph. But a LLM is probably more capable than a knowledge graph. Turn knowledge graphs back into text that can be used to train LLMs. 
  - Vy: Information is represented in multiple forms. Don't think that language is the only useful way to represent data/information. 
    - We still don't understand how LLMs are doing what they are doing
    - Emerging field to understand machine intelligence. Will be able to steer it. 
    - Grapple with the comparative approach.
  - Denny: LLMs are much weaker than what we thought. 
    - Inconsistency: GPT-4 will make mistakes on elementary school math problems. 
    - Where does reasoning come from? If it doesn't come from language, can we use synthetic language to understand reasoning.
- What are some research directions to understand how LLMs are working?
  - Ed
    - Geoff Hinton said if an ML is making a good prediction, why do we care how he or she is making it?
    - Humans are a black box. Cogitive science - humans make decision and justify their decisions post-hoc.
    - You can interrogate the LLM directly for why it made a prediction/decision. 
  - Vy: Deep learning framework for neuroscience. All animal brains are black boxes. Paper said what is being optimized for in the biological brain and AI systems. Look at the loss function that the model is being trained on.
- Is it responsible for human beings to use that much energy? What is the future of AI in this regard?
  - 5 watts to think
  - 1-2 watts to generate a word
  - 540B or 175B parameter model - 400W per GPU
  - Jaime: Not trying to abdicate human intelligence. But like any tech, it's expensive in the beginning. Will expect it to improve in efficiency going forward.
- Are there smarter ways to do research in LLMs without 4,000 GPUs?
  - Denny: Few companies are going to build LLMs. But research community can try to understand HOW LLMs work. 
  - Ed: 30% of job is to how to organize research and set research agendas. Decompose the problem into 4 dimensions: data, compute, people and structure. Think about whether you have an advantage in these 4 areas. When you look at industry, startups and academics, it's hard to compete on data and compute but they can compete with people and structure. It's not just all about scale. The reasoning breakthroughs in LLMs came up by understanding a little bit about cognitive science and education.
- How do I define my core strengths as a researcher?
  - Jaime: We don't know what we don't know. We are trained to figure that. What are the skills that a human needs? These are big hard questions. Whether you are a big tech company, startup or an 18-year old you have to figure stuff out.
  - Ed: Two alternative futures. Living in the Bay Area, encounter investors who are investing into startups for AI for enterprise use cases. 
    1. Treat a LLM as a generic human. Augment it with a retrieval data, it will work as an assistant. As a VC I will be very happy. If you believe in Denny's work, this is Scenario 1.
    2. Train the model on a custom corpus, RAG only will not get to you human-level performance. As a VC I will be unhappy.