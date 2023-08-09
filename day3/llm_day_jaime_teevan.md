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
- Pursuit of GDP Growth 
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
- Grounded in your data: RAG 
  - Trend in using LLMs: In-context learning
    - Don't fine-tune - keep the model weights static
    - Provide task instructions and data in the prompt
  - Advantages of retrieval augmentation
    - Less information needs to be embedded in the model itself
    - Relevant, context-specific results via increased real-time 
    personalization
      - word document
      - chat history
      - transcript
    - Factually grounded, less hallucination
    - Traceable information flow with explicit disclosure policies (e.g., ACL filtering)
  - Interesting research questions abound
    - New types of data: Documents, application context, chat history
    - Finding the right content: Source selection, query generation
    - Context compressing: Creating a working menu application context   
- Grounded in your data: Private by design [slide]
- Trustworthy: Differential privacy
  - Better privacy/utility trade-off: similar accuracy for private vs. non-private fine-tuning
  - Privacy episilon?
- Trustworthy: Measuring privacy
  - Combining DP and PII scrubbing effectively eliminates the risk of privacy
- Trustworthy: Responsible
  - requires layers of mitigation
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