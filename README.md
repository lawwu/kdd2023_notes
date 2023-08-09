# kdd2023_notes

I attended [KDD 2023](https://kdd.org/kdd2023/) which was held in Long Beach, CA from Aug 6-10. 
- Day 1 - I didn't attend
- Day 2 (2023-08-07) - There were half-day workshops organized around a topic. The two I attended were about LLMs (because I'm interested and it's relevant to my work) and Causal Inference (because I haven't used causal machine learning techniques in practice before and wanted exposure).
- Day 3 (2023-08-08), I attended Ed Chi's keynote and a half-day session with various speakers about Language Models.
- Day 4 (2023-08-09), I attended Eric Horvitz's keynote.

# Notes

- Day 2
    - [Notes](./day2/day2_all.md)
- Day 3 
    - [Ed Chi Keynote](./day3/ed_chi_keynote/ed_chi_keynote.md) and some of the [slides](./day3/ed_chi_keynote/slides/)
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

    - LLM Day Notes: Notes from Large Language Model Day with sessions spanning from 10am-5:30pm on 2023-08-08. <https://bigmodel.ai/llmday-kdd23/>
        - [From Documents to Dialogues: How LLMs are Shaping the Future of Work - Jaime Teevan (Microsoft)](./day3/llm_day_jaime_teevan.md)
            - Described the secretive meeting where Sam Altman demo'd GPT-4 for the first time. On the drive home she had to pull over and scream because Jaime recognized the implications of the technology.
            - **Just like how everyone uses the internet, everyone will use language models**
            - There is knowledge in conversations, chat history. How to synthesize all of this knowledge? Process of grounding (initial few prompts) to grounded knowledge.
        - [Teach language models to reason (Denny Zhou (Google DeepMind))](./day3/llm_day_denny_zhou.md)
            - The thing that is missing from traditional ML is reasoning. Humans can learn from a few examples because humans can reason. ML requires thousands of examples.
            - Created a toy example "last-letter-concatenation" task that is difficult for ML and LLMs but trivial for humans. From this example came Chain-of-thought prompting, Self-consistency and least-to-most prompting
            - Key idea of instruction-tuning: making a big prompt by combining prompts from different tasks, and then using it for any task. The big prompt is typically too large for one context window so these instruction pairs are used to fine-tune the model through instruction-tuning where then the instructions are encoded as weights in the model!
            - LMs as tool makers: save cost while more accurate
        - [Llama 2: Open Foundation and Fine-Tuned Chat Models - Vedanuj Goswami (Meta FAIR)](./day3/llm_day_vedanuj.md)
        - [From GLM-130B to ChatGLM - Peng Zhang (Zhipu AI)](./day3/llm_day_peng_zheng.md)
        - [The large language model renaissance: paradigms and challenges - Jason Wei (OpenAI)](./day3/llm_day/jason_wei/llm_day_jason_wei.md)
            - Fascinating how he made observations about how his field (AI Research) is going to change because of LLMs across 4 facets of LLMs: scaling laws, emergent abilities, and reasoning via prompting. Jason, working at the intersection of LLMs and AI Research has a good view into how LLMs will change the field of AI Research. We need this sort of thinking applied in other fields to get a sense for how to answer the question "How will LLMs change the future of work?"

                ### 1. Scaling Laws:
                Scaling is a predictable and vital aspect of improving AI performance. How will this affect AI Research work?
                - 5 years ago: Many individual or small-scale projects, bottom-up research culture, run the code once; then submit to NeurIPS
                - Technical paradigm shift (b/c/ training the best models require scaling compute and data)
                - Now: Teams usually have dozens of people, everyone works together toward one focused goal (top down set?), tooling and infra matter a lot (increased value in being a good software engineering)

                ### 2. Emergent Abilities:
                Emergent abilities refer to capabilities arising only in larger models (tens of billions of parameters). How will this affect AI Research work?

                - 5 years ago
                    - a few benchmarks for many years (CIFAR, ImageNet)
                    - easy to rank models
                    - task-specific architectures, data and protocols
                - Technical paradigm shift: a single model performs many tasks without the tasks being explicitly specified at pre-training
                - Now
                    - Need to create new benchmarks all the time
                    - Hard to decide if one model is universally better
                    - Create general technology; relatively easy to pivot, AI work now aims for general technologies, not relying on task-specific architectures and data.

                ### 3. Reasoning via Prompting:
                Reasoning and chain-of-thought prompting have transformed AI's approach to multi-step reasoning:
                - 5 years ago:
                    - Type 1 tasks: easy to evaluate, debug models
                    - Task specification via training data and protocols
                    - Black magic of AI = hyperparameter tuning
                - Technical paradigm shift: LLMs can perform multi-step reasoning via prompting
                - Now
                    - Type 2 tasks: harder to evaluate / debug models
                    - Task specification via natural language prompt
                    - Black magic of AI = prompt engineering

        - [Panel - Paradigm Shifts in the Era of LLMs: Opportunities and Challenges in Academia, Industry, and Society](./day3/llm_day_panel.md)
- Day 4
    - [People and Machines: Pathways to Deeper Human-AI Synergy - Eric Horvitz, Microsoft](./day4/keynote_eric_horvitz.md)
    - Pretrained Language Representations for Text Understanding: A Weakly-Supervised Perspective <https://yumeng5.github.io/kdd23-tutorial/>
        - [Notes](./day4/pretrained_llm_nlu_workshop/notes.md)
    - Education panel on LLMs
        - [Notes](./day4/panel_llms_in_education.md)

# Day 2 Schedule

| Time                          | Speaker                                                                                               | Title                                                                                                                         |
|-------------------------------|-------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
| 8:00-8:10AM, 2023/08/07 (PDT) | Host Chair                                                                                            | Welcome and Open Remarks                                                                                                      |
| 8:10-8:40AM, 2023/08/07 (PDT) | Ed Chi [Google]                                                                                       | Talk 1: LLM Revolution: Implications rom Chatbots and Tool-Use to Reasoning |
| 8:40-9:10AM, 2023/08/07 (PDT) | Tania Bedrax-Weiss [Google]                                                                           | Talk 2: Large-scale AI Model Research at Google Pre-training, Fine-tuning, and Prompt-based Learning                          |
| 9:10-9:25AM, 2023/08/07 (PDT) | Michihiro Yasunaga, Armen Aghajanyan, Weijia Shi, Rich James, Jure Leskovec, Percy Liang, Mike Lewis, Luke Zettlemoyer and Wen-Tau Yih | Paper-1: Retrieval-Augmented Multimodal Language Modeling                                                                    |
| 9:25-9:40AM, 2023/08/07 (PDT) | Silvia Terragni, Modestas Filipavicius, Nghia Khau, Bruna Guedes, André Manso and Roland Mathis      | Paper-2: In-Context Learning User Simulators for Task-Oriented Dialog Systems                                                 |
| 9:40-9:55AM, 2023/08/07 (PDT) | Piotr Kluska, Florian Scheidegger, A. Cristano I. Malossi and Enrique S. Quintana-Ortí                | Paper-3 : Challenges in post-training quantization of Vision Transformers                                                     |
| 9:55-10:10AM, 2023/08/07 (PDT) | Haotian Ju, Dongyue Li, Aneesh Sharma and Hongyang Zhang                                             | Paper-4 : Generalization in Graph Neural Networks: Improved PAC-Bayesian Bounds on Graph Diffusion                            |
| 10:10-10:30AM, 2023/08/07 (PDT) | Coffee Break                                                                                        |                                                                                                                               |
| 10:30-11:00AM, 2023/08/07 (PDT) | Shafiq Joty [Salesforce]                                                                              | Talk 3: NLP Research in the Era of LLMs                                                                                       |
| 11:00-11:30AM, 2023/08/07 (PDT) | YiKang Shen[IBM]                                                                                      | Talk 4: Modular Large Language Model and Principle-Driven alignment with Minimal Human Supervision                            |
| 11:30-11:40AM, 2023/08/07 (PDT) | Hong Sun, Xue Li, Yinchuan Xu, Youkow Homma, Qi Cao, Min Wu, Jian Jiao and Denis Charles             | Paper-5: AutoHint: Automatic Prompt Optimization with Hint Generation                                                         |
| 11:40-11:50AM, 2023/08/07 (PDT) | Zhichao Wang, Mengyu Dai and Keld Lundgaard                                                          | Paper-6: Text-to-Video: a Two-stage Framework for Zero-shot Identity-agnostic Talking-head Generation                         |
| 11:50-12:00PM, 2023/08/07 (PDT) | Long Hoang Dang, Thao Minh Le, Tu Minh Phuong and Truyen Tran                                        | Paper-7: Compositional Prompting with Successive Decomposition for Multimodal Language Models                                 |
| 12:00PM-12:10PM, 2023/08/07 (PDT) | Zhen Guo, Yanwei Wang, Peiqi Wang and Shangdi Yu                                                     | Paper-8: Dr. LLaMA: Improving Small Language Models on PubMedQA via Generative Data Augmentation                              |
| 12:10-12:20PM, 2023/08/07 (PDT) | Haopeng Zhang, Xiao Liu and Jiawei Zhang                                                              | Paper-9 : Extractive Summarization via ChatGPT for Faithful Summary Generation                                                |
| 12:20-12:30PM, 2023/08/07 (PDT) | Closing Remarks                                                                                      |                                                                                                                               |

# Day 3 Schedule - LLM Day

| Time         | Event                                              | Speaker/Details                                                                                                 |
|--------------|----------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| **Date:**    | **Aug. 8, 2023**                                   | **Room:** Grand A                                                                                               |
| 10:00-12:00  | Distinguished Keynotes                             |                                                                                                                  |
| 10:00-10:10  | Opening Remarks                                    | Jie Tang (Tsinghua University)                                                                                   |
| 10:10-11:00  | Jaime Teevan (Microsoft)                           | From Documents to Dialogues: How LLMs are Shaping the Future of Work                                             |
| 11:00-11:50  | Denny Zhou (Google DeepMind)                       | Teach language models to reason                                                                                  |
| 12:00-13:30  | Lunch                                              |                                                                                                                  |
| 13:30-15:30  | Keynotes                                           |                                                                                                                  |
| 13:30-14:10  | Vedanuj Goswami (Meta FAIR)                        | Llama 2: Open Foundation and Fine-Tuned Chat Models                                                              |
| 14:10-14:50  | Peng Zhang (Zhipu AI)                              | From GLM-130B to ChatGLM                                                                                         |
| 14:50-15:30  | Jason Wei (OpenAI)                                 | The large language model renaissance: paradigms and challenges                                                   |
| 15:30-16:00  | Coffee Break                                       |                                                                                                                  |
| 16:00-17:30  | Panel: Paradigm Shifts in the Era of LLMs          | Opportunities and Challenges in Academia, Industry, and Society. Moderator: Qiaozhu Mei (University of Michigan)  |
|              |                                                    | Invited Panelists: Ed Chi (Google DeepMind), Vedanuj Goswami (Meta FAIR), Jaime Teevan (Microsoft), Vy Vo (Intel Labs), Denny Zhou (Google DeepMind) |
