# kdd2023_notes

I attended [KDD 2023](https://kdd.org/kdd2023/) which was held in Long Beach, CA from Aug 6-10. Here are some notes from the conference, mainly for my sake.

- Day 2
    - [Notes](./day2/day2_all.md)
- Day 3 
    - [The LLM (Large Language Model) Revolution: Implications from Chatbots and Tool-use to Reasoning - Ed Chi](./day3/ed_chi_keynote/ed_chi_keynote.md) and some of the [slides](./day3/ed_chi_keynote/slides/)
    - LLM Day Notes: Notes from Large Language Model Day with sessions spanning from 10am-5:30pm on 2023-08-08. <https://bigmodel.ai/llmday-kdd23/>
        - [From Documents to Dialogues: How LLMs are Shaping the Future of Work - Jaime Teevan (Microsoft)](./day3/llm_day/jamie_teevan/llm_day_jaime_teevan.md)
        - [Teach language models to reason (Denny Zhou (Google DeepMind))](./day3/llm_day/denny_zhou/llm_day_denny_zhou.md)
        - [Llama 2: Open Foundation and Fine-Tuned Chat Models - Vedanuj Goswami (Meta FAIR)](./day3/llm_day/vedanuj/llm_day_vedanuj.md)
        - [From GLM-130B to ChatGLM - Peng Zhang (Zhipu AI)](./day3/llm_day/peng_zheng/llm_day_peng_zheng.md)
        - [The large language model renaissance: paradigms and challenges - Jason Wei (OpenAI)](./day3/llm_day/jason_wei/llm_day_jason_wei.md)
        - [Panel - Paradigm Shifts in the Era of LLMs: Opportunities and Challenges in Academia, Industry, and Society](./day3/llm_day/panel/llm_day_panel.md)
- Day 4
    - [People and Machines: Pathways to Deeper Human-AI Synergy - Eric Horvitz, Microsoft](./day4/keynote_eric_horvitz.md)
    - [Pretrained Language Representations for Text Understanding: A Weakly-Supervised Perspective](./day4/pretrained_llm_nlu_workshop/notes.md) - though the slides below are much better. There was an intro and Parts 1-3 were presented. Part 4 was rushed through. Part 5 was not presented at KDD.
        - [Webpage](https://yumeng5.github.io/kdd23-tutorial/)
        - Introduction ([Slides](./day4/pretrained_llm_nlu_workshop/slides/Part0.pdf))
        - Part I: Language Foundation Models ([Slides](./day4/pretrained_llm_nlu_workshop/slides/Part1.pdf))
        - Part II: Embedding-Driven Topic Discovery ([Slides](./day4/pretrained_llm_nlu_workshop/slides/Part2.pdf))
        - Part III: Weakly-Supervised Text Classification ([Slides](./day4/pretrained_llm_nlu_workshop/slides/Part3.pdf))
        - Part IV: Language Models for Knowledge Base Construction ([Slides](./day4/pretrained_llm_nlu_workshop/slides/Part4.pdf))
        - Part V: Advanced Text Mining Applications ([Slides](./day4/pretrained_llm_nlu_workshop/slides/Part5.pdf))
        - https://github.com/mickeysjm/awesome-taxonomy - a previous student from their research group maintains this repo
    - [LLMs in Education: Opportunities and Challenges](./day4/panel_llms_in_education.md)
- Day 5
    - [Keynote: The Next Frontier in Machine Learning - Mihaela van der Schaar](./day5/keynote_next_frontier_ml.md)
    - [Robust ML Papers](./day5/robust_ml.md)


# Schedules

## Day 2 Schedule

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

## Day 3 Schedule - LLM Day

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
