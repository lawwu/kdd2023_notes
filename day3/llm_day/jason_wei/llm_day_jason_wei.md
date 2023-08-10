# The large language model renaissance: paradigms and challenges - Jason Wei (OpenAI)

Slides are not public yet but these are [slides](https://docs.google.com/presentation/d/1hQF8EXNdePFPpws_jxwqHWi5ohV_TeGL17WIjvUvG6E/edit?resourcekey=0-xA6WdGyYp1EexLgoXgjOjg#slide=id.g16197112905_0_158) from a similar talk Jason gave.

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