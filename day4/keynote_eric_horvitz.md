# People and Machines: Pathways to Deeper Human-AI Synergy - Eric Horvitz, Microsoft

- Sparks of Artificial General Intelligence: Early experiments with GPT-4
    - phase transition
    - polymathic (across other fields)
    - Jump in Capabilities
    - Compositionality 
    - Data Visualization
    - Drawing
    - Vision
        - understanding ASCII drawings
    - Theory of Mind 
        - AI Grand challenge (Science Robotics, 2018)
        - Emergence of theory of mind (ToM)
        - Ability to impute mental states of others
            - What does Alice believe?
            - What does Bob think alice think?
        - GPT-4 example - Thanksgiving example
        - Theory of Mind May Have Spontaneously Emerged in Large Language Models (Paper)
    - Professional Compentency Exams
    - Unexpected Capabilities 
        - Medical question: can you change the history or lab results with as minimal changes as possible to make acute kidney injury (F) the best answer. can you shift the labs to make AKI even more likely?
            - Implications for medical education!
        - Why might've the student have selected malacoplakia? 
- Computation Inflection for Scientific Discovery (ACM) [slide]
    - Input the paper as a whole context and being able to ask very detailed questions
- Opportunities for synthesis & integration [slide]
    - LLMs still lack ability to help in these areas below
    - Core substrate of probability & utility
        - Data --> Probability --> Decisions (in any context like a hospital managing readmissions)
    - Insights about Decisions
    - High Stakes Decisions
- Coordinating Human & AI Initiative
    - Human Intellect + Machine Intelligence --> Coordinated Initiative
    - Example: Tesla Autopilot
    - Expected utility to guide AI contributions
        - expected value of taking an action
        - expected value of not taking an action
    - LLMs: Log Probs to guide Initiative? 
        - In
    - Calibration of GPT-4 - for datapoints assigned a probability of 95%, GPT-4 correct 93%
    - Github Copilot
        - Mixed initiative interaction
        - Paper: Reading Between the Lines: Modeling User Behavior and Costs in AI-Assisted Programming
            - Built on ontology of states
                - Thinking/Verifying Suggestion
                - Deferring Thought for Later
        - A significant time is spent on thinking/verifying suggestions. Some of these are rejected. 
        - Fascinating to see the breakdown of time spent in each of these states
    - Continual learning
        - identify an object (orange) and as you move around in a 3D space, you see different aspects of the orange to learn more features of the object
- Leveraging Complementarity of People & Machines
    - Learning & Inference for complementarity 
    - Shaping ML for Complementarity  - identifying metastatic breast cancer
        - Human is superior (3.4% error rate)
        - Human + AI expert (0.5% error rate)
    - Consider Human Expertise in Model Optimization 
        - Jointly learn task x + Value of human input h (kind of like an early days of instruction tuning)
        - Learn a policy for when to ask a human
    - Similar work [slide]
    - Complementarity & Cognition
        - where do people have blind spots, biases, gaps
        - Machine intellect to fill these gaps
        - Memory example:
            - Pull out commitments from emails
            - You may have overlooked these commitments 
            - Could do something similar with prayer requests
    - Extended LLMs for Complementarity
    - Grounding: Achieving Mutual Understanding [slide]
    - Study of Grounding
        - Testing Code Interpreter
        - Paper: Conversational Grounding Acts in Human and Language Model Dialogue (Stanford)
        - Findings [slide]
        - Linguists, Psychologists, AI Researchers, KDD community can work together to make improvements on grounding
- Wish list [slide]
- JCR Licklider quote [slide]

# Q&A

- In the context of LLMs and scientific discovery, how can you ground LLMs in centuries of scientific language?
    - Non-LLM Methods - when to use expensive wet-lab studies vs. inexpensive in-silicon studies (similar for LLM vs. Non-LLM methods). 
    - GPT-4 can be used for drug repurposing 
- Do you think human intellectual abilities will decline given that people can use it to generate essays?
    - We are at an inflection-point that will be recognized in 500 years
    - Next 25-50 years will be recognizable and will be called something
    - You have to wonder what the scribes said when the printing press came out. What is this? How is it going to change our work?
    - I am concerned about education, learning, reflection. 
    - On the topic of essays: the medical board of medical examiners is excited to use this tech, things are changing fast
- ?
    - Use a large scale LLM to probe a small LLM to understand what neurons are being activated
- How do we trust LLMs? In a situation where we don't know and the LLM doesn't know, how do we know we're in this situation?
    - How can these situations be calibrated with probabilities 
    - They can be calibrated with multiple choice questions but in other situations it's not clear
    - In the case where we're not sure if the LLM is correct, use disclosures
    - There are different use cases:
        - Low-stakes: Use them for administrative use cases, doctors doing paper work
        - High-stakes: high-risk surgery
- In human and computer collaboration, do you see the computer being more computer and the human abdicating responsibilities. Some examples: GPS (do not use memory). Do you see a time where there will not be much of a collaboration?
    - Another interesting question about the implication of these technologies of the future of human work and what makes us human
    - There are less medical school graduates going into radiology because of AI (this was 1.5 years ago, this was even before the LLMs). Will LLMs affect people choosing careers?
        - Lawrence: of course
    - Gave a talk at Stanford Medical School last week - will these models be able to see a patient as a whole, mental state, physical state. No, there will always be an incredible a need for humans to touch, human-human care-givers.
    - Let's see what happens in the next 25 years (we should name this period)
- What kind of role do you see national funding agencies? Especially because training an LLM requires resources that is not available to a typical academic.
    -  Recently invited by Senator Warren to speak to 15-20 senators in a closed session, the last question, "What is one thing we should do?" Eric said fund the national AI research resource. Last week Bill Gates went to Congress with the same people. 
    - Just in the last 3-4 months we've seen the power of small models. Use large models to train small models like Orca. He shared these things with Congress.
    - We need to empower our smartest people to supercharge their work and not just limit this to a few companies
- What do you see somethings in 10 years that an AI cannot do?
    - Optimistically, in a world of rising automation, there will be a rise of a human-human, human-touch economy. Understand what it means to be human better. What does he mean by this?
    - Haven't seen genius coming out of these systems. Flashes of insight out of the blue and put it all together. 
    - It's still a mystery to me what will be possible. But what is not a mystery is the human-human interaction that is so important.
- Github copilot: spend more time verifying, type slower, and it interrupts our flow of work. will LLMs impact the way we're using the tools? 
    - Latency is represented in the model as a weight. Suppress recommendations from Copilot because of this latency weight.
    - Github Copilot is a largely bolt on into an existing system. Can we do better with a more insightful design?
    - What does it mean to integrate a powerful into an existing system vs. re-designing the whole system? 
        - One principle is human agency. 
        - Humans should be pilots. 
        - LLMs should be co-pilots.