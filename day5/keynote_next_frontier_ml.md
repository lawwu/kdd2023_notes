# Keynote: Time: The Next Frontier in Machine Learning - Mihaela van der Schaar

 
- Time: the next-frontier in machine learning
- Reality-centric AI [slide]
- Time is a fundamental dimension of reality
    - aims to solve real-world problems
    - accounts for complexity of world
    - empowers and does not marginalize humans
- Time, this talk - four canonical problems [slide]
- Time requires new ways of thinking for machine learners

## Timely prediction (Early Warning)

- Goal: foreseeing potential issues or risks in a system
- Proactive
- Example: healthcare
    - MIMIC dataset
    - multiple streams of measurements
    - sparse, irregular and informatively/actively sampled - not randomly sampled but more likely to be sampled when something is wrong (patient is sick)
    - many possible temporal patterns
- ML: Numerous Dynamic and Recurrent Models
    - RNN/LSTM/GRU/CNN
    - Transformers
- Are these methods useful for early warning?
    - No, not a classification or regression problem
    - Time-series forecasting is about predicting the future values of a continuous sequence
    - OTOH, Early warning is about predicting the time until specific events occur
- Vital role of time-to-event analysis in different sectors [slide]
- Early warning: Dynamic time-to-event (survival) analysis
    - We need to compute time-to-event dynamically as time-series data becomes available [slide] - notice the red and green lines are different measurements over time
- Data in longitudinal time-to-event ("survival") analysis [slide]
    - Time-to-event including right-censoring - such patients/data points are informative and we shouldn't discard them (they may hae passed away?)
- Objective?
    - use cumulative incidence functions
    - Dynamic-DeepHIT [Lee & vdS, TBME 2019]
- Dynamic-DeepHIT [Lee & vdS, TBME 2019]
    - Competing risks; multi-task architecture 
    - Shared subnetwork vs. Cross network
    - Loss Functions [slide]
        - total loss = log-likelihood of joint TTE distribution + ranking loss + step-ahead prediction
        - ranking: penalize the incorrect ordering of loss
        - step-ahead prediction: penalize the incorrect prediction of the time-to-event
    - Application: Developing ML algos for dynamic estimation of progression during active surveillance for prostate cancer
    - YouTube video - Revolutionizing Healthcare
    - Applications in HR: 
        - attrition models (dynamically estimate the time until an employee leaves the company)
- Early warning - many interesting & impactful challenges [slide]

## Detection (Early Diagnosis)

- Goal: detect **existing** problems at teh earliest possible time
- Early diagnosis focuses on identifying an issue that has already emerged but remains hidden or subtle
- Early diagnosis & detection for cancer, survival rates changes significiantly 
- Very hard problem
    - Orange is when Stage 2 diagnosed
    - Red is when surgery happened
    - Sparse & diverse observations
- How can we model early diagnosis problem from sparse indicators?
    - Indicators (signs/symptoms) modulate the likelihood of future indicators, e.g. system evolution
- Learning temporal relationship between sparse indicators [slide]
- Paper: Deep Diffusion Processes (DDP)
    - Creating a graph of the indicators at different time t
    - Each indicator is an edge
    - G(t) = (V, L(t), E(t), W(t))
    - How can we learn the dynamic indicator network?
- Intensity function depends on two terms: exogenous factors and past indicators
- Early diagnosis - many interesting & impactful challenges [slide]

## Understanding & Action (Causality over Time)

- Goal: understanding the causal mechanisms that drive the system
- A Path for real-world impact with casuality
    - Causality has great potential but limited adoption in the real-world
    - vDS developed Causal Deep Learning
- Causal Deep Learning paper, March 2023 <https://arxiv.org/abs/2303.02186>
- Pearl's Ladder of Causation vs. Our Causal Deep Learning Map
    - Adds a dimension of time
- Most current causal methods discard temporal features even though causality is inherently temporal based
- One key focus of our lab: CATE/ITE
    - Goal: Choose targeted (individualized) and effective actions and interventions over time
    - ITE over time
        - How to treat?
        - When to treat?
        - When to stop treatment?
- Applications
    - Finance, e.g. personalized investment strategies
    - Smart cities, e.g. adaptive energy consumption
- CATE learning from longtiudinal patient observation data [slide]
    - Why does she categorize "patient features" as static? doesn't this vary over time
- Handling time-dependent confounding bias
    - Marginal structural models paper
    - Rec
- Counterfactual Recurrent Network [Bica, Alaa, Jordan & van der Schaar, ICLR 2020]
    - Estimates counterfactural trajectories using seq-to-seq architecture
- Treatment effects over time
    - How to handle learning from informative sampling?
    - Accounting For Informative Sampling When Learning to Forecast Treatment Outcomes Over Time [vDS, ICML 2023]

## Dynamical System Comprehension (Discovery & Understanding of Complex Dynamical Systems)

- This is the Ultimate Frontier, encompasses the previous 3 and goes further
- Goal: Discovery of dynamical system from observational data
- Unique challenges [slide]
- Discover closed-form ordinary differential equations (ODEs) from observed trajectories - D-CODE [ICLR 2022]
    - https://openreview.net/forum?id=wENMvIsxNN
    - This paper introduces a new technique for discovering closed-form functional forms (ordinary differential equations) that explain noisy observed trajectories x(t) where the "label" x'(t) = f(x(t), t) is not observed, but without trying to approximate it. The method first tries to approximate a smoother trajector x^hat(t), then relies on a variational formulation using a loss function over functionals {C_j}_j, defined in terms of an orthonormal basis {g_1, …, g_S} of sampling functions such that the sum of squares of all the C_j approximates the theoretical distance between f(x) and the solution f*(x). These sampling functions are typically chosen to be a basis of sine functions. The method is evaluated on several canonical ODEs (growth model, glycolitic oscillator, Lorenz chaotic attractor) and compared to gaussian processes-based differentiation, to spline-based differentiation, regularised differentiation, and applied to model the temporal effect of chemotherapy on tumor volume. 
    - Reviewers found that the paper was well-motivated and easy to follow (EBvJ), well evaluated (EBvJ), offering new perspectives to symbolic regression (79Ft). Reviewer vaG3 had their concerns addressed. Reviewer ZddY had concerns about the running time (a misunderstanding that was clarified) and the lack of comparison to a simple baseline consisting in double optimisation over f and x^hat(0) using Neural ODEs (the authors have added a Neural ODE baseline but were in disagreement with ZddY and 79Ft about their limitations).
    - Reviewers engaged in a discussion with the authors, and the scores are 6, 6, 8, 8. I believe that the paper definitely meets the conference acceptance bar and would advocate for its inclusion as a spotlight in the conference.
- D-CODE 
    - <https://github.com/ZhaozhiQIAN/D-CODE-ICLR-2022>
    - in action: discover temporal effects of chemotherapy on tumor volume

- They do online engagement sessions with the ML research community every month on YouTube

## Q&A

- For D-CODE, did you look into physics inspired models?
    - We wanted to have interpretable models, physics models are more black boxy. 
- Have you thought about privacy and ethics esp. in healthcare with patient data?
    - We are the first group to generate time-series synthetic data
    - <https://github.com/vanderschaarlab/synthcity>
    - Will have a workshop at NeurIPS on generative synthetic data
    - Ran tutorials on synthetic data - ICML 2021, AAAI 2023
- How do you think about image analyses for tumor detection?
    - I think this is powerful work
    - Imaging may be invasive itself because of the radiation
    - Can we push the frontier even earlier?
- Question about the definition of causality, how does it in relate to granger causality?
    - https://en.wikipedia.org/wiki/Granger_causality - The Granger causality test is a statistical hypothesis test for determining whether one time series is useful in forecasting another, first proposed in 1969.[1] Ordinarily, regressions reflect "mere" correlations, but Clive Granger argued that causality in economics could be tested for by measuring the ability to predict the future values of a time series using prior values of another time series. Since the question of "true causality" is deeply philosophical, and because of the post hoc ergo propter hoc fallacy of assuming that one thing preceding another can be used as a proof of causation, econometricians assert that the Granger test finds only "predictive causality".[2] Using the term "causality" alone is a misnomer, as Granger-causality is better described as "precedence",[3] or, as Granger himself later claimed in 1977, "temporally related".[4] Rather than testing whether X causes Y, the Granger causality tests whether X forecasts Y.[5]
    - We want to go further than granger causality
    - Right now we assume no causality or 100% causality 
    - Let's use plausible causal models to empower us to think about causal ideas in a pragmatic way to solve problems
    - I don't have one definition, goal is not just to publish papers but to solve real-world problems
- How do you think about evaluation with time as another dimension?
    - Do we have the right metrics/metric to evaluate performance?
    - We have time-to-event static metrics but moving them to dynamic was non-trivial.

## Links

- <https://www.vanderschaar-lab.com/the-case-for-reality-centric-ai/> - It is little noticed that two camps have emerged in AI and machine learning (ML). One, which we call Petri-dish AI and which is exemplified by clean, simple-to-define yet challenging-to-solve problems like playing games or making biological or chemical  discoveries. The other camp, in which the van der Schaar Lab is a leader, and which we call Reality-centric AI, puts the inherent and unavoidable complexity of the real world at the heart of designing, training, testing, and deploying models.
- <https://github.com/vanderschaarlab>
