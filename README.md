# Predicting Policy Diffusion (2025)
This repository includes the replication data for Hemauer, Saunders, and Desmarais.

Our analysis first replicates the original policy diffusion pooled event history analysis (PEHA) models from Boehmke et al. (2017), Boushey (2016), Bricker and Lacombe (2021), and Karch (2016). This data enables us to compare results across different sample sizes, policy domains, and eras.

After obtaining a baseline, we then turn to using several models to assess the out-of-sample predictive performance of current models of policy diffusion.

## Methodology
We apply and compare pooled event history analysis (PEHA), dyadic event history analysis, network event history analysis (NEHA), random forests, and XGBoost. 

We focus on answering the four following methodological questions:

  1. How far in the future do policy diffusion models predict accurately?
  2. What is the difference between the predicted and observed time-to-adoption?
  3. How do the answers to the above questions vary across policy domains, time, methods, and any other state covariates?
  4. How much data is needed for an efficient and effective model?

## Results
Our results suggest...

Notably, our article provides a benchmark reference for future research on policy diffusion. By assessing predictive accuracy, we provide evidence of the “robustness” of current models used in the study of policy diffusion and provide a deeper empirical understanding of the methodological improvements provided by each model. Rather than assuming NEHA provides the most robust estimates, we provide evidence of the improvements each statistical methodology provides.

