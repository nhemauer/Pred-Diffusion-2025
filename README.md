# Predicting Policy Diffusion (2025)
This repository includes the replication data for Hemauer, Saunders, and Desmarais.

Our analysis first replicates the original policy diffusion event history analysis (PEHA) models from Boehmke et al. (2017), Boushey (2016), Bricker & Lacombe (2021), and Karch et al. (2016). This data lets us compare results across different sample sizes, policy domains, and eras.

After obtaining a baseline, we use several models to assess the out-of-sample predictive performance of current models of policy diffusion.

## Methodology
We apply and compare logistic regression, regularized logistic regression, random forests, XGBoost, and SVM algorithms. 

We focus on answering the following four methodological questions in our article:

  1. How far in the future do policy diffusion models predict accurately?
  2. What is the difference between the predicted and observed time-to-adoption?
  3. How do the answers to the above questions vary across policy domains, time, methods, and any other state covariates?
  4. How much data is needed for an efficient and effective model?

## Results
Our results suggest...

Notably, our article provides a benchmark reference and framework for future research on policy diffusion. By assessing predictive accuracy, we provide evidence of the “robustness” of current models used in the study of policy diffusion and provide a deeper empirical understanding of the methodological improvements each model provides. In our article, we discussed and analyzed how well typical models from the literature predict, we quantify the difference between the predicted and observed time-to-adoption, and we answered each question among different policy domains, timeframes, methods, and other state covariates. Finally, and perhaps most importantly, we suggest how much data is needed to produce an efficient and effective model.

## File Structure

The "_ml.py" files are our contribution. These files replicate the accuracy and classification performance of the logistic, random forest, XGBoost, and SVM models found in the article.

The "original_replications.ipynb" produces the original coefficient estimates found in our reference articles.

The "figures" folder includes the AUC-PR Curve and a .txt file with other performance statistics for each model.

The "data" folder includes the data provided by the original authors.

To run our replication scripts, set your working directory to the ROOT of our replication folder. 
