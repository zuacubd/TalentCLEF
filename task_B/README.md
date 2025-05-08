# TalentCLEF
This project develops the information retrieval (IR) system for our participation to the TalentCLEF2025- Task B:  Job Title-Based Skill Prediction.

## Goal: 
  - Developing systems capable of retrieving relevant skills associated with a given job title.

## Data: 
  - **Training set**: A training set of at least **5.000 job titles** along with the professional skills required for each position will be provided. This data is sourced from __actual job descriptions__ and semi-automatically curated to ensure high accuracy in the training set.

  - **Development set**: The development set will consist of **200 job titles** along with their related __skills__, normalized to __ESCO__ terminology.

  - **Test set**: The test set comprises a list of **500 job titles**. The participants will be required to **predict** the related __skills__ using the provided gazetteer.

## Evaluation: 
  - The model performance in this task will be evaluated with information retrieval metrics, being the Mean Average Precision (MAP) the official metric of the task, as well as other metrics such as Mean Reciprocal Rank (MRR) and Precision@K(1,5,10).
