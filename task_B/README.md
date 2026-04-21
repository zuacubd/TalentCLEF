# TalentCLEF
In this project, we develop an Information retrieval (IR) system for our participation to the TalentCLEF2025 Task B: Job Title-Based Skill Prediction. After the competition, we further improve the performance of our system and conduct deep analysis with the train, validation, and test dataset.

## Goal: 
  - **Objective**: Developing systems capable of retrieving relevant skills for a given job title.

## Data: 
  - **Training set**: A training set of at least **5.000 job titles** along with the professional skills required for each position is provided. This data is sourced from __actual job descriptions__ and semi-automatically curated to ensure high accuracy in the training set.

  - **Development set**: The development set consists of **200 job titles** along with their related __skills__, normalized to __ESCO__ terminology.

  - **Test set**: The test set comprises a list of **500 job titles**. The task is to **predict** the related __skills__ using the provided gazetteer.

## Evaluation: 
  - The model performance is evaluated with IR metrics, being the Mean Average Precision (MAP) the official metric of the task, as well as other metrics such as Mean Reciprocal Rank (MRR) and Precision@K(1,5,10).

## Ideas
  - I1: Match the query with the documents using Dense embedding representation
  - I2: Match the query with the documents using Sparse VoWs representation (TF-IDF, BM25)
  - I3: Match the query with the documents using Dense and Sparse representation (Hybrid approach)
  - I4: JobBERT
  - I5: Prompting
