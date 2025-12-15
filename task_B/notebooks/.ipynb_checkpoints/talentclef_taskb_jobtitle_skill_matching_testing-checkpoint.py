## Imports

"""
- Basics
"""
print ("importing ...")

import os
import sys

"""- Medium"""

import pandas as pd
import numpy as np
import json
import subprocess
import ast

"""- Advanced"""

import networkx as nx

from sentence_transformers import SentenceTransformer, util
from codecarbon import EmissionsTracker

print ("done.")

print ("Path preparation (root, data, models, results, and evaluations)")

root_path="/Users/zia-mac/Documents/projects/data/ir_data/talent_clef/TaskB"

training_data_path = os.path.join(root_path, "training")
validation_data_path=os.path.join(root_path, "validation")
test_data_path=os.path.join(root_path, "test")

models_path=os.path.join(root_path, "models")
results_path=os.path.join(root_path, "results")
evaluations_path=os.path.join(root_path, "evaluations")

sbert_model="all-MiniLM-L6-v2"
sentence_transformer_model=os.path.join(models_path, sbert_model)

print ("done")

print ("Training Data (Preparation and Modeling)")

# Read job2skill file
job2skill = pd.read_csv(os.path.join(training_data_path, 'job2skill.tsv'),
                        sep="\t",
                        names=["job_id","skill_id","rel_type"])
job2skill.head()

jobid2terms_path = os.path.join(training_data_path, "jobid2terms.json")
with open(jobid2terms_path, 'r') as file:
    jobid2terms = json.load(file)


skillid2terms_path = os.path.join(training_data_path, "skillid2terms.json")
with open(skillid2terms_path, 'r') as file:
    skillid2terms = json.load(file)

print (len(job2skill), len(jobid2terms), len(skillid2terms))

print ("Test data (Preparataion and Prediction)")

test_queries_path = os.path.join(test_data_path, "queries")
test_corpus_elements_path = os.path.join(test_data_path, "corpus_elements")

test_queries = pd.read_csv(test_queries_path,sep="\t")
test_corpus_elements = pd.read_csv(test_corpus_elements_path, sep="\t")

print(len(test_queries), len(test_corpus_elements))

print("Transform `skill_aliases` column to a list of strings:")
test_corpus_elements["skill_aliases"] = test_corpus_elements["skill_aliases"].apply(lambda x: ast.literal_eval(x))

print ("Approaches of Skill term expansion")
print ("Approach A (doc_A): Extracting skill related terms")

def related_skill_terms(terms_list):
    return " ".join(terms_list)

test_corpus_elements["doc_A"] = test_corpus_elements["skill_aliases"].apply(lambda x: related_skill_terms(x))

print ("Approach B (doc_B): Extracting terms of jobid associated with the skill")

def get_essential_jobids(skill_id):
    return list(job2skill[(job2skill["skill_id"]==skill_id) & (job2skill["rel_type"]=="essential")]["job_id"])

def related_jobid_terms(skill_id):
    related_essential_jobids = get_essential_jobids(skill_id)
    terms_B = []
    for related_jobid in related_essential_jobids:
        related_jobid_terms = jobid2terms[related_jobid]
        terms_B.extend(related_jobid_terms)
    return " ".join(terms_B)

test_corpus_elements["doc_B"] = test_corpus_elements["esco_uri"].apply(lambda x: related_jobid_terms(x))


print("Approach C (doc_C): Extracting terms of related skill of related jobid")
"""
	 - Skill_id -> {J_1, J_2, ..., J_N} :: J_1-> {S_1, S_2, ..., S_M} :: S_1 -> Terms
"""

def get_essential_skillids(job_id):
    return list(job2skill[(job2skill["job_id"]==job_id) & (job2skill["rel_type"]=="essential")]["skill_id"])

def related_job2skill_terms(skill_id):
    related_essential_jobids = get_essential_jobids(skill_id)

    related_skillids = []
    for related_jobid in related_essential_jobids:
        skill_ids = get_essential_skillids(related_jobid)
        skill_ids.remove(skill_id)
        related_skillids.extend(skill_ids)

    terms_C = []
    for related_skillid in related_skillids:
        related_skillid_terms = skillid2terms[related_skillid]
        terms_C.extend(related_skillid_terms)

    return " ".join(terms_C)

test_corpus_elements["doc_C"] = test_corpus_elements["esco_uri"].apply(lambda x: related_job2skill_terms(x))

print("Load simple sentence transformer model:")
model = SentenceTransformer("all-MiniLM-L6-v2", token=False)

tracker = EmissionsTracker()
tracker.start_task("all-MiniLM-L6-v2")

print (" Generate a mapping dictionary between IDs and texts from query")
test_queries_ids = test_queries.q_id.to_list()
test_queries_texts = test_queries.jobtitle.to_list()
test_queries_map = dict(zip(test_queries_ids, test_queries_texts))

print (" Creating a mapping dictionary of corpus element texts from each of the approaches to corpus_ids")
corpus_ids = test_corpus_elements.c_id.to_list()
corpus_esco_uri = test_corpus_elements.esco_uri.to_list()

print (" Approaches A, B, and C")
#doc_A
corpus_doc_A_texts = test_corpus_elements.doc_A.to_list()
map_corpus_doc_A = dict(zip(corpus_ids, corpus_doc_A_texts))

#doc_B
corpus_doc_B_texts = test_corpus_elements.doc_B.to_list()
map_corpus_doc_B = dict(zip(corpus_ids, corpus_doc_B_texts))

#doc_C
corpus_doc_C_texts = test_corpus_elements.doc_C.to_list()
map_corpus_doc_C = dict(zip(corpus_ids, corpus_doc_C_texts))
print ("done.")

print ("Encode queries and corpus elements:")

query_embeddings = model.encode(test_queries_texts, convert_to_tensor=True)
#corpus_embeddings = model.encode(corpus_doc_A_texts, convert_to_tensor=True)

corpus_doc_A_embedding = model.encode(corpus_doc_A_texts, convert_to_tensor=True)
corpus_doc_B_embedding = model.encode(corpus_doc_B_texts, convert_to_tensor=True)
corpus_doc_C_embedding = model.encode(corpus_doc_C_texts, convert_to_tensor=True)

print ("Compute similarities")
similarities_query_doc_A = util.cos_sim(query_embeddings, corpus_doc_A_embedding).cpu().numpy()

similarities_query_doc_B = util.cos_sim(query_embeddings, corpus_doc_B_embedding).cpu().numpy()

similarities_query_doc_C = util.cos_sim(query_embeddings, corpus_doc_C_embedding).cpu().numpy()

emissions = tracker.stop_task("all-MiniLM-L6-v2")

similarities_query_doc_AB = similarities_query_doc_A + similarities_query_doc_B
similarities_query_doc_AC = similarities_query_doc_A + similarities_query_doc_C
similarities_query_doc_BC = similarities_query_doc_B + similarities_query_doc_C
similarities_query_doc_ABC = similarities_query_doc_A + similarities_query_doc_B + similarities_query_doc_C
similarities_query_doc_WABC = similarities_query_doc_A*0.5 + similarities_query_doc_B*0.3 + similarities_query_doc_C*0.2

print ("Prepare submission file")

import numpy as np

def get_ranked_result_list(similarities_query_doc, corpus_doc):
    """ 
   	The submissions must follow the TREC Run File format, including headers in the output file. This means that the fle have 6 space-spearated columns per line, with following information:
   
   	- q_id: Query ID.
   	- Q0: A constant identifier, usually "Q0".
   	- doc_id: ID of the retrieved document.
   	- rank: Position of the document in the ranking.
   	- score: Relevance score assigned by the model.
   	- tag: Experiment name
    """
    results = []
    results_name = []

    for q_idx, q_id in enumerate(test_queries_ids):
        sorted_indices = np.argsort(-similarities_query_doc[q_idx])
        used_doc_ids = set()
        rank_counter = 0
        for c_idx in sorted_indices:  # Consider the full list.
            doc_id = corpus_ids[c_idx]
            # If doc_id was already processed, go to the next one.
            if doc_id in used_doc_ids:
                continue
            used_doc_ids.add(doc_id)
            rank_counter += 1

            query_name = test_queries_map[q_id]
            doc_name = corpus_doc[c_idx]
            score = similarities_query_doc[q_idx, c_idx]

            results.append(f"{q_id} Q0 {doc_id} {rank_counter} {score:.4f} A_model")
            results_name.append(f"{query_name} Q0 {doc_name} {rank_counter} {score:.4f} A_model")

    return results, results_name

print ("Saving to a file")

results_A, results_A_name = get_ranked_result_list(similarities_query_doc_A, corpus_doc_A_texts)
results_B, results_B_name = get_ranked_result_list(similarities_query_doc_B, corpus_doc_B_texts)
results_C, results_C_name = get_ranked_result_list(similarities_query_doc_C, corpus_doc_C_texts)
results_AB, results_AB_name = get_ranked_result_list(similarities_query_doc_AB, corpus_doc_A_texts)
results_AC, results_AC_name = get_ranked_result_list(similarities_query_doc_AC, corpus_doc_A_texts)
results_BC, results_BC_name = get_ranked_result_list(similarities_query_doc_BC, corpus_doc_B_texts)
results_ABC, results_ABC_name = get_ranked_result_list(similarities_query_doc_ABC, corpus_doc_A_texts)
results_WABC, results_WABC_name = get_ranked_result_list(similarities_query_doc_WABC, corpus_doc_A_texts)

result_approach_A_taskB = os.path.join(results_path, "test_approach_A_taskB.trec")
with open(result_approach_A_taskB, "w", encoding="utf-8") as f:
    f.write("\n".join(results_A))

result_approach_B_taskB = os.path.join(results_path, "test_approach_B_taskB.trec")
with open(result_approach_B_taskB, "w", encoding="utf-8") as f:
    f.write("\n".join(results_B))

result_approach_C_taskB = os.path.join(results_path, "test_approach_C_taskB.trec")
with open(result_approach_C_taskB, "w", encoding="utf-8") as f:
    f.write("\n".join(results_C))

result_approach_AB_taskB = os.path.join(results_path, "test_approach_AB_taskB.trec")
with open(result_approach_AB_taskB, "w", encoding="utf-8") as f:
    f.write("\n".join(results_AB))

result_approach_AC_taskB = os.path.join(results_path, "test_approach_AC_taskB.trec")
with open(result_approach_AC_taskB, "w", encoding="utf-8") as f:
    f.write("\n".join(results_AC))

result_approach_BC_taskB = os.path.join(results_path, "test_approach_BC_taskB.trec")
with open(result_approach_BC_taskB, "w", encoding="utf-8") as f:
    f.write("\n".join(results_BC))

result_approach_ABC_taskB = os.path.join(results_path, "test_approach_ABC_taskB.trec")
with open(result_approach_ABC_taskB, "w", encoding="utf-8") as f:
    f.write("\n".join(results_ABC))

result_approach_WABC_taskB = os.path.join(results_path, "test_approach_WABC_taskB.trec")
with open(result_approach_WABC_taskB, "w", encoding="utf-8") as f:
    f.write("\n".join(results_WABC))

emissions_path = os.path.join(tests_path, "test_emissions.json")
json.dump(dict(emissions.values), open(emissions_path, "w"), ensure_ascii=False, indent=4)
print ("done")
