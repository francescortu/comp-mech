import os
import sys
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))

# Add the parent directory (..) to sys.path
sys.path.append(os.path.join(script_dir, ".."))

# Optionally, add the 'src' directory directly
sys.path.append(os.path.join(script_dir, "..", "src"))


import json

model_name = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "pythia-6.9b"]
similarity_type = ["word2vec", "logit"]



for similarity in similarity_type:
    print("Similarity", similarity)
    for model in model_name:
        print("Model", model)
        data = json.load(
            open(
                f"../data/tmp/full_data_sampled_{model}_similarity_{similarity}_10000.json",
                "r",
            )
        )
        sim_1, sim_2, sim_3, sim_4 = 0, 0, 0, 0
        position_1 = []
        for i, d in enumerate(data):
            if d["target_new"][1:] in d["similar_tokens_1"] or d["target_new"] in d["similar_tokens_1"]:
                position_1.append(i)
                sim_1 += 1
            if d["target_new"][1:] in d["similar_tokens_2"] or d["target_new"] in d["similar_tokens_2"]:
                sim_2 += 1
            if d["target_new"][1:] in d["similar_tokens_3"] or d["target_new"] in d["similar_tokens_3"]:
                sim_3 += 1
            if d["target_new"][1:] in d["similar_tokens_4"] or d["target_new"] in d["similar_tokens_4"]:
                sim_4 += 1


        print(sim_1, sim_2, sim_3, sim_4, "total find", sim_1+sim_2+sim_3+sim_4, sim_true)
#compute the mean of position_ 1

