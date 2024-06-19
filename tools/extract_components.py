import torch
import numpy as np
from sklearn.decomposition import PCA
import argparse
import json

parser = argparse.ArgumentParser(description="Extract And Compute Embeddings")

parser.add_argument("model_path", type=str, help="path to the model file")
parser.add_argument("model_config_path", type=str, default="config.json", help="path to the model config file")

parser.add_argument("-n", "--num_components", type=int, default=16, help="number of PCA components")
parser.add_argument("-r", "--remove_org_emb", action="store_true", help="remove original embeddings")
parser.add_argument("-s", "--saved_model_name", type=str, default="base", help="path to the model file")

args = parser.parse_args()

model_path = args.model_path
model_config = args.model_config_path

num_components = args.num_components
saved_model_name = args.saved_model_name


with open(model_config, "r") as f:
    config = json.load(f)

model = torch.load(model_path, map_location="cpu")

org_embs = model["model"]["emb_g.weight"]

pca = PCA(n_components=num_components)
reduced = pca.fit_transform(org_embs)

pca_mean = pca.mean_
components = pca.components_
explained_variance = pca.explained_variance_

mu_reduced = np.mean(reduced, axis=0)
sigma_reduced = np.cov(reduced, rowvar=False)

if args.remove_org_emb:
    model["model"]["emb_g.weight"] = np.zeros((1, 256))
    config["spk"] = {"new_spk": 0}

else:
    new_idx = len(org_embs) + 1
    model["model"]["emb_g.weight"] = np.zeros((new_idx, 256))

    model["model"]["emb_g.weight"][:-1] = org_embs 
    config["spk"]["new_spk"] = new_idx - 1 
    config["model"]["n_speakers"] = new_idx  

model["voice_lab"] = {
    "pca_mean": pca_mean,
    "components": components,
    "explained_var": explained_variance,
    "mu_reduced": mu_reduced,
    "sigma_reduced": sigma_reduced
}

model["name"] = saved_model_name

with open(saved_model_name + ".json", "w") as f:
    json.dump(config, f, indent=4)
torch.save(model, saved_model_name + ".pth")