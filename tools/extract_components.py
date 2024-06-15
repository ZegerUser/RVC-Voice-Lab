import torch
import numpy as np
from sklearn.decomposition import PCA
import argparse

NUM_COMPONENTS = 16
REMOVE_ORG_EMB = True

model_name = "base"
model_path = ""
model = torch.load(model_path, map_location="cpu")

org_embs = model["model"]["emb_g.weight"]

pca = PCA(n_components=NUM_COMPONENTS)
reduced = pca.fit_transform(org_embs)

pca_mean = pca.mean_
components = pca.components_
explained_variance = pca.explained_variance_

mu_reduced = np.mean(reduced, axis=0)
sigma_reduced = np.cov(reduced, rowvar=False)

if REMOVE_ORG_EMB:
    model["model"]["emb_g.weight"] = np.zeros((1, 256))

model["project_voice"] = {
    "pca_mean": pca_mean,
    "components": components,
    "explained_var": explained_variance,
    "mu_reduced": mu_reduced,
    "sigma_reduced": sigma_reduced
}

model["name"] = model_name

torch.save(model, model_name + ".pth")