import copy
import torch

g_base_model = ""
d_base_model = ""
num_spk = 1
g = torch.load(g_base_model, map_location="cpu")
d = torch.load(d_base_model, map_location="cpu")

new_embs = torch.zeros((num_spk, 256))
g["model"]["emb_g.weight"] = new_embs

def create_random_model(model):
    for key in model["model"].keys():
        N = model["model"][key].numel()
        bound = 1.0 / N
        model["model"][key] = torch.empty_like(model["model"][key]).uniform_(-bound, bound)
    return model

torch.save(create_random_model(g), f'G_base_{num_spk}spk.pth')
torch.save(create_random_model(d), f'D_base_{num_spk}spk.pth')