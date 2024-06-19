import gradio as gr
import random
import os
import torch
import numpy as np
from sklearn.decomposition import PCA
import copy
from datetime import datetime
import librosa
import uuid
import json

from so_vits_svc_fork.inference.main import infer
from so_vits_svc_fork.utils import get_optimal_device

NUM_COMPONENTS = 32

models_path =os.path.join(os.getcwd(), "models/base")
models =  [os.path.join(models_path, f) for f in os.listdir(models_path) if f.endswith('.pth')]
config_paths = [os.path.join(models_path, f) for f in os.listdir(models_path) if f.endswith('.json')]
configs = [] 
for path in config_paths:
    with open(path, "r") as f:
        configs.append(json.load(f))

component_samples = []
audio_path = os.path.join(os.getcwd(), "samples/audio")
audio_samples = [os.path.join(audio_path, f) for f in os.listdir(audio_path)]

loaded_model = None
loaded_index = None
KEYS =  ["pca_mean", "components", "explained_var", "mu_reduced", "sigma_reduced"]
def load_model(index):
    global loaded_model, loaded_index
    loaded_model = torch.load(models[index], map_location="cpu")
    loaded_index = index

    if "voice_lab" not in loaded_model.keys():
        return "Model not supported"
    
    if not all(key in loaded_model["voice_lab"] for key in KEYS):
        return "Model has not all keys"
    
    sample_names = list(configs[index]["spk"].keys())
    sample_names.remove("new_spk")

    return "Model loaded succesful", gr.Dropdown.update(choices=sample_names)

def randomize_sliders():
    if loaded_model == None:
        return "No Model Selected", *[0 for _ in range(NUM_COMPONENTS)]

    return "Randomized succesful", *np.random.multivariate_normal(loaded_model["voice_lab"]["mu_reduced"], loaded_model["voice_lab"]["sigma_reduced"])

def sliders_to_emb(scaling, *sliders):
    global loaded_model
    pca = PCA(n_components=loaded_model["voice_lab"]["components"].shape[0])
    pca.mean_ = loaded_model["voice_lab"]["pca_mean"]
    pca.components_ = loaded_model["voice_lab"]["components"]
    pca.explained_variance_ = loaded_model["voice_lab"]["explained_var"]


    scaled =  np.asarray(sliders[0]) * scaling

    return pca.inverse_transform(sliders) 

def generate_preview(sample_index, scaling, *sliders):
    global loaded_model, loaded_index
    if sample_index == None or loaded_model == None:
        return "No sample selected or no model loaded", None
    
    new_emb = sliders_to_emb(scaling, sliders)
    new_model = copy.deepcopy(loaded_model)
    new_model["model"]["emb_g.weight"] = torch.tensor(new_emb)

    tmp_model_path = os.path.join(os.getcwd(), "models", "temp", f"{int(uuid.uuid4())}.pth")
    torch.save(new_model, tmp_model_path)

    
    #audio_samples[sample_index]
    audio_ouput_name = os.path.join(os.getcwd(), "samples", "generated", f"{int(uuid.uuid4())}.wav")

    config_path = config_paths[loaded_index]

    infer(
        # paths
        input_path=audio_samples[sample_index],
        output_path=audio_ouput_name,
        model_path=tmp_model_path,
        config_path=config_path,
        recursive=False,
        # svc config
        speaker="new_spk",
        cluster_model_path=None,
        transpose=0,
        auto_predict_f0=True,
        cluster_infer_ratio=0,
        noise_scale=0.4,
        f0_method="dio",
        # slice config
        db_thresh=-40,
        pad_seconds=0.5,
        chunk_seconds=0.5,
        absolute_thresh=False,
        max_chunk_seconds=40,
        device=get_optimal_device(),
    )
    os.remove(tmp_model_path)
    audio, sr = librosa.load(audio_ouput_name)
    return "Generated preview", (sr, audio)

def export_model(model_name, scaling, *sliders):
    global loaded_model
    if model_name == None or model_name.strip() == "" or loaded_model == None:
        return "No model name or no base model selected"
    
    new_emb = sliders_to_emb(scaling, sliders)
    new_model = copy.deepcopy(loaded_model)
    new_model["model"]["emb_g.weight"] = torch.tensor(new_emb)
    
    new_model["name"] = model_name

    unique_id = str(uuid.uuid4())[:8]
    model_path = os.path.join(os.getcwd(), "models", f"{model_name}_{loaded_model['name']}_{unique_id}.pth")
    config_new_path = os.path.join(os.getcwd(), "models", f"{model_name}_{loaded_model['name']}_{unique_id}.json")
    with open(os.path.join(os.getcwd(), "models/base/config.json"), "r") as f:
        config = json.load(f)
    
    config["spk"] = {model_name: 0}

    with open(config_new_path, "w") as f:
        json.dump(config, f, indent=4)

    torch.save(new_model, model_path)
    return f"Saved model to: {model_path}"

def load_example(example_name):
    global loaded_model, loaded_index
    if loaded_model == None:
        return "No base model selected", *[0 for _ in range(NUM_COMPONENTS)]
    
    pca = PCA(n_components=loaded_model["voice_lab"]["components"].shape[0])
    pca.mean_ = loaded_model["voice_lab"]["pca_mean"]
    pca.components_ = loaded_model["voice_lab"]["components"]
    pca.explained_variance_ = loaded_model["voice_lab"]["explained_var"]

    return "Loaded Example", *pca.transform([loaded_model["model"]["emb_g.weight"][configs[loaded_index]["spk"][example_name]]])[0]

with gr.Blocks(title="Voice Lab") as demo:
    gr.Markdown("# SO-VITS-SVC Voice Lab")

    with gr.Row():
        model_dropdown = gr.Dropdown(label="Load Model", choices=models, interactive=True, type="index", value=0)

    with gr.Row():
        example_dropdown = gr.Dropdown(label="Load Example Components", choices=[], interactive=True)
        randomize_button = gr.Button("Randomize")

    component_sliders = []
    for i in range(8): 
        with gr.Row():
            for j in range(4):
                component_sliders.append(gr.Slider(minimum=-5, maximum=5, step=0.01, value=0, label=f"Component {i*4 + j + 1}", interactive=True))

    with gr.Row():
        scaling = gr.Slider(minimum=0, maximum=10, step=0.01, value=3, label="Scaling Factor", interactive=True)

    with gr.Row():
        audio_example_selector = gr.Dropdown(label="Sample audio's", choices=audio_samples, interactive=True, type="index")
        preview_button = gr.Button("Generate Preview Audio")

    with gr.Row():
        output_audio = gr.Audio(label="Preview Audio", interactive=False)

    with gr.Row():
        export_name = gr.Textbox(max_lines=1, label="Speaker Name")
        export_button = gr.Button("Export Model")
    
    with gr.Row():
        info_text_box = gr.Textbox(label="Info", interactive=False)

    model_dropdown.change(load_model, inputs=model_dropdown, outputs=[info_text_box, example_dropdown])

    randomize_button.click(randomize_sliders, outputs=[info_text_box] + component_sliders)

    preview_button.click(generate_preview, inputs=[audio_example_selector, scaling] + component_sliders, outputs=[info_text_box, output_audio])
    
    export_button.click(export_model, inputs=[export_name, scaling] + component_sliders, outputs=info_text_box)

    example_dropdown.change(load_example, inputs=example_dropdown, outputs=[info_text_box] + component_sliders)

demo.launch()