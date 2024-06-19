# So Vits SVC Voice Lab
A minimal application to generate new and unique voice to use with so-vits-svc.

## Requirements
A working installation of the so-vits-svc-fork.

## Usage
### Webui
Download the base model and config from [here](https://huggingface.co/Zeger56644/voice-lab-v1) and place it in the models/base folder.

to start the webui:
```
python webui.py
```
When the webui has started you can load the model. If didn't have the embeddings removed you can load these as examples.

To export a model give it a name and press export. The model will be saved to models folder. This model can then be used with the so-vits-svc-fork.


### Tools
#### extract_components.py
Script used to compute and store all info in the model.

```
python script_name.py <model_path> <model_config_path> [-n NUM_COMPONENTS] [-r] [-s SAVED_MODEL_NAME]
```

### Credits
[sbersier](https://github.com/sbersier/pca_svc): Original experiments
[so-vits-svc-fork](https://github.com/voicepaw/so-vits-svc-fork)