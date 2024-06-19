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
When the webui has started you can load the model.

You can now randomize and change the sliders. To generate a preview select an example file and press generate.

To export a model give it a name and press export. The model will be saved to models folder. This model can then be used with the so-vits-svc-fork.

### Tools
#### extract_components.py
Script used to compute and store all info in the model.

```
python extract_components.py <model_path> <model_config_path> [-n NUM_COMPONENTS] [-r FLAG TO REMOVE ORG EMBEDDIGNS] [-s SAVED_MODEL_NAME]
```

## Model
### V1
The model seems somewhat decent at generating new voices but there is still lots of room for improvement

### Credits
[sbersier](https://github.com/sbersier/pca_svc)
[so-vits-svc-fork](https://github.com/voicepaw/so-vits-svc-fork)