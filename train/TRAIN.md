# How to train a new multispeaker model
this is a guide on how to train a new multispeaker 48k RVC v2 model


### Requirements
 - A working install of the offical RVC project
 - A working python installation with torch

#### Dataset preparation
download or gather dataset the folder structure should be like this:
```
dataset/
├─ speaker_1/
├─ speaker_2/
├─ speaker_n/
```
move preprocess_dataset.py to the rvc folder

change the variables in the preprocess file.

open a terminal in the rvc folder and run following command
```
python preprocess_dataset.py
```

#### Model preparation

change the vars in generate_models.py and run this program. This should save 2 randomized .pth files

#### Changing config
change the num_speakers in the example config,
change the learning rate to something lower like 0.0005
place the config in the logs dir of your experiment

#### Start training
Go to the rvc folder and open a terminal and run the following command
```
"runtime\python.exe" infer/modules/train/train.py -e "exp_dir" -sr 48k -f0 1 -bs 24 -g 0 -te total_epochs -se 1 -pg path_to_generated_g -pd path_to_generated_d -l 0 -c 0 -sw 1 -v v2
```
Training should start now