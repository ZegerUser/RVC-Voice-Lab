# How to Train a New Multispeaker Model (RVC V2)

### Requirements

 - A working installation of the official RVC project (Nvidia only)
 - A working Python installation with Torch

### Dataset Preparation

Download or gather your dataset. The folder structure should be like this:
```
dataset/
├─ speaker_1/
├─ speaker_2/
├─ speaker_n/
```
Move preprocess_dataset.py to the RVC folder.

Change the variables in the preprocess file.

Open a terminal in the RVC folder and run the following command:
```
python preprocess_dataset.py
```
### Model Preparation

Change the variables in generate_models.py and run this program. This should save two randomized .pth files.

### Changing Config

Change the num_speakers in the example config file. Place the config in the logs directory of your experiment.

### Starting Training

Go to the RVC folder, open a terminal, and run the following command:
```bash
"runtime\python.exe" infer/modules/train/train.py -e experiment_dir -sr 48k -f0 1 -bs batchsize -g 0 -te total_epochs -se 1 -pg path_to_generated_g -pd path_to_generated_d -l 0 -c 0 -sw 1 -v v2
```
A batch size of 32 fits in 24GB of VRAM.
Training should start now.

### Monitoring Training
You can monitor training like every other rvc model

When I trained the model, the first 10 epochs had very high gradient norms: around 1600 for the generator and 500 for the discriminator. After 10 epochs, it dropped down to 400 and 60.

Don't look at the total loss (g/total) since it is not an indication of the model's performance.
FM loss should ideally increase over time.
KL loss and Mel loss take turns: when Mel loss goes down, KL loss goes up. Both should eventually converge over time to a lower state.