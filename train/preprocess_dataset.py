import os.path
import subprocess
import os

procs = 14

# Path to the dataset
dataset_dir = ""
# Path to the exp in the logs folder (empty folder)
exp_dir = ""
# Path where the final preprocessed dataset will be save
final_dir = ""
# Path to the mute dir
mute_dir = ""

def preprocess_and_extract(save_dir, spk_dir):
    subprocess.run(['runtime/python.exe', 'D:/rvc/infer/modules/train/preprocess.py', spk_dir, '48000', procs, save_dir, 'False', '3.0'])
    subprocess.run(['runtime/python.exe','infer/modules/train/extract/extract_f0_rmvpe.py', '1', '0', '0', save_dir, 'True'])
    subprocess.run(['runtime/python.exe','infer/modules/train/extract/extract_f0_rmvpe.py', '2', '1', '0', save_dir, 'True'])
    subprocess.run(['runtime/python.exe','infer/modules/train/extract_feature_print.py', 'cuda:0', '1', '0', '0', save_dir, 'v2'])


speakers = os.listdir(dataset_dir)
for i, speaker in enumerate(speakers):
    spk_exp_dir = os.path.join(final_dir, speaker)
    spk_dir = os.path.join(dataset_dir, speaker)

    os.makedirs(spk_exp_dir, exist_ok=True)
    preprocess_and_extract(spk_exp_dir, spk_dir)

    with open(os.path.join(exp_dir, "filelist.txt"), "w") as f:
        for file in os.listdir(os.path.join(spk_exp_dir, "0_gt_wavs")):
            filename, ext = os.path.splitext(file)

            gt_wavs = os.path.join(spk_exp_dir, "0_gt_wavs", f"{filename}.wav")
            feature768 = os.path.join(spk_exp_dir, "3_feature768", f"{filename}.npy")
            a_f0 = os.path.join(spk_exp_dir, "2a_f0", f"{filename}.wav.npy") 
            b_f0nsf = os.path.join(spk_exp_dir, "2b-f0nsf", f"{filename}.wav.npy")
            f.write(f"{gt_wavs}|{feature768}|{a_f0}|{b_f0nsf}|{i}\n")
        
        mute_gt = os.path.join(mute_dir, "0_gt_wavs/mute48k.wav")
        mute_3_768 = os.path.join(mute_dir, "3_feature768/mute.npy")
        mute_2a_f0 = os.path.join(mute_dir, "2a_f0/mute.wav.npy")
        mute_2b_f0nsf = os.path.join(mute_dir, "2b-f0nsf/mute.wav.npy")
        for _ in range(4):
             f.write(f"{mute_gt}|{mute_3_768}|{mute_2a_f0}|{mute_2b_f0nsf}|{i}\n")