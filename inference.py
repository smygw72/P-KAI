import os
import warnings
import random
import glob
import time
from tqdm import tqdm
from mutagen.mp3 import MP3
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from config.config import get_config
from preprocessing.make_mfcc import spec_to_image, get_melspectrogram_db
from src.network.model import MyModel
from src.metric import mean_scores
from src.utils import set_seed, get_timestamp

# global variables
model = None
file_path = None
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def mutagen_length(path):
    audio = MP3(path)
    length = audio.info.length
    return length

def get_mfcc(i):
    start_segment = i * cfg.data.mfcc_window
    spec = get_melspectrogram_db(
        file_path,
        offset=start_segment,
        duration=cfg.data.mfcc_window
    )
    mfcc_arr = spec_to_image(spec)
    mfcc = torch.from_numpy(mfcc_arr).float()
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(
            (cfg.data.img_size, cfg.data.img_size)
        ),
        transforms.ToTensor(),
    ])
    mfcc = transform(mfcc)
    mfcc = mfcc.unsqueeze(0)
    return mfcc


def inference(n_mfcc):
    mfccs = torch.Tensor()
    for i in range(n_mfcc):
        mfcc = get_mfcc(i)
        mfccs = torch.cat([mfccs, mfcc], dim=0)

    dataset = TensorDataset(mfccs)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.inference.n_frame,
        shuffle=False,
        num_workers=cfg.inference.n_worker,
        pin_memory=True
    )

    scores = torch.Tensor()
    model.eval()
    with torch.no_grad():
        for mfcc in tqdm(dataloader):
            outs = model(mfcc[0].to(device))
            outs = mean_scores(outs)
            if cfg.model.architecture != 'PDR':
                score = outs[0] - outs[3]
                visualize_attention(outs)
            else:
                score = outs[0]
            scores = torch.cat([scores, score.to('cpu')], dim=0)

    return scores.squeeze().tolist()


def main(sound_path=None, learning_log_dir=None, train_or_test='train') -> float:

    global cfg
    cfg = get_config(test_mode=True)
    print(cfg)

    start_time = time.time()
    warnings.filterwarnings('ignore')
    set_seed(cfg.seed)

    # log
    if learning_log_dir is None:
        timestamp = get_timestamp()
        log_dir = f'./inference_logs/{timestamp}'
    else:
        log_dir = learning_log_dir

    # sound
    global file_path
    if sound_path is None:
        file_path = './misc/test.mp3'
    else:
        file_path = sound_path

    file_name = os.path.splitext(os.path.basename(file_path))[0]
    length = mutagen_length(file_path)

    # model
    global model
    model = MyModel(cfg, 'inference').to(device)

    if learning_log_dir is None:
        # when called from aws lambda
        state_dict_path = glob.glob('./model/**/state_dict.pt', recursive=True)[0]
    else:
        # when called from learning.py
        state_dict_path = f'{log_dir}/state_dict.pt'

    checkpoint = torch.load(state_dict_path, map_location=device)
    model.load_state_dict(checkpoint['best_model'])
    print(f'Best epoch    : {checkpoint["best_epoch"]}')
    print(f'Best accuracy : {checkpoint["best_accuracy"]}')

    # main
    n_mfcc = int(length / cfg.data.mfcc_window)
    outputs = inference(n_mfcc)

    # save
    if cfg.inference.save_log is True:
        writer = SummaryWriter(f'{log_dir}/{train_or_test}/{file_name}')
        for i in range(len(outputs)):
            writer.add_scalar("timeline", outputs[i], i)
        writer.close()

    score_avg = sum(outputs) / len(outputs)
    print(f"average score: {score_avg}")

    end_time = time.time()
    sec_per_frame = (end_time - start_time) / n_mfcc
    print(f"elapsed time: {sec_per_frame}")
    return score_avg


def visualize_attention(outs):
    att_good, att_bad = outs[2], outs[5]
    # TODO


if __name__ == "__main__":
    main()
