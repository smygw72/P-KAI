import os
import warnings
import random
from multiprocessing import cpu_count  # Pool is restricted
import time
import numpy as np
from tqdm import tqdm
from mutagen.mp3 import MP3
import torch
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from config.config import CONFIG
from preprocessing.make_mfcc import spec_to_image, get_melspectrogram_db
from src.pool import Pool
from src.network.interface import get_model
from src.metric import mean_scores
from src.utils import set_seed

# global variables
model = None
sound_path = None
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def mutagen_length(path):
    audio = MP3(path)
    length = audio.info.length
    return length


def get_mfcc(i):
    start_segment = i * CONFIG.data.mfcc_window
    spec = get_melspectrogram_db(
        sound_path,
        offset=start_segment,
        duration=CONFIG.data.mfcc_window
    )
    mfcc = spec_to_image(spec)
    mfcc = torch.from_numpy(mfcc).float()
    mfcc = mfcc.unsqueeze(0).unsqueeze(0)
    transform = transforms.Compose([
        transforms.Resize(
            (CONFIG.data.img_size, CONFIG.data.img_size)),
    ])
    mfcc = transform(mfcc)
    return mfcc


def cpu_inference(i):
    mfcc = get_mfcc(i)
    with torch.no_grad():
        outs = model(mfccs).detach()
        outs = mean_scores(outs)
        if CONFIG.model.architecture != 'PDR':
            score = outs[0] - outs[3]
            visualize_attention(outs)
        else:
            score = outs[0]

    return score


def gpu_inference(n_mfcc):
    mfccs = torch.Tensor()
    for i in tqdm(range(n_mfcc)):
        mfcc = get_mfcc(i)
        mfccs = torch.cat([mfccs, mfcc], dim=0)

    dataset = torch.utils.TensorDataset(mfccs)
    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG.inference.batch_size,
        shuffle=False,
        num_workers=CONFIG.inference.n_worker,
        pin_memory=True
    )

    scores = torch.Tensor()
    with torch.no_grad():
        for mfcc in dataloader:
            outs = model(mfccs.to(device)).detach().to('cpu')
            outs = mean_scores(outs)
            if CONFIG.model.architecture != 'PDR':
                score = outs[0] - outs[3]
                visualize_attention(outs)
            else:
                score = outs[0]
            scores = torch.cat([scores, score], dim=0)

    return outputs.squeeze().tolist()


def main(path=None) -> float:

    start_time = time.time()
    warnings.filterwarnings('ignore')
    set_seed(CONFIG.seed)

    global sound_path
    if path is None:
        sound_path = './misc/test.mp3'
    else:
        sound_path = path

    file_name = os.path.splitext(os.path.basename(sound_path))[0]
    length = mutagen_length(sound_path)

    global model
    model = get_model().to(device)
    model_path = f'./model/model.pth'
    model.load_state_dict(torch.load(model_path))
    model.eval()

    n_mfcc = int(length / CONFIG.data.mfcc_window)
    if device == torch.device('cpu'):
        # parallel learning
        if CONFIG.inference.enable_multiproc is True:
            model.share_memory()
            p = Pool(cpu_count() - 1)
            args = [i for i in range(n_mfcc)]
            outputs = p.map(cpu_inference, args)
        else:
            outputs = []
            for i in tqdm(range(n_mfcc)):
                outputs.append(cpu_inference(i))

    elif device == torch.device('cuda'):
        # batch processing
        outputs = gpu_inference(n_mfcc)

    if CONFIG.inference.log.save is True:
        writer = SummaryWriter(f'{CONFIG.inference.log.dir}/{file_name}')
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
