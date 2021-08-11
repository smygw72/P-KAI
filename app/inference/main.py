import os
import warnings
import random
from multiprocessing import Pool, cpu_count
import time
import numpy as np
from tqdm import tqdm
from mutagen.mp3 import MP3
import torch
from torchvision import transforms
from tensorboardX import SummaryWriter

from config import CONFIG
from network.interface import get_model
from utils.make_mfcc import spec_to_image, get_melspectrogram_db
from utils.common import set_seed

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def mutagen_length(path):
    try:
        audio = MP3(path)
        length = audio.info.length
        return length
    except Exception:
        return None


def cpu_inference(model, sound_path, i):
    start_segment = i * CONFIG.inference.window_wid
    spec = get_melspectrogram_db(
        sound_path,
        offset=start_segment,
        duration=CONFIG.inference.window_wid
    )
    input_arr = spec_to_image(spec)
    input_tensor = torch.from_numpy(input_arr).float()
    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)

    transform = transforms.Compose([
        transforms.Resize(
            (CONFIG.common.input_size, CONFIG.common.input_size)),
    ])

    input_tensor = transform(input_tensor)
    with torch.no_grad():
        score = model(input_tensor.to(device)).to('cpu').item()
    return score


def gpu_inference(model, sound_path, n_mfcc):
    inputs = torch.Tensor()
    outputs = torch.Tensor()

    for i in tqdm(range(n_mfcc)):
        start_segment = i * CONFIG.inference.window_wid

        spec = get_melspectrogram_db(sound_path,
                                     offset=start_segment,
                                     duration=CONFIG.inference.window_wid)
        input_arr = spec_to_image(spec)
        input_tensor = torch.from_numpy(input_arr)  # 2D tensor
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
        transform = transforms.Compose([
            transforms.Resize(
                (CONFIG.common.input_size, CONFIG.common.input_size)),
        ])
        input_tensor = transform(input_tensor)
        inputs = torch.cat([inputs, input_tensor], dim=0)

        if (len(inputs) == CONFIG.inference.batch_size) or (i == n_mfcc - 1):
            inputs = inputs.to(device)
            output = model(inputs).detach().to('cpu')
            outputs = torch.cat([outputs, output], dim=0)
            inputs = torch.Tensor()  # make empty
    return outputs.squeeze().tolist()


def main(sound_path=None) -> float:
    start_time = time.time()
    warnings.filterwarnings('ignore')
    set_seed(CONFIG.seed)

    if sound_path is None:
        sound_path = './inference/test.mp3'

    file_name = os.path.splitext(os.path.basename(sound_path))[0]
    length = mutagen_length(sound_path)

    version = f'{CONFIG.version.data}-{CONFIG.version.code}-{CONFIG.version.param}'

    model = get_model(CONFIG.common.arch, pretrained=False).to(device)
    model_path = f'{CONFIG.common.model_dir}/{version}/model.pth'
    model.load_state_dict(torch.load(model_path))
    model.eval()

    n_mfcc = int(length / CONFIG.inference.window_wid)
    if device == torch.device('cpu'):
        # parallel learning
        if CONFIG.inference.enable_multiprocessing is True:
            model.share_memory()
            p = Pool(cpu_count())
            args = [(model, sound_path, i) for i in range(n_mfcc)]
            outputs = p.starmap(cpu_inference, tqdm(args, total=n_mfcc))
        else:
            outputs = []
            for i in tqdm(range(n_mfcc)):
                outputs.append(cpu_inference(model, sound_path, i))

    elif device == torch.device('cuda'):
        # batch processing
        outputs = gpu_inference(model, sound_path, n_mfcc)

    if CONFIG.inference.log.save is True:
        writer = SummaryWriter(f'{CONFIG.inference.log.dir}/{version}/{file_name}')
        for i in range(len(outputs)):
            writer.add_scalar("timeline", outputs[i], i)
        writer.close()

    score_avg = sum(outputs) / len(outputs)
    print(f"average score: {score_avg}")

    end_time = time.time()
    sec_per_frame = (end_time - start_time) / n_mfcc
    print(f"elapsed time: {sec_per_frame}")
    return score_avg


if __name__ == "__main__":
    import _paths
    main()
