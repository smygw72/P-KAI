import os
import random
import numpy as np
import torch
from mutagen.mp3 import MP3
from torchvision import transforms
from tensorboardX import SummaryWriter

import _paths
from config import CONFIG
from network.interface import get_model
from utils.make_mfcc import spec_to_image, get_melspectrogram_db


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def mutagen_length(path):
    try:
        audio = MP3(path)
        length = audio.info.length
        return length
    except Exception:
        return None


def main(*args, **kwargs):
    set_seed(CONFIG.inference.seed)

    # file_path = args.path
    file_path = '../dataset/sounds/2oGJrufm_4w.mp3'
    file_name = os.path.basename(file_path)
    length = mutagen_length(file_path)

    writer = SummaryWriter(f'{CONFIG.inference.path.log_dir}/{file_name}')

    model = get_model(CONFIG.common.arch).to(device)
    model_path = f'{CONFIG.common.path.model_dir}/{CONFIG.common.version}.pth'
    model.load_state_dict(torch.load(model_path))
    model.eval()

    inputs = torch.Tensor()
    outputs = torch.Tensor()

    n_mfcc = int(length / CONFIG.inference.window_wid)
    with torch.no_grad():
        for i in range(n_mfcc):
            start_segment = i * CONFIG.inference.window_wid

            spec = get_melspectrogram_db(
                file_path, offset=start_segment,
                duration=CONFIG.inference.window_wid
            )
            input_arr = spec_to_image(spec)  # 2D
            input_tensor = torch.from_numpy(input_arr).unsqueeze(0)

            transform = transforms.Compose([
                transforms.Resize((
                    CONFIG.common.input_size, CONFIG.common.input_size
                )),
            ])
            input_tensor = transform(input_tensor)

            inputs = torch.cat([inputs, input_tensor.unsqueeze(0)], dim=0)

            if (len(inputs) == CONFIG.inference.batch_size) or (i == n_mfcc - 1):
                inputs = inputs.to(device)
                output = model(inputs).detach()
                outputs = torch.cat([outputs, output], dim=0)
                inputs = torch.Tensor()  # make empty

    for i in len(outputs):
        writer.add_scalar("timeline", outputs[i], i)

    score_avg = torch.mean(outputs).item()
    print(f"average score: {score_avg}")

    writer.close()

    return score_avg


if __name__ == "__main__":
    main()
