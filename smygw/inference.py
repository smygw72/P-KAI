import argparse
import random
import numpy as np
import torch
from mutagen.mp3 import MP3

from .utils.make_mfcc import spec_to_image, get_melspectrogram_db
from network import get_model
from config import CONFIG


window_width = 2
batch_size = 16


parser = argparse.ArgumentParser()
parser.add_argument('--path', default='./smygw/sounds/Z_-hWZetOS0.mp3')
args = parser.parse_args()

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
    set_seed(CONFIG.seed)

    model = get_model().to(device)
    model.eval()

    file_path = args.path
    length = mutagen_length(file_path)

    inputs = torch.Tensor()
    outputs = torch.Tensor()

    n_mfcc = int(length / window_width)
    with torch.no_grad():
        for i in range(n_mfcc):
            start_segment = i * window_width

            spec = get_melspectrogram_db(
                file_path, offset=start_segment,
                duration=window_width
            )
            input_arr = spec_to_image(spec)
            input_tensor = torch.from_numpy(input_arr).unsqueeze(0)
            inputs = torch.cat([inputs, input_tensor], dim=0)

            if (len(inputs) == batch_size) or (i == n_mfcc - 1):
                output = model(inputs).detach()
                outputs = torch.cat([outputs, output], dim=0)
                inputs = torch.Tensor()

    score_avg = torch.mean(outputs).item()
    print(f"average score: {score_avg}")
    return score_avg


if __name__ == "__main__":
    main()
