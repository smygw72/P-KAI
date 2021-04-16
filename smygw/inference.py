import random
import numpy as np
import torch
from mutagen.mp3 import MP3

from .utils.make_mfcc import spec_to_image, get_melspectrogram_db
from network import get_model
from config import CONFIG


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
    except:
        return None


def main(*args, **kwargs):
    set_seed(CONFIG.seed)

    model = get_model().to(device)
    model.eval()

    file = None  # TODO
    length = mutagen_length(file)

    scores = torch.Tensor()

    i = 0
    segment_len = 3
    with torch.no_grad():
        while i * segment_len < length:
            file_idx = str(i).zfill(6)
            start_segment = i*segment_len

            spec = get_melspectrogram_db(
                sound_file_path, offset=start_segment,
                duration=segment_len
            )
            mfcc_np = spec_to_image(spec)
            mfcc_tensor = torch.from_numpy(mfcc_np).unsqueeze(0)

            score = model(mfcc_tensor).detach()
            scores = torch.cat([scores, score], dim=0)

            i += 1

    score_avg = torch.mean(scores).item()
    print(f"average score: {score_avg}")


if __name__ == "__main__":
    main()
