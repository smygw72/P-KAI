import random
import numpy as np
import torch
from mutagen.mp3 import MP3

from .utils.make_mfcc import spec_to_image, get_melspectrogram_db
from network import get_model
from inference_config import CONFIG


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
    model_path = f'./model/{CONFIG.version}.pth'
    model.load_state_dict(torch.load(model_path))
    model.eval()

    file_path = args.path
    length = mutagen_length(file_path)

    inputs = torch.Tensor()
    outputs = torch.Tensor()

    n_mfcc = int(length / CONFIG.window_wid)
    with torch.no_grad():
        for i in range(n_mfcc):
            start_segment = i * CONFIG.window_wid

            spec = get_melspectrogram_db(
                file_path, offset=start_segment,
                duration=CONFIG.window_wid
            )
            input_arr = spec_to_image(spec)
            input_tensor = torch.from_numpy(input_arr).unsqueeze(0)
            inputs = torch.cat([inputs, input_tensor], dim=0)

            if (len(inputs) == CONFIG.batch_size) or (i == n_mfcc - 1):
                output = model(inputs).detach()
                outputs = torch.cat([outputs, output], dim=0)
                inputs = torch.Tensor()  # make empty

    score_avg = torch.mean(outputs).item()
    print(f"average score: {score_avg}")
    return score_avg


if __name__ == "__main__":
    main()
