import os
import warnings
import random
import time
from tqdm import tqdm
from mutagen.mp3 import MP3
import torch
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from config.config import CONFIG
from preprocessing.make_mfcc import spec_to_image, get_melspectrogram_db
from src.pool import Pool
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
    start_segment = i * CONFIG.data.mfcc_window
    spec = get_melspectrogram_db(
        file_path,
        offset=start_segment,
        duration=CONFIG.data.mfcc_window
    )
    mfcc_arr = spec_to_image(spec)
    mfcc = torch.from_numpy(mfcc_arr).float()
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(
            (CONFIG.data.img_size, CONFIG.data.img_size)),
        transforms.ToTensor(),
    ])
    mfcc = transform(mfcc)
    mfcc = mfcc.unsqueeze(0)
    return mfcc


def inference(n_mfcc):
    mfccs = torch.Tensor()
    for i in tqdm(range(n_mfcc)):
        mfcc = get_mfcc(i)
        mfccs = torch.cat([mfccs, mfcc], dim=0)

    dataset = torch.utils.TensorDataset(mfccs)
    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG.inference.n_frame,
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


def main(sound_path=None, learning_log_dir=None, train_or_test='train') -> float:

    start_time = time.time()
    warnings.filterwarnings('ignore')
    set_seed(CONFIG.seed)

    if learning_log_dir is None:
        timestamp = get_timestamp()
        log_dir = f'./inference_logs/{timestamp}'
    else:
        log_dir = learning_log_dir

    global file_path
    if sound_path is None:
        file_path = './misc/test.mp3'
    else:
        file_path = sound_path

    file_name = os.path.splitext(os.path.basename(file_path))[0]
    length = mutagen_length(file_path)

    global model
    model = MyModel(CONFIG.inference.n_frame).to(device)

    if learning_log_dir is None:
        model_dir = os.path.dirname(CONFIG.path)
        state_dict_path = f'{model_dir}/split_id=0/state_dict.pt'       # lambda運用時
    else:
        state_dict_path = f'{log_dir}/state_dict.pt'                    # 学習結果評価時
    checkpoint = torch.load(state_dict_path, map_location=device)
    model.load_state_dict(checkpoint['best_model'])
    print(f'Best epoch: {checkpoint["best_epoch"]}')
    print(f'Best loss : {checkpoint["best_loss"]}')
    model.eval()

    n_mfcc = int(length / CONFIG.data.mfcc_window)
    outputs = inference(n_mfcc)

    if CONFIG.inference.save_log is True:
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
