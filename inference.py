import os
import warnings
import glob
import time
from tqdm import tqdm
import torch
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from config.config import get_config
from src.audio import get_samples
from src.network.model import MyModel
from src.metric import mean_scores
from src.singledata import get_dataloader
from src.utils import set_seed, get_timestamp

# global variables
model = None
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


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
    if sound_path is None:
        sound_path = './misc/test.mp3'
    file_name = os.path.splitext(os.path.basename(sound_path))[0]

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
    outputs = inference(sound_path)

    # save
    if cfg.inference.save_log is True:
        writer = SummaryWriter(f'{log_dir}/{train_or_test}/{file_name}')
        for i in range(len(outputs)):
            writer.add_scalar("timeline", outputs[i], i)
        writer.close()

    score_avg = sum(outputs) / len(outputs)
    print(f"average score: {score_avg}")

    end_time = time.time()
    print(f"elapsed time: {end_time - start_time}")
    return score_avg


def inference(sound_path):

    img_size = cfg.data.img_size
    img_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    dataloader = get_dataloader(cfg, sound_path, img_transform)

    scores = torch.Tensor()
    model.eval()
    with torch.no_grad():
        for sample in tqdm(dataloader):
            outs = model(sample.to(device))
            outs = mean_scores(outs)
            if cfg.model.architecture != 'PDR':
                score = outs[0] - outs[3]
                visualize_attention(outs)
            else:
                score = outs[0]
            scores = torch.cat([scores, score.to('cpu')], dim=0)

    return scores.squeeze().tolist()


def visualize_attention(outs):
    att_good, att_bad = outs[2], outs[5]
    # TODO


if __name__ == "__main__":
    main()
