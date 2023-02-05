import os
import warnings
import glob
import time
from tqdm import tqdm
import torch
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

# from pytorch_grad_cam import GradCAM
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# from pytorch_grad_cam.utils.image import show_cam_on_image

from src.config import get_config
# from calc_score import read_score, predict_absolute_score
from src.network.model import get_model
from src.metric import mean_scores
from src.singledata import get_dataloader
from src.utils import set_seed, get_timestamp

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def main(sound_path=None, learning_log_dir=None, local_or_lambda='local') -> float:

    cfg = get_config(inference_mode=True)
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

    model = get_model(cfg, 'inference').to(device)

    if learning_log_dir is None:
        # when called from aws lambda
        state_dict_path = glob.glob(
            './model/**/state_dict.pt', recursive=True)[0]
    else:
        # when called from learning.py
        state_dict_path = f'{log_dir}/state_dict.pt'

    checkpoint = torch.load(state_dict_path, map_location=device)
    model.load_state_dict(checkpoint['best_model'])
    print(f'Best epoch    : {checkpoint["best_epoch"]}')
    print(f'Best accuracy : {checkpoint["best_accuracy"]}')

    # main
    outputs = inference(cfg, model, sound_path)

    # save
    if (cfg.inference.save_log is True) and (local_or_lambda == 'local'):
        writer = SummaryWriter(f'{log_dir}/{file_name}')
        for i in range(len(outputs)):
            writer.add_scalar("score_change", outputs[i], i)
        writer.close()

    score_avg = sum(outputs) / len(outputs)
    print(f"average score: {score_avg}")

    end_time = time.time()
    print(f"elapsed time: {end_time - start_time}")
    return score_avg


def inference(cfg, model, sound_path):

    img_size = cfg.data.img_size
    img_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    dataloader = get_dataloader(cfg, sound_path, img_transform)
    if len(dataloader) == 0:
        print("Warning: len(dataloader) == 0")

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
                # gradcam()
            scores = torch.cat([scores, score.to('cpu')], dim=0)

    return scores.squeeze().tolist()


def gradcam(model, input):
    target_layers = [model.layer4[-1]]
    targets = [ClassifierOutputTarget(0)]
    cam = GradCAM(model=model, target_layers=target_layers,
                  use_cuda=(device == torch.device('cuda')))
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=False)
    visualization.save('')


def visualize_attention(outs):
    att_good, att_bad = outs[2], outs[5]
    # TODO

if __name__ == '__main__':
    main()