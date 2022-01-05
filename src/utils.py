import random
from datetime import datetime
from pytz import timezone
import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# for dataloader
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_timestamp():
    utc_now = datetime.now(timezone('UTC'))
    jst_now = utc_now.astimezone(timezone('Asia/Tokyo'))
    timestamp = datetime.strftime(jst_now, '%m-%d-%H-%M-%S')
    return timestamp

def debug_setting(enable=True):
    # torch.autograd.detect_anomaly(enable)
    # torch.autograd.profiler.profile(enable)
    torch.autograd.set_detect_anomaly(enable)
    # torch.autograd.profiler.emit_nvtx(enable)
    # torch.autograd.gradcheck(enable)
    # torch.autograd.gradgradcheck(enable)