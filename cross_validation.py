
from config import CONFIG
from learning import main as learn

if __name__=='__main__':
    split_ids = [0, 1, 2]
    for split_id in split_ids:
        CONFIG.learning.split_id = split_id
        learn()