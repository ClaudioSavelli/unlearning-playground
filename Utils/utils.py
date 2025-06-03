import torch
import numpy as np
import random

import argparse

def set_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    return torch.Generator().manual_seed(random_seed)

def parse_cmd_line_params():
    parser = argparse.ArgumentParser(description="Unlearning Script")
    parser.add_argument(
        "--batch",
        help="batch size",
        default=8, 
        type=int,
        required=False)
    parser.add_argument(
        "--model_name_or_path",
        help="model to use",
        default="models",  
        type=str,                          
        required=False)                     
    parser.add_argument(
        "--unlearner",
        help="unlearner to use",
        default="None",
        type=str,
        required=False)
    parser.add_argument(
        "--output_dir",
        help="path to the output directory",
        default="models",
        type=str,
        required=False)
    parser.add_argument(
        "--lr",
        help="learning rate",
        default=1e-4,
        type=float,
        required=False)
    parser.add_argument(
        "--use_bad_teaching",
        default=1,
        type=int,
        required=False)
    parser.add_argument(
        "--seed",
        help="seed",
        default=0,
        type=int,
        required=False)
    parser.add_argument(
        "--unfreeze_encoder_layer", 
        help="unfreeze encoder layer",
        default=-1,
        type=int,
        required=False)
    parser.add_argument(
        "--epochs",
        help="epochs",
        default=1,
        type=int,
        required=False
    )
    args = parser.parse_args()
    return args