# 라이브러리 선언
import argparse
import torch


def get_args():

    parser = argparse.ArgumentParser()
    
    # DATA
    parser.add_argument('--train_x_img_dir',type=str, default='./data/train/x')
    parser.add_argument('--train_y_img_dir',type=str, default='./data/train/y')
    parser.add_argument('--test_x_img_dir',type=str, default='./data/test/x')
    parser.add_argument('--test_y_img_dir',type=str, default='./data/test/y')
    parser.add_argument('--valid_x_img_dir',type=str, default='./data/valid/x')
    parser.add_argument('--valid_y_img_dir',type=str, default='./data/valid/y')
    
    parser.add_argument('--sampling_frequency',type=int, default=20000)
    parser.add_argument('--lr_length',type=int, default=1000, help='length of low-resolution signal')
    parser.add_argument('--hr_length',type=int, default=10000)
    
    parser.add_argument('--n_samples',type=int, default=100)
    parser.add_argument('--patch_size',type=int, default=128)
    parser.add_argument('--stride',type=int, default=96)
    parser.add_argument('--batch_size',type=int, default=16)
    
    # Learning
    parser.add_argument('--epochs',type=int, default=50)
    parser.add_argument('--lr',type=float, default=1e-4)
    parser.add_argument('--early_stop',type=int, default=20, help='early stop_patience')
    
    # GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument('--device', type=str, default=device, help='device')
    
    opt = parser.parse_args('')

    return opt