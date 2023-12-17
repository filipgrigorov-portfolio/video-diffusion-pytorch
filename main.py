import argparse
import sys
sys.path.append('./tools')
sys.path.append('./video_diffusion_pytorch/')

import torch
from video_diffusion_pytorch import Unet3D, GaussianDiffusion, Trainer

# Definitions
GRID_SIZE = 64
OUTPUT_CHANNELS = 3
TOTAL_SIMULATION_TIME = 20
TRAINING_DATA_PATH = "video_diffusion_pytorch/data_20s/"

EPOCHS = 500
BATCH_SIZE = 4
LR = 1e-4
SAVE_EVERY = 50
GRAD_ACC = 2
EMA_DECAY = 0.995

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--sample', action='store_true')

    args = parser.parse_args()

    model = Unet3D(
        dim=GRID_SIZE, 
        dim_mults=(1, 2, 4, 8), 
        channels=OUTPUT_CHANNELS)
    
    ddpm = GaussianDiffusion(
        model,
        image_size = GRID_SIZE, 
        num_frames = TOTAL_SIMULATION_TIME, 
        timesteps = 400, 
        loss_type = 'l2').cuda()
    
    if args.train:
        trainer = Trainer(ddpm, 
            TRAINING_DATA_PATH, 
            train_batch_size=BATCH_SIZE, 
            train_lr=LR,
            save_and_sample_every=SAVE_EVERY,
            train_num_steps=EPOCHS,
            gradient_accumulate_every=GRAD_ACC,
            ema_decay=EMA_DECAY,
            amp=True)
        trainer.train()
        print("Done training")
        
    elif args.sample:
        text = torch.randn(2, 64) # TODO: Remove txt conditioning and replace with ICs and BCs
        sampled_videos = ddpm.sample(cond = text)
        sampled_videos.shape # (2, 3, 5, 32, 32)

    print("Done")
