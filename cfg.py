import torch
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FRAME_SKIP = 8
GAMMA = 0.95
EPS_MAX = 1
EPS_DECAY = 1e-3
EPS_MIN = 0.1
TARGET_NET_UPDATE = 100
BUFFER_SIZE = 100000
BATCH_SIZE = 256
LR = 1e-4
TRAIN_STEP = 5e5
SAVE_STEP = 2e3
SAVE_PATH = os.path.dirname(os.path.abspath(__file__)) + '/weights/dqn.pt'
ENV_NAME = 'ALE/Seaquest-v5'

# --- RAM version ---
TORSO_SHAPE = [128, 128, 128]


# --- RGB version ---
FRAME_HIST = 4
FRAME_HEIGHT = 96
FRAME_WIDTH = 96
TO_GREY_SCALE = True
IN_CHANNEL = FRAME_HIST if TO_GREY_SCALE else FRAME_HIST * 3
