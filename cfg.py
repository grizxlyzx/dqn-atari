import torch
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FRAME_SKIP = 8
GAMMA = 0.95
EPS_MAX = 1.
EPS_MIN = 0.1
EPS_DECAY_STEP = 1e5
EPS_DECAY = (EPS_MAX - EPS_MIN) / EPS_DECAY_STEP
TARGET_NET_UPDATE = 1000
BUFFER_SIZE = 200000
# BUFFER_SIZE = 10
BATCH_SIZE = 32
LR = 1e-4
TRAIN_STEP = 5e5
SAVE_STEP = 2e3
SAVE_PATH = os.path.dirname(os.path.abspath(__file__)) + '/weights/dqn.pt'
ENV_NAME = 'ALE/Seaquest-v5'

# --- RAM version ---
TORSO_SHAPE = [128, 128, 128]


# --- RGB version ---
FRAME_HIST = 4
FRAME_HEIGHT = 110
FRAME_WIDTH = 84
TO_GREY_SCALE = True
IN_CHANNEL = FRAME_HIST if TO_GREY_SCALE else FRAME_HIST * 3
