import time

import numpy as np
import torch
from model import MLP, DQNConv, ReplayBuffer
from wrappers import AtariFrameWrapper
from Logger import Logger
from cfg import *
import gymnasium as gym
import tqdm
import os
import sys

SAVE_PATH = os.path.dirname(os.path.abspath(__file__)) + '/weights/dqn_rgb.pt'

def make_env(render):
    env = gym.make(ENV_NAME, frameskip=FRAME_SKIP,
                   obs_type='rgb', repeat_action_probability=0.1,
                   render_mode=render, full_action_space=False)
    env = AtariFrameWrapper(env, FRAME_HEIGHT, FRAME_WIDTH, TO_GREY_SCALE, FRAME_HIST)
    return env

def play_one_game(dqn, render=True):
    render_mode = 'human' if render else None
    env = make_env(render_mode)
    ob, _ = env.reset()
    done = False
    reward = 0
    while not done:
        a = dqn.choose_action(ob, eps=0)
        ob, r, done, _, _ = env.step(a)
        reward += r
        if render:
            env.render()
    return reward

def evaluation(env, dqn, iteration=10):
    ret = 0
    for i in range(iteration):
        ob, _ = env.reset()
        done = False
        while not done:
            a = dqn.choose_action(ob, eps=0.05)
            ob, r, done, _, _ = env.step(a)
            ret += r
    return ret / iteration

def run():
    env = make_env(render=None)
    eps = EPS_MAX
    replay_buffer = ReplayBuffer(capacity=BUFFER_SIZE)
    dqn = DQNConv(in_channels=FRAME_HIST, n_actions=env.action_space.n, device=DEVICE, gamma=GAMMA)
    dqn_tgt = DQNConv(in_channels=FRAME_HIST, n_actions=env.action_space.n, device=DEVICE, gamma=GAMMA)
    dqn.load_weights(SAVE_PATH)
    dqn_tgt.load_weights(SAVE_PATH)
    # for _ in range(10):
    #     play_one_game(dqn, True)
    optimizer = torch.optim.Adam(dqn.parameters(), lr=LR)
    loss_log = Logger()
    reward_log = Logger()
    epi_ctr = 0
    with tqdm.tqdm(total=TRAIN_STEP) as pbar:
        while True:
            done = False
            ob, _ = env.reset()
            r = 0
            while not done:
                a = dqn.choose_action(ob, eps)
                ob_nx, r_nx, done, _, _ = env.step(a)
                replay_buffer.add(ob, ob_nx, a, r, r_nx, done)
                ob = ob_nx
                r = r_nx
                reward_log.log(r)
                if replay_buffer.size() > BATCH_SIZE:
                    b_ob, b_ob_nx, b_a, _, b_r_nx, b_done = replay_buffer.sample(BATCH_SIZE)
                    loss = dqn.calc_1_step_td_loss(b_ob, b_a, b_r_nx, b_ob_nx, b_done, dqn_tgt)
                    loss_log.log(loss.item())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                if pbar.n % TARGET_NET_UPDATE == 0:
                    dqn_tgt.load_state_dict(dqn.state_dict())
                if pbar.n % SAVE_STEP == 0 and pbar.n > 1:
                    dqn.save_weights(SAVE_PATH)
                pbar.update(1)
            eps = eps - EPS_DECAY if eps > EPS_MIN else eps
            reward_log.new_epoch()
            loss_log.new_epoch()

            epi_ctr += 1
            if epi_ctr % 5 == 1:
                r_eval = evaluation(env, dqn, 3)
                pbar.set_postfix({
                    'epoch': f'{epi_ctr}',
                    'avg_loss': f'{loss_log.latest_mean(50):.6f}',
                    'avg_reward': f'{reward_log.latest_sum(50):.1f}',
                    'avg_eval_reward': f'{r_eval:.1f}',
                    'epsilon': f'{eps:.3f}'
                })

            if pbar.n > TRAIN_STEP:
                break


if __name__ == '__main__':
    run()
