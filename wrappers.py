import gymnasium as gym
import numpy as np
import collections
import cv2

class AtariBufferWrapper(gym.Wrapper):
    def __init__(self,
                 env,
                 to_float=True):
        super(AtariBufferWrapper, self).__init__(env)
        self._dtype = np.uint8 if not to_float else np.float32
        self.observation_space = gym.spaces.Box(0, 255, list(env.observation_space.shape),
                                                self._dtype)
        self.action_space = gym.spaces.Discrete(6)

    def _process_ob(self, ob):
        ob = ob.astype(self._dtype)
        return ob / 255.

    @staticmethod
    def _process_reward(reward):
        reward /= 20.
        return reward

    def reset(self, **kwargs):
        ob, info = self.env.reset(**kwargs)
        ob = self._process_ob(ob)
        return ob, info

    def step(self, action):
        ob, r, done, truncated, info = self.env.step(action)
        reward = self._process_reward(r)
        ob = self._process_ob(ob)
        return ob, reward, done, truncated, info


class AtariBufferWrapperObStack(gym.Wrapper):
    def __init__(self,
                 env,
                 to_float=True,
                 ob_num=3):
        super(AtariBufferWrapperObStack, self).__init__(env)
        self._dtype = np.uint8 if not to_float else np.float32
        self.observation_space = gym.spaces.Box(0, 255, [128 * ob_num],
                                                self._dtype)
        self.action_space = gym.spaces.Discrete(6)
        self.ob_stack = np.zeros([ob_num, 128], dtype=np.float32)


    def _process_ob(self, ob):
        ob = ob.astype(self._dtype)
        ob /= 255.
        self.ob_stack[1:] = self.ob_stack[:-1]
        self.ob_stack[0] = ob
        return self.ob_stack.flatten()

    @staticmethod
    def _process_reward(reward):
        reward /= 20.
        return reward

    def reset(self, **kwargs):
        ob, info = self.env.reset(**kwargs)
        ob = self._process_ob(ob)
        return ob, info

    def step(self, action):
        ob, r, done, truncated, info = self.env.step(action)
        reward = self._process_reward(r)
        ob = self._process_ob(ob)
        return ob, reward, done, truncated, info


class AtariBufferWrapperTestEnt(gym.Wrapper):
    def __init__(self,
                 env,
                 to_float=True,
                 act_stack=100):
        super(AtariBufferWrapperTestEnt, self).__init__(env)
        self._dtype = np.uint8 if not to_float else np.float32
        # self.observation_space = gym.spaces.Box(0, 255, list(env.observation_space.shape),
        #                                         self._dtype)
        self.observation_space = gym.spaces.Box(0, 1, [128 + (10 * 6)], self._dtype)
        self.action_space = gym.spaces.Discrete(6)

        self.action_stack = np.ones(act_stack, dtype=int) * -1
        self.action_ctr = np.ones(self.action_space.n)

    def _process_action(self, a):
        if self.action_stack[-1] != -1:
            self.action_ctr[self.action_stack[-1]] -= 1
        self.action_ctr[a] += 1
        self.action_stack[1:] = self.action_stack[0: -1]
        self.action_stack[0] = a
        action_prob = self.action_ctr / self.action_ctr.sum()
        ent = 0
        p_log_p = action_prob * np.log(action_prob)
        for i in range(10):
            ent -= p_log_p[self.action_stack[i]]
        # ent = -np.sum(action_prob * np.log(action_prob))

        return ent

    def _process_ob(self, ob):
        ob = ob.astype(self._dtype)
        ob /= 255.
        one_hot_action = np.zeros([10, self.action_space.n], dtype=self._dtype)
        one_hot_action[np.arange(len(one_hot_action)), self.action_stack[:10]] = 1.
        ob = np.concatenate([ob, one_hot_action.flatten()], dtype=self._dtype)
        return ob

    @staticmethod
    def _process_reward(reward):
        reward /= 20.
        return reward

    def reset(self, **kwargs):
        ob, info = self.env.reset(**kwargs)
        ob = self._process_ob(ob)
        return ob, info

    def step(self, action):
        ob, r, done, truncated, info = self.env.step(action)
        ent = self._process_action(action)
        reward = self._process_reward(r)
        reward -= ent / 30.
        ob = self._process_ob(ob)
        return ob, reward, done, truncated, info

class AtariFrameWrapper(gym.Wrapper):
    def __init__(self,
                 env,
                 height=96,
                 width=96,
                 grey_scale=False,
                 frame_hist=4):
        super(AtariFrameWrapper, self).__init__(env)
        self.height = height
        self.width = width
        self.chn = 1 if grey_scale else 3
        self.frame_hist = frame_hist
        self.observation_space = gym.spaces.Box(low=0, high=1,
                                                shape=(self.height, self.width, self.chn * frame_hist))
        self.hist_stack = self._make_hist_stack()

    def reset(self, **kwargs):
        ob, info = self.env.reset(**kwargs)
        self.hist_stack = self._make_hist_stack()
        ob = self._process_frame(ob)
        return ob, info

    def step(self, action):
        ob, reward, done, truncated, info = self.env.step(action)
        ob = self._process_frame(ob)
        reward = self._process_reward(reward)
        return ob, reward, done, truncated, info


    def _make_hist_stack(self):
        return np.zeros([self.height, self.width, self.chn * self.frame_hist], dtype=np.float32)

    def _process_frame(self, ob):
        if self.chn == 1:
            ob = cv2.cvtColor(ob, cv2.COLOR_RGB2GRAY)
        ob = cv2.resize(ob, (self.width, self.height), interpolation=cv2.INTER_AREA)[:, :, np.newaxis]
        self.hist_stack[:, :, self.chn:] = self.hist_stack[:, :, 0: -self.chn]
        self.hist_stack[:, :, :self.chn] = ob
        return self.hist_stack

    @staticmethod
    def _process_reward(reward):
        reward /= 20
        return reward
