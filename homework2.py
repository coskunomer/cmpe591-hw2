import time
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import random
import environment
from replay_buffer import ReplayBuffer
from DQN_model import DQN
import math

class Hw2Env(environment.BaseEnv):
    def __init__(self, n_actions=8, **kwargs) -> None:
        super().__init__(**kwargs)
        self._n_actions = n_actions
        self._delta = 0.05

        theta = np.linspace(0, 2*np.pi, n_actions)
        actions = np.stack([np.cos(theta), np.sin(theta)], axis=1)
        self._actions = {i: action for i, action in enumerate(actions)}

        self._goal_thresh = 0.01
        self._max_timesteps = 50

    def _create_scene(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        scene = environment.create_tabletop_scene()
        obj_pos = [np.random.uniform(0.25, 0.75),
                   np.random.uniform(-0.3, 0.3),
                   1.5]
        goal_pos = [np.random.uniform(0.25, 0.75),
                    np.random.uniform(-0.3, 0.3),
                    1.025]
        environment.create_object(scene, "box", pos=obj_pos, quat=[0, 0, 0, 1],
                                  size=[0.03, 0.03, 0.03], rgba=[0.8, 0.2, 0.2, 1],
                                  name="obj1")
        environment.create_visual(scene, "cylinder", pos=goal_pos, quat=[0, 0, 0, 1],
                                  size=[0.05, 0.005], rgba=[0.2, 1.0, 0.2, 1],
                                  name="goal")
        return scene

    def state(self):
        if self._render_mode == "offscreen":
            self.viewer.update_scene(self.data, camera="topdown")
            pixels = torch.tensor(self.viewer.render().copy(), dtype=torch.uint8).permute(2, 0, 1)
        else:
            pixels = self.viewer.read_pixels(camid=1).copy()
            pixels = torch.tensor(pixels, dtype=torch.uint8).permute(2, 0, 1)
            pixels = transforms.functional.center_crop(pixels, min(pixels.shape[1:]))
            pixels = transforms.functional.resize(pixels, (128, 128))
        return pixels / 255.0

    def high_level_state(self):
        ee_pos = self.data.site(self._ee_site).xpos[:2]
        obj_pos = self.data.body("obj1").xpos[:2]
        goal_pos = self.data.site("goal").xpos[:2]
        return np.concatenate([ee_pos, obj_pos, goal_pos])

    def reward(self):
        state = self.high_level_state()
        ee_pos = state[:2]
        obj_pos = state[2:4]
        goal_pos = state[4:6]
        ee_to_obj = max(100*np.linalg.norm(ee_pos - obj_pos), 1)
        obj_to_goal = max(100*np.linalg.norm(obj_pos - goal_pos), 1)
        return 1/(ee_to_obj) + 1/(obj_to_goal)

    def is_terminal(self):
        obj_pos = self.data.body("obj1").xpos[:2]
        goal_pos = self.data.site("goal").xpos[:2]
        return np.linalg.norm(obj_pos - goal_pos) < self._goal_thresh

    def is_truncated(self):
        return self._t >= self._max_timesteps

    def step(self, action_id):
        action = self._actions[action_id] * self._delta
        ee_pos = self.data.site(self._ee_site).xpos[:2]
        target_pos = np.concatenate([ee_pos, [1.06]])
        target_pos[:2] = np.clip(target_pos[:2] + action, [0.25, -0.3], [0.75, 0.3])
        self._set_ee_in_cartesian(target_pos, rotation=[-90, 0, 180], n_splits=30, threshold=0.04)
        self._t += 1

        state = self.state()
        reward = self.reward()
        terminal = self.is_terminal()
        truncated = self.is_truncated()
        return state, reward, terminal, truncated


BUFFER_SIZE = 10000
BATCH_SIZE = 32
GAMMA = 0.99
EPSILON_START = 1
EPSILON_END = 0.1
EPSILON_DECAY = 1000
LEARNING_RATE = 0.001
TARGET_UPDATE = 100
NUM_EPISODES = 3000
TAU = 0.005

if __name__ == "__main__":
    N_ACTIONS = 8
    env = Hw2Env(n_actions=N_ACTIONS, render_mode="offscreen")
    epsilon = EPSILON_START
    steps_done = 0
    replay_buffer = ReplayBuffer(BUFFER_SIZE)
    dqn = DQN(N_ACTIONS)
    dqn_target = DQN(N_ACTIONS)
    optimizer = torch.optim.Adam(dqn.parameters(), lr=LEARNING_RATE)

    reward_per_episode = []
    reward_per_step = []

    for episode in range(NUM_EPISODES):
        _ = env.reset() 
        high_level_state = env.high_level_state() 
        done = False
        cumulative_reward = 0.0
        episode_steps = 0
        reward = 0
        new_reward = 0

        while not done:
            epsilon_threshold = EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(-1. * steps_done / EPSILON_DECAY)
            if random.random() < epsilon_threshold:
                action = np.random.randint(N_ACTIONS)
            else:
                state_tensor = torch.tensor(high_level_state, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    q_values = dqn(state_tensor)
                action = torch.argmax(q_values).item()

            _, new_reward, is_terminal, is_truncated = env.step(action)
            done = is_terminal or is_truncated
            next_high_level_state = env.high_level_state()  

            rb_reward = new_reward - reward
            if rb_reward < 0:
                rb_reward *= 3
            if is_terminal:
                rb_reward = 1_000
            replay_buffer.push(high_level_state, action, new_reward - reward, next_high_level_state, is_terminal)

            high_level_state = next_high_level_state
            cumulative_reward += new_reward
            reward = new_reward
            episode_steps += 1

            if replay_buffer.size() >= BATCH_SIZE:
                # Sample a batch of experiences from the replay buffer
                batch = replay_buffer.sample(BATCH_SIZE)
                states, actions, rewards, next_states, dones = zip(*batch)

                # Convert to torch tensors
                states = torch.tensor(np.array(states), dtype=torch.float32)
                actions = torch.tensor(np.array(actions), dtype=torch.int64)
                rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
                next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
                dones = torch.tensor(np.array(dones), dtype=torch.float32)

                q_values = dqn(states)
                next_q_values = dqn_target(next_states)

                q_value = q_values.gather(1, actions.unsqueeze(1))

                next_q_value = next_q_values.max(1)[0]
                target = rewards + (GAMMA * next_q_value * (1 - dones))

                loss_fn = nn.SmoothL1Loss()
                loss = loss_fn(target, q_value.squeeze())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            target_net_state_dict = dqn_target.state_dict()
            policy_net_state_dict = dqn.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            dqn_target.load_state_dict(target_net_state_dict)

            steps_done += 1
        
        # Store rewards for plotting later
        reward_per_episode.append(cumulative_reward)
        reward_per_step.append(cumulative_reward / episode_steps)

        # Save model every 20th episode
        if (episode + 1) % 20 == 0:
            epsilon_threshold = EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(-1. * steps_done / EPSILON_DECAY)
            print("Epsilon:", epsilon_threshold)
            torch.save(dqn.state_dict(), f"dqn_model.pth")
            np.save('reward_per_episode.npy', reward_per_episode)
            np.save('reward_per_step.npy', reward_per_step)

        print(f"Episode {episode + 1}/{NUM_EPISODES}, Reward: {cumulative_reward}, Steps: {episode_steps}")