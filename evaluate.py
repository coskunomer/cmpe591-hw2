from homework2 import Hw2Env
from DQN_model import DQN
import torch

model_path = "dqn_model.pth"
N_ACTIONS = 8
dqn = DQN(N_ACTIONS)
dqn.load_state_dict(torch.load(model_path))
dqn.eval()

env = Hw2Env(n_actions=N_ACTIONS, render_mode="gui")

num_test_episodes = 20
total_reward = 0

for episode in range(num_test_episodes):
    _ = env.reset() 
    high_level_state = env.high_level_state() 
    done = False
    cumulative_reward = 0.0
    episode_steps = 0

    while not done:
        state_tensor = torch.tensor(high_level_state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = dqn(state_tensor)
        action = torch.argmax(q_values).item()  # Full greedy policy

        _, new_reward, is_terminal, is_truncated = env.step(action)
        done = is_terminal or is_truncated
        next_high_level_state = env.high_level_state()  

        cumulative_reward += new_reward
        high_level_state = next_high_level_state
        episode_steps += 1

    total_reward += cumulative_reward
    print(f"Episode {episode + 1}/{num_test_episodes}, Reward: {cumulative_reward}, Steps: {episode_steps}")

average_reward = total_reward / num_test_episodes
print(f"Average reward over {num_test_episodes} episodes: {average_reward}")