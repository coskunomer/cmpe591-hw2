import numpy as np
import matplotlib.pyplot as plt

reward_per_episode = np.load('3000_episodes/reward_per_episode.npy')
reward_per_step = np.load('3000_episodes/reward_per_step.npy')

def exponential_moving_average(data, alpha=0.1):
    ema = np.zeros_like(data)
    ema[0] = data[0]
    for t in range(1, len(data)):
        ema[t] = alpha * data[t] + (1 - alpha) * ema[t-1]
    return ema

ema_episode = exponential_moving_average(reward_per_episode, alpha=0.1)
ema_step = exponential_moving_average(reward_per_step, alpha=0.1)

x = np.arange(len(ema_episode))  
slope_episode, intercept_episode = np.polyfit(x, ema_episode, 1)  
slope_step, intercept_step = np.polyfit(x, ema_step, 1) 

linear_fit_episode = slope_episode * x + intercept_episode
linear_fit_step = slope_step * x + intercept_step

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(reward_per_episode, label='Reward per Episode')
plt.plot(ema_episode, label='EMA (alpha=0.1)', linestyle='--')
plt.plot(x, linear_fit_episode, label='Linear Fit to EMA', linestyle='-', linewidth=3)  
plt.title('Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(reward_per_step, label='Reward per Step')
plt.plot(ema_step, label='EMA (alpha=0.1)', linestyle='--')
plt.plot(x, linear_fit_step, label='Linear Fit to EMA', linestyle='-', linewidth=3)  
plt.title('Reward per Step')
plt.xlabel('Episode')
plt.ylabel('Reward per Step')
plt.legend()

plt.tight_layout()
plt.show()
