import gym
import numpy as np
env = gym.make('FrozenLake-v1', is_slippery=True, render_mode=None)
learning_rate = 0.1
discount_factor = 0.99
epsilon = 1.0  # Exploration rate
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.005  # Decay rate for exploration probability
state_size = env.observation_space.n
action_size = env.action_space.n
q_table = np.zeros((state_size, action_size))
num_episodes = 10000
max_steps = 100  
for episode in range(num_episodes):
    state = env.reset()[0]
    done = False
    for step in range(max_steps):
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(q_table[state, :])  # Exploit
        new_state, reward, done, truncated, info = env.step(action)
        q_table[state, action] = q_table[state, action] + learning_rate * (
            reward + discount_factor * np.max(q_table[new_state, :]) - q_table[state, action])
        state = new_state
        if done:
            break
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
num_test_episodes = 100
total_rewards = 0
for episode in range(num_test_episodes):
    state = env.reset()[0]
    done = False
    episode_rewards = 0
    for step in range(max_steps):
        action = np.argmax(q_table[state, :])  # Choose best action
        new_state, reward, done, truncated, info = env.step(action)
        episode_rewards += reward
        state = new_state
        if done:
            break
    total_rewards += episode_rewards
print(f"Average reward over {num_test_episodes} test episodes: {total_rewards / num_test_episodes}")
env.close()
