import numpy as np
import random
import matplotlib.pyplot as plt

# Environment setup:
# 2 roads, each can have 0, 1, 2, ..., max_cars cars waiting
max_cars = 4

# Q-table dimensions: (states = all combinations of waiting cars on both roads)
state_space = (max_cars + 1) ** 2
action_space = 2  # 0 = green for road A, 1 = green for road B
q_table = np.zeros((state_space, action_space))

# Hyperparameters
num_episodes = 1000
max_steps = 100
alpha = 0.1        # learning rate
gamma = 0.9        # discount factor
epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.005

# Helper function to encode the state as a single number
def encode_state(cars_a, cars_b):
    return cars_a * (max_cars + 1) + cars_b

# Helper function to decode the state from number back to car counts
def decode_state(state):
    return state // (max_cars + 1), state % (max_cars + 1)

rewards_per_episode = []

# Training loop
for episode in range(num_episodes):
    # Start with random car counts on each road
    cars_a = np.random.randint(0, max_cars + 1)
    cars_b = np.random.randint(0, max_cars + 1)
    state = encode_state(cars_a, cars_b)
    total_reward = 0

    for step in range(max_steps):
        # Decide whether to explore or exploit
        if random.uniform(0, 1) < epsilon:
            action = random.randint(0, 1)  # explore
        else:
            action = np.argmax(q_table[state])  # exploit

        # Simulate environment behavior
        if action == 0:
            # Green for road A
            cars_a = max(0, cars_a - 1)
            cars_b = min(max_cars, cars_b + np.random.randint(0, 2))  # new car may arrive
        else:
            # Green for road B
            cars_b = max(0, cars_b - 1)
            cars_a = min(max_cars, cars_a + np.random.randint(0, 2))

        new_state = encode_state(cars_a, cars_b)
        reward = - (cars_a + cars_b)  # the fewer cars waiting, the better (reward is negative waiting time)

        # Update Q-table using the Q-learning formula
        q_table[state, action] = q_table[state, action] + alpha * (reward + gamma * np.max(q_table[new_state]) - q_table[state, action])

        state = new_state
        total_reward += reward

    # Decay epsilon over time
    epsilon = min_epsilon + (1.0 - min_epsilon) * np.exp(-decay_rate * episode)
    rewards_per_episode.append(total_reward)

# After training
print("Training completed!\n")
print("Final Q-table:")
print(q_table)

# Plotting
plt.plot(rewards_per_episode)
plt.xlabel("Episode")
plt.ylabel("Total Negative Waiting Time")
plt.title("Learning Progress")
plt.show()
