# Traffic Signal Control using Reinforcement Learning (Q-Learning)

This project demonstrates a simplified example of how reinforcement learning, specifically Q-learning, can be used to control traffic signals in an intersection to reduce vehicle waiting times.

## 🧠 What is Reinforcement Learning?

Reinforcement Learning (RL) is a type of machine learning where an agent learns how to behave in an environment by performing actions and receiving rewards. The goal is to maximize the total reward over time.

In our case:
- **Agent** = traffic signal controller
- **Environment** = the traffic scenario (number of cars waiting on roads)
- **Action** = which road gets the green light
- **Reward** = negative of total cars waiting (we want to reduce it)

## 💡 Problem Setup

We simulate a 2-way intersection:
- Two roads: Road A and Road B.
- At each time step, the agent (traffic light) must decide whether to give green to Road A or Road B.
- Cars keep arriving randomly.
- The goal is to learn a policy that minimizes the total number of waiting cars.

## 📦 Requirements

- Python 3
- NumPy
- Matplotlib

Install dependencies:

```bash
pip install numpy matplotlib
# traffic-signal-control-q-learning
