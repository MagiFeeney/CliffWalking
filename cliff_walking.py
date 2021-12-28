import numpy as np
import matplotlib.pyplot as plt
import gym

# Cliff Walking
## initialization
gamma = 0.9
learning_rate = 0.01
num_states = 48
num_actions = 4

# env setting
env = gym.make('CliffWalking-v0')

# episilon greedy policy
def episilon_greedy(q_values, state, episilon=0.1):
    if np.random.random() < episilon:
        return np.random.choice(4)
    else:
        return np.argmax(q_values[state])

# Q_learning
def Q_learning(initial_q_values, MaxEpisodes=500):
    q_values = initial_q_values
    episodic_reward = np.zeros(MaxEpisodes)
    for i in range(MaxEpisodes):
        state = env.reset()
        done = False

        while not done:
            action = episilon_greedy(q_values, state, episilon=0.1)
            next_state, reward, done, info = env.step(action)
            episodic_reward[i] += reward
            td_target = reward + gamma * np.max(q_values[next_state])
            td_error = td_target - q_values[state][action]
            q_values[state][action] += learning_rate * td_error
            state = next_state
    return episodic_reward 

# Sarsa
def Sarsa(initial_q_values, MaxEpisodes=500):
    q_values = initial_q_values
    episodic_reward = np.zeros(MaxEpisodes)
    for i in range(MaxEpisodes):
        state = env.reset()
        done = False

        while not done:
            action = episilon_greedy(q_values, state, episilon=0.1)
            next_state, reward, done, info = env.step(action)
            episodic_reward[i] += reward
            next_action = episilon_greedy(q_values, next_state, episilon=0.1)
            td_target = reward + gamma * q_values[next_state][next_action]
            td_error = td_target - q_values[state][action]
            q_values[state][action] += learning_rate * td_error
            state = next_state
            action = next_action
    return episodic_reward 
    

initial_q_values = np.zeros((num_states, num_actions))
# sum of reward of Q_learning
qlearning_reward = Q_learning(initial_q_values)
# sum of reward of SARSA
sarsa_reward = Sarsa(initial_q_values)

plt.plot(list(range(1, len(qlearning_reward) + 1)), qlearning_reward, '-', color = 'red')
plt.plot(list(range(1, len(sarsa_reward) + 1)), sarsa_reward, '-', color = 'blue')
plt.show()