# In deep reinforcement learning, we need to create an environment for the agent to learn in, and that's our snake game.
from snake_env import Snake

import random
import numpy as np
from keras import Sequential
'''
A Sequential deep learning model is appropraite for a plain stack of layers where each layer has exactly
one input tensor and one output tensor.
'''
from collections import deque
'''
Deques are a generalization of stacks and queues (the name is pronounced "deck" and is short for "double-ended queue").
Deques support thread-safe, memory efficient appends and pops from either side of the deque with approximately the
same O(1) performance in either direction.
'''
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
'''
Adam optimization is a stochastic gradient descent method that is based on adaptive estimation of first-order
and second order moments. Whereas momentum can be seen as a ball running down a slope, Adam behaves like a
heavy ball with friction, which thus prefers flat minima in the error surface.
'''
from plot_script import plot_result # We want to plot some stats about how the algorithm learned to play.

class DQN:
    '''
    applies a Q-learning algorithm to a neural network to teach an agent to play snake
    '''
    def __init__(self, env, params):

        self.action_space = env.action_space # the dimension of the action space (4 because up, down, left, right)
        self.state_space = env.state_space # the dimension of the state space (12 binary elements)
        self.epsilon = params['epsilon'] # the initial ratio of steps taken to randomly explore vs move in a predicted direction
        self.gamma = params['gamma'] # the discount factor for future rewards (0 is short-sighted, 1 is long-sighted)
        self.batch_size = params['batch_size'] # the number of time steps (i.e. states) to gather before submitting a batch for training
        self.epsilon_min = params['epsilon_min'] # the minimum ratio of time steps we'd like the agent to move randomly vs in a predicted direction
        self.epsilon_decay = params['epsilon_decay'] # how much of the ratio of random moving we want to take into the next iteration of gathering a batch of states
        self.learning_rate = params['learning_rate'] # to what extent newly acquired info overrides old info (0 learn nothing, exploit prior knowledge exclusively; 1 only consider the most recent information)
        self.layer_sizes = params['layer_sizes'] # The number of nodes for the hidden layers of our Q network.
        self.memory = deque(maxlen=2500) # This is our defined working memory array of the state of the agent and the environment.
        self.model = self.build_model()

    def build_model(self):
        '''
        builds a neural network of dense layers consisting of an input layer, 3 hidden layers, and an output layer
        '''
        model = Sequential()
        for i, layer_size in enumerate(self.layer_sizes):
            if i == 0: # The input layer's shape (i.e. number of nodes) is defined by the dimension of the state space.
                model.add(Dense(layer_size, input_shape=(self.state_space,), activation='relu'))
            else: # The three hidden layers will have an integer number of nodes.
                model.add(Dense(layer_size, activation='relu'))
                # Recall that the Rectified Linear Unit (ReLU) activation function that outputs the input
                # directly if the input is positive, otherwise it outputs zero.
        model.add(Dense(self.action_space, activation='softmax')) # The output layer's number of nodes is equal to the dimension of the action space.
        # Softmax function is good here because it's best applied to multi-class classification problems where
        # class membership is required on more than two class labels.
        # In this instance, maybe the snake needs to travel in 3 or more directions to get to the apple.
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model


    def remember(self, state, action, reward, next_state, done):
        '''
        adds the current state, next state, proposed action, total reward, and whether we are done in
        the agent's running memory buffer of states
        '''
        self.memory.append((state, action, reward, next_state, done))


    def act(self, state):
        '''
        moves in a random direction or the direction predicted to give the best reward outcome
        '''
        if np.random.rand() <= self.epsilon: # If we are under the explore threshold parameter,
            return random.randrange(self.action_space) # move in a random direction.
        # Otherwise, move in the direction which maximizes the probability of a large reward.
        act_values = self.model.predict(state) # e.g. [[0.08789534, 0.8699538 , 0.03103394, 0.01111698]]
                                               #           0:up      1:right      2:down      3:left
        return np.argmax(act_values[0])


    '''
    The batch size defines the number of samples that will be propagated through the network.
    For instance, let's say you have 1050 training samples and you want to set up a batch_size equal to 100.
        - The algorithm takes the first 100 samples (from 1st to 100th) from the training dataset and trains the network.
        - Next, it takes the second 100 samples (from 101st to 200th) and trains the network again.
        - We can keep doing this procedure until we have propagated all samples through of the network.
    Problems might arise with the last set of samples. In our example, we've used 1050 which is not divisible
    by 100 without remainder. The simplest fix here is just to get the final 50 samples and use them to train the network.
    '''
    def replay(self):
        if len(self.memory) < self.batch_size: # If we haven't conducted enough samples for a training batch,
            return # go collect more samples.

        # If we have enough samples for a learning batch...
        minibatch = random.sample(self.memory, self.batch_size) # Get a batch_size'd random sample from the working memory buffer.
        states = np.array([memory[0] for memory in minibatch])
        actions = np.array([memory[1] for memory in minibatch])
        rewards = np.array([memory[2] for memory in minibatch])
        next_states = np.array([memory[3] for memory in minibatch])
        dones = np.array([memory[4] for memory in minibatch])
        states = np.squeeze(states) # Convert the state vectors from 1x12 matrices to 12-element arrays.
        next_states = np.squeeze(next_states)

        # The core of this algorithm is a Bellman equation as a simple value iteration update,
        # using the weighted average of the old value and the new information.
        targets = rewards + self.gamma*(np.amax(self.model.predict_on_batch(next_states), axis=1))*(1-dones)
        targets_full = self.model.predict_on_batch(states)

        ind = np.arange(self.batch_size)
        targets_full[[ind], [actions]] = targets

        self.model.fit(states, targets_full, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min: # If our random exploration parameter is greater than the minimum
            self.epsilon *= self.epsilon_decay # attenuate it just a bit.

def train_dqn(num_episodes, env):
    total_rewards_history = []
    agent = DQN(env, params)
    for e in range(num_episodes):
        state = env.reset()
        state = np.reshape(state, (1, env.state_space)) # Convert the state to a 1x12 matrix.
        total_reward = 0
        max_steps = 10000
        for _ in range(max_steps):
            action = agent.act(state)
            prev_state = state
            next_state, reward, done = env.step(action)
            total_reward += reward
            next_state = np.reshape(next_state, (1, env.state_space))
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if params['batch_size'] > 1:
                agent.replay()
            if done:
                print(f'ending state: {str(prev_state)}')
                print(f'episode: {e+1}/{num_episodes}, total reward: {total_reward}')
                break
        total_rewards_history.append(total_reward)
    return total_rewards_history

if __name__ == '__main__':

    params = dict()
    params['name'] = None
    params['epsilon'] = 1
    params['gamma'] = .95
    params['batch_size'] = 500
    params['epsilon_min'] = .01
    params['epsilon_decay'] = .995
    params['learning_rate'] = 0.00025
    params['layer_sizes'] = [128, 128, 128]

    results = dict()
    num_episodes = 20

    # for batchsz in [1, 10, 100, 1000]:
    #     print(batchsz)
    #     params['batch_size'] = batchsz
    #     nm = ''
    #     params['name'] = f'Batchsize {batchsz}'
    # env_infos = {'States: only walls':{'state_space':'no body knowledge'},
    #              'States: direction 0 or 1':{'state_space':None},
    #              'States: coordinates':{'state_space':'coordinates'},
    #              'States: no direction':{'state_space':'no direction'}}
    # for key in env_infos.keys():
    #     params['name'] = key
    #     env_info = env_infos[key]
    #     print(env_info)
    #     env = Snake(env_info=env_info)

    env = Snake()
    total_rewards_history = train_dqn(num_episodes, env)

    results[params['name']] = total_rewards_history
    print(total_rewards_history)
    plot_result(results, direct=True, k=20)
