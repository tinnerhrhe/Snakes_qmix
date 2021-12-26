import os
import random
import gym
import pylab
import numpy as np
from collections import deque
from keras.models import Model, load_model
from keras.layers import Input, Dense, Lambda, Add, Conv2D, Flatten
from keras.optimizers import Adam, RMSprop
from keras import backend as K
from PER import *
import cv2


def OurModel(input_shape, action_space, dueling):
    X_input = Input(input_shape)
    X = X_input

    X = Conv2D(64, 5, strides=(3, 3), padding="valid", input_shape=input_shape, activation="relu",
               data_format="channels_first")(X)
    X = Conv2D(64, 4, strides=(2, 2), padding="valid", activation="relu", data_format="channels_first")(X)
    X = Conv2D(64, 3, strides=(1, 1), padding="valid", activation="relu", data_format="channels_first")(X)
    X = Flatten()(X)
    # 'Dense' is the basic form of a neural network layer
    # Input Layer of state size(4) and Hidden Layer with 512 nodes
    X = Dense(512, activation="relu", kernel_initializer='he_uniform')(X)

    # Hidden layer with 256 nodes
    X = Dense(256, activation="relu", kernel_initializer='he_uniform')(X)

    # Hidden layer with 64 nodes
    X = Dense(64, activation="relu", kernel_initializer='he_uniform')(X)

    if dueling:
        state_value = Dense(1, kernel_initializer='he_uniform')(X)
        state_value = Lambda(lambda s: K.expand_dims(s[:, 0], -1), output_shape=(action_space,))(state_value)

        action_advantage = Dense(action_space, kernel_initializer='he_uniform')(X)
        action_advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), output_shape=(action_space,))(
            action_advantage)

        X = Add()([state_value, action_advantage])
    else:
        # Output Layer with # of actions: 2 nodes (left, right)
        X = Dense(action_space, activation="linear", kernel_initializer='he_uniform')(X)

    model = Model(inputs=X_input, outputs=X, name='CartPole PER D3QN CNN model')
    model.compile(loss="mean_squared_error", optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01),
                  metrics=["accuracy"])

    model.summary()
    return model


class DQNAgent:
    def __init__(self, env, args, buffer):
        self.env = env
        #self.env.seed(0)
        # by default, CartPole-v1 has max episode steps = 500
        # we can use this to experiment beyond 500
        self._max_episode_steps = 200
        self.state_size = 206
        self.action_size = 4
        self.EPISODES = 1000

        # Instantiate memory
        memory_size = 10000
        self.MEMORY = buffer
        self.memory = deque(maxlen=2000)

        self.gamma = 0.95  # discount rate

        # EXPLORATION HYPERPARAMETERS for epsilon and epsilon greedy strategy
        self.epsilon = 1.0  # exploration probability at start
        self.epsilon_min = 0.01  # minimum exploration probability
        self.epsilon_decay = 0.0005  # exponential decay rate for exploration prob

        self.batch_size = 32

        # defining model parameters
        self.ddqn = True  # use doudle deep q network
        self.Soft_Update = False  # use soft parameter update
        self.dueling = True  # use dealing netowrk
        self.epsilon_greedy = False  # use epsilon greedy strategy
        self.USE_PER = True  # use priority experienced replay

        self.TAU = 0.1  # target network soft update hyperparameter

        self.Save_Path = 'Models'
        if not os.path.exists(self.Save_Path): os.makedirs(self.Save_Path)
        self.scores, self.episodes, self.average = [], [], []

        self.Model_name = os.path.join(self.Save_Path, "_PER_D3QN_CNN.h5")

        self.ROWS = 160
        self.COLS = 240
        self.REM_STEP = 4

        self.image_memory = np.zeros((self.REM_STEP, self.ROWS, self.COLS))
        self.state_size = (self.REM_STEP, self.ROWS, self.COLS)

        # create main model and target model
        self.model = OurModel(input_shape=self.state_size, action_space=self.action_size, dueling=self.dueling)
        self.target_model = OurModel(input_shape=self.state_size, action_space=self.action_size, dueling=self.dueling)

        # after some time interval update the target model to be same with model

    def update_target_model(self):
        if not self.Soft_Update and self.ddqn:
            self.target_model.set_weights(self.model.get_weights())
            return
        if self.Soft_Update and self.ddqn:
            q_model_theta = self.model.get_weights()
            target_model_theta = self.target_model.get_weights()
            counter = 0
            for q_weight, target_weight in zip(q_model_theta, target_model_theta):
                target_weight = target_weight * (1 - self.TAU) + q_weight * self.TAU
                target_model_theta[counter] = target_weight
                counter += 1
            self.target_model.set_weights(target_model_theta)

    def remember(self, state, action, reward, next_state, done):
        experience = state, action, reward, next_state, done
        if self.USE_PER:
            self.MEMORY.store(experience)
        else:
            self.memory.append((experience))

    def act(self, state, decay_step):
        # EPSILON GREEDY STRATEGY
        if self.epsilon_greedy:
            # Here we'll use an improved version of our epsilon greedy strategy for Q-learning
            explore_probability = self.epsilon_min + (self.epsilon - self.epsilon_min) * np.exp(
                -self.epsilon_decay * decay_step)
        # OLD EPSILON STRATEGY
        else:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= (1 - self.epsilon_decay)
            explore_probability = self.epsilon

        if explore_probability > np.random.rand():
            # Make a random action (exploration)
            return random.randrange(self.action_size), explore_probability
        else:
            # Get action from Q-network (exploitation)
            # Estimate the Qs values state
            # Take the biggest Q value (= the best action)
            return np.argmax(self.model.predict(state)), explore_probability
