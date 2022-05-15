import random
from collections import deque

import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


class Agent:
    def __init__(
        self, state_size, action_size, epsilon_initial, epsilon_decay, epsilon_minimun
    ):
        """Implementation of a DQN Agent, which uses Fully Connected layers to approximate the agent's Q-table.

        Args:
            state_size (int): environment state space size
            action_size (int): environment action space size
            epsilon_initial (float): initial exploration factor value
            epsilon_decay (float): exploration factor value decay
            epsilon_minimun (float): minimum exploration factor value
        """

        # Model
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)

        # Reinforcement Learning Hyperparameters
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon_initial = epsilon_initial
        self.epsilon_decay = epsilon_decay
        self.epsilon_minimun = epsilon_minimun
        self.batch_size = 64
        self.train_start = 1000

        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

    def build_model(self):
        """Build and initialize the model.

        Returns:
            keras.models.Sequential: model that in charge of learning the Q-table
        """

        model = Sequential()
        model.add(
            Dense(
                24,
                input_dim=self.state_size,
                activation="relu",
                kernel_initializer="he_uniform",
            )
        )
        model.add(Dense(24, activation="relu", kernel_initializer="he_uniform"))
        model.add(
            Dense(
                self.action_size, activation="linear", kernel_initializer="he_uniform"
            )
        )
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))

        return model

    def load_model(self, path):
        """Load the model weights from the given path.

        Args:
            path (string): path to the saved model weights
        """

        self.model.load_weights(path)

    def update_target_model(self):
        """Updates the weights of the target model."""

        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state):
        """Get the next action using epsilon-greedy policy

        Args:
            state (numpy.array): current state

        Returns:
            int: action chosen by the agent
        """

        if np.random.rand() <= self.epsilon_initial:
            return random.randrange(self.action_size)

        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])

    def append_sample(self, state, action, reward, next_state, done):
        """Add the sample <s, a, r, s',d> to the replay memory.

        Args:
            state (numpy.array): initial state
            action (int): action chosen by the agent
            reward (float): receive read
            next_state (_type_): resulting state
            done (bool): environment finished
        """

        self.memory.append((state, action, reward, next_state, done))

        if self.epsilon_initial > self.epsilon_minimun:
            self.epsilon_initial *= self.epsilon_decay

    def train_model(self):
        """Trains the model by pick random samples from the replay memory, according to batch_size."""

        if len(self.memory) < self.train_start:
            return

        # Decide the minibatch size
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        # Inputs for model and target model
        update_input = np.zeros((batch_size, self.state_size))
        update_target = np.zeros((batch_size, self.state_size))
        action, reward, done = [], [], []

        # Get the values for each sample
        for sample in range(self.batch_size):
            update_input[sample] = mini_batch[sample][0]
            action.append(mini_batch[sample][1])
            reward.append(mini_batch[sample][2])
            update_target[sample] = mini_batch[sample][3]
            done.append(mini_batch[sample][4])

        # Run the models
        target = self.model.predict(update_input)
        target_val = self.target_model.predict(update_target)

        # Get target values
        for sample in range(self.batch_size):
            if done[sample]:
                target[sample][action[sample]] = reward[sample]
            else:
                target[sample][action[sample]] = reward[
                    sample
                ] + self.discount_factor * (np.amax(target_val[sample]))

        # Train the model
        self.model.fit(
            update_input, target, batch_size=self.batch_size, epochs=1, verbose=0
        )
