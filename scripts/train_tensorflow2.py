import tensorflow as tf
from tensorflow.keras import layers


class DDQN(tf.keras.Model):
    def __init__(self, state_size, action_size, layer_size):
        super(DDQN, self).__init__()
        #         self.input_shape = state_size
        self.action_size = action_size
        self.head_1 = layers.Dense(layer_size, activation="relu")
        self.ff_1 = layers.Dense(layer_size, activation="relu")
        self.ff_2 = layers.Dense(action_size)

    def call(self, inputs):
        x = self.head_1(inputs)
        x = tf.nn.relu(x)
        x = self.ff_1(x)
        x = tf.nn.relu(x)
        out = self.ff_2(x)

        return out


import tensorflow as tf
from tensorflow.keras import layers, optimizers, losses
import numpy as np


class CQLAgent:
    def __init__(self, state_size, action_size, hidden_size=256, device="cpu"):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.tau = 1e-3
        self.gamma = 0.99

        self.network = DDQN(
            state_size=self.state_size,
            action_size=self.action_size,
            layer_size=hidden_size,
        )

        self.target_net = DDQN(
            state_size=self.state_size,
            action_size=self.action_size,
            layer_size=hidden_size,
        )

        self.optimizer = optimizers.Adam(learning_rate=1e-3)

    def get_action(self, state, epsilon):
        if np.random.random() > epsilon:
            state = np.expand_dims(state, axis=0).astype(np.float32)
            self.network.trainable = False
            action_values = self.network(state)
            self.network.trainable = True
            action = np.argmax(action_values.numpy(), axis=1)
        else:
            action = np.random.choice(np.arange(self.action_size), size=1)
        return action

    def cql_loss(self, q_values, current_action):
        """Computes the CQL loss for a batch of Q-values and actions."""
        logsumexp = tf.reduce_logsumexp(q_values, axis=1, keepdims=True)
        q_a = tf.gather(q_values, current_action, axis=1, batch_dims=1)

        return tf.reduce_mean(logsumexp - q_a)

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        with tf.GradientTape() as tape:
            Q_targets_next = tf.reduce_max(
                self.target_net(next_states), axis=1, keepdims=True
            )
            Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

            Q_a_s = self.network(states)
            Q_expected = tf.gather(Q_a_s, actions, axis=1, batch_dims=1)

            cql1_loss = self.cql_loss(Q_a_s, actions)

            bellman_error = losses.mean_squared_error(Q_expected, Q_targets)

            q1_loss = cql1_loss + 0.5 * bellman_error

        gradients = tape.gradient(q1_loss, self.network.trainable_variables)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        self.optimizer.apply_gradients(
            zip(clipped_gradients, self.network.trainable_variables)
        )

        # ------------------- update target network ------------------- #
        self.soft_update(self.network, self.target_net)
        return (
            q1_loss.numpy(),
            cql1_loss.numpy(),
            bellman_error.numpy(),
        )

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(
            target_model.trainable_variables, local_model.trainable_variables
        ):
            target_param.assign(
                self.tau * local_param + (1.0 - self.tau) * target_param
            )


import tensorflow as tf
import numpy as np
from collections import deque, namedtuple
from tensorflow import data
import random


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"],
        )

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def add_batch(self, states, actions, rewards, next_states, dones):
        """Add a batch of experiences to memory."""
        for i in range(len(states)):
            self.add(states[i], actions[i], rewards[i], next_states[i], dones[i])

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = np.array([e.state for e in experiences if e is not None])
        actions = np.vstack([e.action for e in experiences if e is not None])
        rewards = np.vstack([e.reward for e in experiences if e is not None])
        next_states = np.array([e.next_state for e in experiences if e is not None])
        dones = np.vstack([e.done for e in experiences if e is not None])

        return (states, actions, rewards, next_states, dones)

    def create_dataset(self):
        """Create a Dataset from the replay buffer."""
        states = tf.convert_to_tensor([e.state for e in self.memory], dtype=tf.float32)
        actions = tf.convert_to_tensor([e.action for e in self.memory], dtype=tf.int32)
        rewards = tf.convert_to_tensor(
            [e.reward for e in self.memory], dtype=tf.float32
        )
        next_states = tf.convert_to_tensor(
            [e.next_state for e in self.memory], dtype=tf.float32
        )
        dones = tf.convert_to_tensor([e.done for e in self.memory], dtype=tf.int32)

        dataset = data.Dataset.from_tensor_slices(
            (states, actions, rewards, next_states, dones)
        )
        dataset = dataset.shuffle(len(self.memory)).batch(self.batch_size)

        return dataset

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


import gym

env = gym.make("CartPole-v1")

buffer = ReplayBuffer(buffer_size=1000, batch_size=32)

agent = CQLAgent(state_size=4, action_size=2, device="cpu")

for episode in range(300):
    # explore
    state, info = env.reset()
    for i in range(1000):
        action = agent.get_action(state, epsilon=0.1)[0]
        next_state, reward, done, _, info = env.step(action)
        buffer.add(state, action, reward, next_state, done)
        state = next_state
        if done:
            state, info = env.reset()

    # train
    for i in range(10):
        agent.learn(buffer.sample())

    # test
    test_return = 0
    state, info = env.reset()
    while True:
        action = agent.get_action(state, epsilon=0)[0]
        next_state, reward, done, _, info = env.step(action)
        state = next_state
        test_return += 1
        if done:
            break
    print("episode", episode, "test return:", test_return)
