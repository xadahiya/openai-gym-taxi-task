import numpy as np
from collections import defaultdict


class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))

    def gen_e_greedy_probabilities(self, Q, state, nA, epsilon=1):
        """Generate Epsilon greedy probabilities for a state s."""
        if state in Q.keys():
            probs = [epsilon / nA for i in range(nA)]
            argmax = np.argmax(Q[state])
            probs[argmax] += (1 - epsilon)
        else:
            probs = [1 / nA for i in range(nA)]

        return probs

    def select_action(self, state, epsilon=1):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        probs = self.gen_e_greedy_probabilities(
            self.Q, state, self.nA, epsilon)
        return np.random.choice(self.nA, p=probs)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """

        alpha = 0.1
        gamma = 1

        expected_probs = self.gen_e_greedy_probabilities(
            self.Q, next_state, self.nA, 0.005)
        expected_a_value = np.dot(expected_probs, self.Q[next_state])
        self.Q[state][action] = (
            1 - alpha) * self.Q[state][action] + alpha * (
            reward + gamma * expected_a_value)
