# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        for state in states:
            self.values[state] = 0

        k = 0
        while k < self.iterations:
            values = self.values.copy()
            for state in states:
                if self.mdp.isTerminal(state):
                    values[state] = 0
                else:
                    Q_values = []
                    actions = self.mdp.getPossibleActions(state)
                    for action in actions:
                        Q_value = self.computeQValueFromValues(state, action)
                        Q_values.append(Q_value)
                    values[state] = max(Q_values)
            self.values = values
            k += 1

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        Q_value = 0
        T_funcs = self.mdp.getTransitionStatesAndProbs(state, action)

        for T_func in T_funcs:
            R_func = self.mdp.getReward(state, action, T_func[0])
            V_func = self.getValue(T_func[0])
            Q_value += T_func[1] * (R_func + self.discount * V_func)
        return Q_value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = self.mdp.getPossibleActions(state)
        Q_value = -float('inf')
        action_star = None
        for action in actions:
            q = self.getQValue(state, action)
            if q > Q_value:
                Q_value = q
                action_star = action

        return action_star

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class PrioritizedSweepingValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        predecessors = {}
        for s in self.mdp.getStates():
            if not self.mdp.isTerminal(s):
                for action in self.mdp.getPossibleActions(s):
                    for T_func in self.mdp.getTransitionStatesAndProbs(s, action):
                        if T_func[0] in predecessors:
                            predecessors[T_func[0]].add(s)
                        else:
                            predecessors[T_func[0]] = {s}

        queue = util.PriorityQueue()
        for s in self.mdp.getStates():
            if not self.mdp.isTerminal(s):
                values = []
                for action in self.mdp.getPossibleActions(s):
                    Q_value = self.computeQValueFromValues(s, action)
                    values.append(Q_value)
                diff = abs(max(values) - self.values[s])
                queue.update(s, - diff)

        for i in range(self.iterations):
            if queue.isEmpty():
                break
            s = queue.pop()
            if not self.mdp.isTerminal(s):
                values = []
                for action in self.mdp.getPossibleActions(s):
                    Q_value = self.computeQValueFromValues(s, action)
                    values.append(Q_value)
                self.values[s] = max(values)

            for p in predecessors[s]:
                if not self.mdp.isTerminal(p):
                    values = []
                    for action in self.mdp.getPossibleActions(p):
                        Q_value = self.computeQValueFromValues(p, action)
                        values.append(Q_value)
                    diff = abs(max(values) - self.values[p])

                    if diff > self.theta:
                        queue.update(p, -diff)
