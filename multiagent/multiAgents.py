# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = 0
        current_food = currentGameState.getFood().asList()
        new_pac_pos = newPos

        for ghostState in newGhostStates:
            new_ghost_pos = ghostState.getPosition()

            if new_pac_pos == new_ghost_pos:
                score -= 1000

            if new_pac_pos in current_food:
                score += 1

        food_distance = []
        for food in current_food:
            food_distance.append(abs(new_pac_pos[0] - food[0]) + abs(new_pac_pos[1] - food[1]))
        score += 1 / (min(food_distance) + 0.01)

        return score


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        legalMoves = gameState.getLegalActions(0)
        score = -99999
        action_idx = -1
        for i in range(len(legalMoves)):
            action = legalMoves[i]
            successor = gameState.generateSuccessor(0, action)
            value = self.value(successor, 1, 0)
            if value > score:
                action_idx = i
                score = value

        return legalMoves[action_idx]

    def value(self, gameState, index, depth):
        if gameState.isWin() or gameState.isLose() or (depth == self.depth and index == 0):
            return self.evaluationFunction(gameState)
        if index == 0:
            return self.max_value(gameState, index, depth)
        if index > 0:
            return self.min_value(gameState, index, depth)

    def max_value(self, gameState, index, depth):
        v = -99999
        actions = gameState.getLegalActions(index)
        for action in actions:
            successor = gameState.generateSuccessor(index, action)
            v = max(v, self.value(successor, index + 1, depth))
        return v

    def min_value(self, gameState, index, depth):
        v = 99999
        actions = gameState.getLegalActions(index)
        for action in actions:
            successor = gameState.generateSuccessor(index, action)
            if index + 1 == gameState.getNumAgents():
                v = min(v, self.value(successor, 0, depth + 1))
            else:
                v = min(v, self.value(successor, index + 1, depth))
        return v


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        legalMoves = gameState.getLegalActions(0)
        score = -99999
        action_idx = -1
        alpha = -99999
        beta = 99999
        for i in range(len(legalMoves)):
            action = legalMoves[i]
            successor = gameState.generateSuccessor(0, action)
            value = self.value(successor, 1, 0, alpha, beta)
            if value > score:
                action_idx = i
                alpha = score = value

        return legalMoves[action_idx]

    def value(self, gameState, index, depth, alpha, beta):
        if gameState.isWin() or gameState.isLose() or (depth == self.depth and index == 0):
            return self.evaluationFunction(gameState)
        if index == 0:
            return self.max_value(gameState, index, depth, alpha, beta)
        if index > 0:
            return self.min_value(gameState, index, depth, alpha, beta)

    def max_value(self, gameState, index, depth, alpha, beta):
        v = -99999
        actions = gameState.getLegalActions(index)
        for action in actions:
            successor = gameState.generateSuccessor(index, action)
            v = max(v, self.value(successor, index + 1, depth, alpha, beta))
            if v > beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_value(self, gameState, index, depth, alpha, beta):
        v = 99999
        actions = gameState.getLegalActions(index)
        for action in actions:
            successor = gameState.generateSuccessor(index, action)
            if index + 1 == gameState.getNumAgents():
                v = min(v, self.value(successor, 0, depth + 1, alpha, beta))
            else:
                v = min(v, self.value(successor, index + 1, depth, alpha, beta))
            if v < alpha:
                return v
            beta = min(beta, v)
        return v


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        legalMoves = gameState.getLegalActions(0)
        score = -99999
        action_idx = -1
        for i in range(len(legalMoves)):
            action = legalMoves[i]
            successor = gameState.generateSuccessor(0, action)
            value = self.value(successor, 1, 0)
            if value > score:
                action_idx = i
                score = value

        return legalMoves[action_idx]

    def value(self, gameState, index, depth):
        if gameState.isWin() or gameState.isLose() or (depth == self.depth and index == 0):
            return self.evaluationFunction(gameState)
        if index == 0:
            return self.max_value(gameState, index, depth)
        if index > 0:
            return self.exp_value(gameState, index, depth)

    def max_value(self, gameState, index, depth):
        v = -99999
        actions = gameState.getLegalActions(index)
        for action in actions:
            successor = gameState.generateSuccessor(index, action)
            v = max(v, self.value(successor, index + 1, depth))
        return v

    def exp_value(self, gameState, index, depth):
        v = 0
        actions = gameState.getLegalActions(index)
        # probabilities = [random.random() for _ in range(len(actions))]
        # total = sum(probabilities)
        # probabilities = [x / total for x in probabilities]
        probabilities = [1 / len(actions) for _ in range(len(actions))]
        for action, prob in zip(actions, probabilities):
            successor = gameState.generateSuccessor(index, action)
            if index + 1 == gameState.getNumAgents():
                v += prob * self.value(successor, 0, depth + 1)
            else:
                v += prob * self.value(successor, index + 1, depth)
        return v


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    (1)consider the distance between pacman and foods, less dis means higher score
    (2)consider the distance between pacman and ghost, more dis means higher score
    (3)consider the number of food, less food means higher score

    Then in the end, I plug into other functions within given, and it performs better
    However, the factor of each function is randomly attempted which I also don't know why
    I just tried from bottom up, and then get the full score, then I stop
    """
    "*** YOUR CODE HERE ***"
    ghostStates = currentGameState.getGhostStates()
    pac_pos = currentGameState.getPacmanPosition()
    food_count = currentGameState.getNumFood()

    score = 0

    food = currentGameState.getFood()
    foodsPos = food.asList()
    food_distance = []
    for food in foodsPos:
        fd = abs(food[0] - pac_pos[0]) + abs(food[1] - pac_pos[1])
        food_distance.append(fd)
    nearest_food = 0
    if len(foodsPos) > 0:
        nearest_food = min(food_distance)
    score -= nearest_food

    # evaluate the current state of the ghosts
    nearest_ghost = 99999
    for ghost in ghostStates:
        ghostPosition = ghost.getPosition()
        gd = abs(ghostPosition[0] - pac_pos[0]) + abs(ghostPosition[1] - pac_pos[1])

        # update the nearest ghost distance
        if ghost.scaredTimer == 0 and gd < nearest_ghost:
            nearest_ghost = gd

        # eat the ghost
        elif ghost.scaredTimer > gd:
            score += ghost.scaredTimer - gd

    score += 0.9*nearest_ghost
    score -= 5*food_count

    return score + len(currentGameState.getCapsules()) + 2*currentGameState.getScore()


# Abbreviation
better = betterEvaluationFunction
