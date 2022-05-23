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

import sys
from util import manhattanDistance, Counter, PriorityQueueWithFunction
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
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
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
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

        It is a development of a "simple response agent", calculated by locations of food and locations of ghosts.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # ghost cost :
        nearestGhost = sys.maxsize  # Initialized to maximum value to perform "reflexive alpha beta"
        for gState in newGhostStates:
            ghostX, ghostY = gState.getPosition()
            # I have no idea why pulling a "location" of a
            #       "ghost" returns a tuple rather than a state
            ghostX = int(ghostX)
            ghostY = int(ghostY)
            if gState.scaredTimer == 0:  # Can kill
                nearestGhost = min(nearestGhost, manhattanDistance((ghostX, ghostY), newPos))
        ghostCost = -(1.0 / (nearestGhost + 0.1))  # Balance to achieve a worthy result
        # 0.1 Is for not dividing by 0

        # Food cost:
        foodAsList = newFood.asList()
        # Using Queue from util.py
        distToFoodQueue = PriorityQueueWithFunction(lambda x: manhattanDistance(x, newPos))
        for food in foodAsList:
            distToFoodQueue.push(food)
        if distToFoodQueue.isEmpty():  # That means no more food
            nearestFood = 0
        else:
            # The distance to the closest because it is a priority queue with a function
            nearestFood = manhattanDistance(distToFoodQueue.pop(), newPos)
        foodCost = (1.0 / (nearestFood + 0.1))  # Balance to achieve a worthy result
        # 0.1 Is for not dividing by 0
        return successorGameState.getScore() + foodCost + ghostCost


def scoreEvaluationFunction(currentGameState):
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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
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
        # Start for the maximum agent
        return self.maxStep(gameState, 1)

    # Checks if the situation in our agent is inactive
    def isInactive(self, gameState):
        return gameState.isWin() or gameState.isLose() or self.depth == 0

    def maxStep(self, gameState, depth):
        if self.isInactive(gameState):  # Checks if the situation in our agent is inactive
            return self.evaluationFunction(gameState)
        maxVal = -sys.maxsize  # Initializes the maximum value (so-called beta)
        bestStep = Directions.STOP  # Initialized to stop by class directions in game.py
        for action in gameState.getLegalActions(0):
            # Performs on each successor a minimum step according to the algorithm
            successor = gameState.generateSuccessor(0, action)
            actionVal = self.minStep(successor, depth, 1)
            if actionVal > maxVal:  # He's better
                maxVal = actionVal
                bestStep = action
        if depth > 1:
            return maxVal  # A value will be returned in favor of a success calculation
        return bestStep  # An action will be returned, since this is the stage for execution

    def minStep(self, gameState, depth, agentIndex):
        if self.isInactive(gameState):  # Checks if the situation in our agent is inactive
            return self.evaluationFunction(gameState)
        minVal = sys.maxsize  # Initializes the minimum value (so-called alpha)
        legalActions = gameState.getLegalActions(agentIndex)
        successors = [gameState.generateSuccessor(agentIndex, action) for action in legalActions]
        if agentIndex == gameState.getNumAgents() - 1:
            if depth < self.depth:  # It will be the turn of the maximum agent
                for successor in successors:
                    minVal = min(minVal, self.maxStep(successor, depth + 1))
            else: # It will be the turn of the minimum agent
                for successor in successors:
                    minVal = min(minVal, self.evaluationFunction(successor))
        else: # Last agent
            for successor in successors:
                minVal = min(minVal, self.minStep(successor, depth, agentIndex + 1))
        return minVal # A value will be returned in favor of a success calculation


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # Start for the maximum agent
        return self.maxStep(gameState, 1, -sys.maxsize, sys.maxsize)

    # Checks if the situation in our agent is inactive
    def isInactive(self, gameState):
        return gameState.isWin() or gameState.isLose() or self.depth == 0

    def maxStep(self, gameState, depth, alpha, beta):
        if self.isInactive(gameState):  # Checks if the situation in our agent is inactive
            return self.evaluationFunction(gameState)
        maxVal = -sys.maxsize  # Initializes the maximum value (so-called beta)
        bestStep = Directions.STOP  # Initialized to stop by class directions in game.py
        for action in gameState.getLegalActions(0):
            # Performs on each successor a minimum step according to the algorithm
            successor = gameState.generateSuccessor(0, action)
            pruningValue = self.minStep(successor, depth, 1, alpha, beta)
            if pruningValue > maxVal:  # He's better
                maxVal = pruningValue
                bestStep = action
            if maxVal > beta: # Puring will act
                return maxVal
            alpha = max(alpha, maxVal)
        if depth > 1:
            return maxVal  # A value will be returned in favor of a success calculation
        return bestStep  # An action will be returned, since this is the stage for execution

    def minStep(self, gameState, depth, agentIndex, alpha, beta):
        if self.isInactive(gameState):  # Checks if the situation in our agent is inactive
            return self.evaluationFunction(gameState)
        minVal = sys.maxsize  # Initializes the minimum value (so-called alpha)
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            if agentIndex == gameState.getNumAgents() - 1:
                if depth < self.depth: # It will be the turn of the maximum agent
                    pruningValue = self.maxStep(successor, depth + 1, alpha, beta)
                else:  # It will be the turn of the minimum agent
                    pruningValue = self.evaluationFunction(successor)
            else:  # Last agent
                pruningValue = self.minStep(successor, depth, agentIndex + 1, alpha, beta)
            # puring check
            if pruningValue < minVal:
                minVal = pruningValue
            if minVal < alpha:
                return minVal
            beta = min(beta, minVal)
        return minVal # A value will be returned in favor of a success calculation


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # Start for the maximum agent
        return self.maxStep(gameState, 1)

    # Checks if the situation in our agent is inactive
    def isInactive(self, gameState):
        return gameState.isWin() or gameState.isLose() or self.depth == 0

    def maxStep(self, gameState, depth):
        if self.isInactive(gameState):  # Checks if the situation in our agent is inactive
            return self.evaluationFunction(gameState)
        maxVal = -sys.maxsize  # Initializes the maximum value (so-called beta)
        bestStep = Directions.STOP  # Initialized to stop by class directions in game.py
        for action in gameState.getLegalActions(0):
            # Performs on each successor a probabilistic minimum step according to the algorithm
            successor = gameState.generateSuccessor(0, action)
            probabilisticPuringValue = self.minStep(successor, depth, 1)
            # Pruning test in relation to the probabilistic value
            if probabilisticPuringValue > maxVal:
                maxVal = probabilisticPuringValue
                bestStep = action
        if depth > 1:
            return maxVal  # A value will be returned in favor of a success calculation
        return bestStep   # An action will be returned, since this is the stage for execution

    def minStep(self, gameState, depth, agentIndex):
        if self.isInactive(gameState):  # Checks if the situation in our agent is inactive
            return self.evaluationFunction(gameState)
        legalActions = gameState.getLegalActions(agentIndex)
        successors = [gameState.generateSuccessor(agentIndex, action) for action in legalActions]
        probabilityPuringValue = 0
        probabilityForMove = 1.0 / len(legalActions)
        if agentIndex == gameState.getNumAgents() - 1:  # Last Agent
            if depth < self.depth:
                for successor in successors: # check Using the max step
                    probabilityPuringValue += probabilityForMove * self.maxStep(successor, depth + 1)
            else:  # good time to evaluation Function
                for successor in successors:  # check Using the evaluationFunction
                    probabilityPuringValue += probabilityForMove * self.evaluationFunction(successor)
        else:  # not the last agent
            for successor in successors: # check Using the min step
                probabilityPuringValue += probabilityForMove * self.minStep(successor, depth, agentIndex + 1)
        return probabilityPuringValue


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: evaluate of state rate to win in game
    """
    "*** YOUR CODE HERE ***"

    # State score
    currentPosition = currentGameState.getPacmanPosition()
    # we will change the power of the state score power with a value of 1 to a power with a value of 1.5
    stateScore = 37 * currentGameState.getScore()

    # Capsules score
    capsQueue = PriorityQueueWithFunction(lambda x: manhattanDistance(currentPosition, x))
    capsCounter = Counter()
    for cap in currentGameState.data.capsules:
        capsCounter[str(cap)] = manhattanDistance(currentPosition, cap)
        capsQueue.push(cap)
    if capsQueue.isEmpty():
        contraPositiveNearestCapsDist = 100
    else:
        # For giving a positive result
        contraPositiveNearestCapsDist = 100 - manhattanDistance(currentPosition, capsQueue.pop())
    # using counter from util
    capsTotalDistant = capsCounter.totalCount()
    # For the State evaluate function to be good A change was made in the "factor" of the parameters
    capsScore = -8 * capsTotalDistant + 10 * contraPositiveNearestCapsDist

    # Ghost score
    ghostStates = currentGameState.getGhostStates()
    ghostQueue = PriorityQueueWithFunction(lambda x: manhattanDistance(currentPosition, x.getPosition()))
    ghostScaredCounter = Counter()
    for ghost in ghostStates:
        ghostQueue.push(ghost)
        ghostScaredCounter[str(ghost)] = ghost.scaredTimer
    nearestGhostDist = manhattanDistance(currentPosition, ghostQueue.pop().getPosition())
    # For the State evaluate function to be good A change was made in the "factor" of the parameters,
    # and factor multiplying between number of ghosts and number of capsules
    ghostScore = 3 * nearestGhostDist + 30 * ghostScaredCounter.totalCount() * capsQueue.count

    # Food score
    foods = currentGameState.getFood()
    foodsPos = foods.asList()
    numFood = currentGameState.getNumFood()
    foodQueue = PriorityQueueWithFunction(lambda x: manhattanDistance(currentPosition, x))
    for food in foodsPos:
        foodQueue.push(food)
    if foodQueue.isEmpty():
        distClosestFood = 0
    else:
        nearestFood = foodQueue.pop()
        distClosestFood = manhattanDistance(currentPosition, nearestFood)
    if numFood < nearestGhostDist:
        numFood = nearestGhostDist
    # For the State evaluate function to be good A change was made in the "factor" of the parameters
    foodScore = - 9 * distClosestFood - 10 * numFood

    # Total score
    return ghostScore + capsScore + stateScore + foodScore


# Abbreviation
better = betterEvaluationFunction
