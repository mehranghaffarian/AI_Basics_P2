import random

import util
from Agents import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def __init__(self, *args, **kwargs) -> None:
        self.index = 0  # your agent always has index 0

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        It takes a GameState and returns a tuple representing a position on the game board.
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions(self.index)

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        The evaluation function takes in the current and proposed successor
        GameStates (Game.py) and returns a number, where higher numbers are better.
        You can try and change this evaluation function if you want but it is not necessary.
        """
        nextGameState = currentGameState.generateSuccessor(self.index, action)
        return nextGameState.getScore(self.index) - currentGameState.getScore(self.index)


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    Every player's score is the number of pieces they have placed on the board.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore(0)


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxAgent, AlphaBetaAgent & ExpectiMaxAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (Agents.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2', **kwargs):
        self.index = 0  # your agent always has index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent which extends MultiAgentSearchAgent and is supposed to be implementing a minimax tree with a
    certain depth.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(**kwargs)

    def getAction(self, state):
        """
        Returns the minimax action using self.depth and self.evaluationFunction

        But before getting your hands dirty, look at these functions:

        gameState.isGameFinished() -> bool
        gameState.getNumAgents() -> int
        gameState.generateSuccessor(agentIndex, action) -> GameState
        gameState.getLegalActions(agentIndex) -> list
        self.evaluationFunction(gameState) -> float
        """
        "*** YOUR CODE HERE ***"
        root = Node(game_sequence=GameSequence([], state, self.evaluationFunction(state)))

        form_tree(root, self)

        def mini_max(curr_root, depth):
            if len(curr_root.children) == 0:
                curr_root.value = curr_root.game_sequence.final_point
                return curr_root

            target = curr_root.children[0]
            for c in curr_root.children:
                if c.value is None:
                    if len(c.children) == 0:
                        c.value = c.game_sequence.final_point
                    else:
                        c = mini_max(c, depth+1)
                if target.value is None:
                    if len(target.children) == 0:
                        target.value = target.game_sequence.final_point
                    else:
                        target = mini_max(target, depth+1)
                if depth % state.getNumAgents() == 0 and c.value > target.value:
                    target = c
                elif depth % state.getNumAgents() != 0 and c.value < target.value:
                    target = c
            return target

        return mini_max(root, 0).game_sequence.actions[0][1]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning. It is very similar to the MinimaxAgent but you need to implement the alpha-beta pruning algorithm too.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(**kwargs)

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction

        You should keep track of alpha and beta in each node to be able to implement alpha-beta pruning.
        """
        "*** YOUR CODE HERE ***"
        root = Node(game_sequence=GameSequence([], gameState, self.evaluationFunction(gameState)))
        form_tree(root, self)

        def alpha_beta_pruning(curr_root, depth):
            if len(curr_root.children) == 0:
                curr_root.value = curr_root.game_sequence.final_point
                return curr_root

            target = curr_root.children[0]
            for c in curr_root.children:
                if c.value is None:
                    if len(c.children) == 0:
                        c.value = c.game_sequence.final_point
                    else:
                        c.alpha = curr_root.alpha
                        c.beta = curr_root.beta
                        c = alpha_beta_pruning(c, depth+1)
                if target.value is None:
                    if len(target.children) == 0:
                        target.value = target.game_sequence.final_point
                    else:
                        target.alpha = curr_root.alpha
                        target.beta = curr_root.beta
                        target = alpha_beta_pruning(target, depth+1)
                if depth % gameState.getNumAgents() == 0:
                    if c.value > target.value:
                        target = c
                    if c.value > curr_root.beta:
                        c.alpha = curr_root.alpha
                        c.beta = curr_root.beta
                        return c
                    curr_root.alpha = max(curr_root.alpha, c.value)
                else:
                    if c.value < target.value:
                        target = c
                    if c.value > curr_root.alpha:
                        c.alpha = curr_root.alpha
                        c.beta = curr_root.beta
                        return c
                    curr_root.beta = min(curr_root.beta, c.value)

            target.alpha = curr_root.alpha
            target.beta = curr_root.beta
            return target

        return alpha_beta_pruning(root, 0).game_sequence.actions[0][1]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent which has a max node for your agent but every other node is a chance node.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(**kwargs)

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All opponents should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
    Your extreme evaluation function.

    You are asked to read the following paper on othello heuristics and extend it for two to four player rollit game.
    Implementing a good stability heuristic has extra points.
    Any other brilliant ideas are also accepted. Just try and be original.

    The paper: Sannidhanam, Vaishnavi, and Muthukaruppan Annamalai. "An analysis of heuristics in othello." (2015).

    Here are also some functions you will need to use:
    
    gameState.getPieces(index) -> list
    gameState.getCorners() -> 4-tuple
    gameState.getScore() -> list
    gameState.getScore(index) -> int

    """

    "*** YOUR CODE HERE ***"

    # parity

    # corners

    # mobility

    # stability

    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction


###########################################
### My own utils ###
###########################################

def get_target_index(elements, target):
    indices = [i for i in range(len(elements)) if elements[i] == target]
    return elements[random.choice(indices)]


class GameSequence:
    def __init__(self, actions, final_game_state, final_point, *args,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.actions = actions
        self.final_game_state = final_game_state
        self.final_point = final_point


class Node:
    def __init__(self, game_sequence, value=None, children=None, alpha=-64, beta=64, *args,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.game_sequence = game_sequence
        self.value = value
        if children is None:
            children = []
        self.children = children
        self.alpha = alpha
        self.beta = beta


def form_tree(root, agent):
    depth_count = 0
    leaves = [root]

    while depth_count < agent.depth:
        new_leaves = []
        for curr_node in leaves:
            gs = curr_node.game_sequence
            current_state = gs.final_game_state
            agent_index = depth_count % current_state.getNumAgents()

            for a in current_state.getLegalActions(agent_index):
                new_state = current_state.generateSuccessor(agent_index, a)
                new_actions = gs.actions.copy()
                new_actions.append((agent_index, a))
                new_node = Node(game_sequence=GameSequence(new_actions, new_state,
                                                           agent.evaluationFunction(new_state)))
                curr_node.children.append(new_node)
                new_leaves.append(new_node)
        leaves = new_leaves
        depth_count += 1
