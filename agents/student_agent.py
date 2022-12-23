# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy

@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """
    # Moves (Up, Right, Down, Left)
    moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
    opposites = {0: 2, 1: 3, 2: 0, 3: 1}
    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.autoplay = True
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }

    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """
        root = StudentAgent.MCTNode(chess_board, my_pos, None, adv_pos, max_step, None)
        selected_node = root.best_action()
        #self.dir_map["u"]
        return selected_node.my_pos, selected_node.d


    class MCTNode():
        def __init__(self, board, my_pos, d, adv_pos, max_step, parent=None):
            # nested list
            self.board = board
            self.board_size = len(board)
            self.parent = parent
            self.my_pos = my_pos
            # direction of the wall
            self.d = d
            self.adv_pos = adv_pos
            self.children = []
            self.number_of_visits = 0
            self.results = dict()
            # win
            self.results[1] = 0
            # lose
            self.results[-1] = 0
            # tie
            self.results[0] = 0
            self.max_step = max_step
            # list of all possible actions
            self.untried_actions = self.get_untried_actions()

        def check_endgame(self, board, my_pos, adv_pos):
            """
            Checks if the game ends and compute the current score of the agents.
            Returns is_endgame : bool, player_score : int
            """
            # Union-Find
            father = dict()
            for r in range(self.board_size):
                for c in range(self.board_size):
                    father[(r, c)] = (r, c)
    
            def find(pos):
                if father[pos] != pos:
                    father[pos] = find(father[pos])
                return father[pos]
    
            def union(pos1, pos2):
                father[pos1] = pos2
    
            for r in range(self.board_size):
                for c in range(self.board_size):
                    for d, move in enumerate(StudentAgent.moves[1:3]):  
                        # Only check down and right
                        if board[r][c][d+1]:
                            continue
                        pos_a = find((r, c))
                        pos_b = find((r + move[0], c + move[1]))
                        if pos_a != pos_b:
                            union(pos_a, pos_b)
    
            for r in range(self.board_size):
                for c in range(self.board_size):
                    find((r, c))
            p0_r = find(tuple(my_pos))
            p1_r = find(tuple(adv_pos))
            p0_score = list(father.values()).count(p0_r)
            p1_score = list(father.values()).count(p1_r)
            if p0_r == p1_r:
                return False, 0
            # tie
            res = 0
            if p0_score > p1_score:
                # win
                res = 1
            elif p0_score < p1_score:
                # lose
                res = -1
            return True, res
    
        def best_action(self):
            """
            Returns the node corresponding to best possible move. 
            The step of expansion, simulation and backpropagation will be triggered.
            """
            # tot_no_of_simulations == no_of_expansions * no_of_simulations
            no_of_expansions = 7 #CHANGE
            no_of_simulations = 1
            for i in range(no_of_expansions):
                best_node = self.tree_policy()
                for j in range(no_of_simulations):
                    # callig default method means when the game is over, return the reward
                    reward = best_node.default()
                    best_node.backpropagate(reward)
            return self.best_child(c_param=0.)

        def get_untried_actions(self):
            """
            Returns the list of untried actions from a given board.
            """
            self.untried_actions = self.get_legal_actions(self.my_pos, self.adv_pos)
            return self.untried_actions

        def q(self):
            """Returns the difference of wins - losses"""
            return self.results[1] - self.results[-1]

        def n(self):
            """ Returns the number of times each node is visited."""
            return self.number_of_visits

        def expand(self):
            """
            From the present board, next board is generated depending on the action which is carried out.
            In this step all the possible child nodes corresponding to generated boards are appended to
            the children array and the child_node is returned. The boards which are possible from the 
            present board are all generated and the child_node corresponding to this generated board is returned.
            """
            board = deepcopy(self.board)
            action = self.untried_actions.pop()
            my_pos, d = action[0], action[1]
            self.move(action, board)
            
            child_node = StudentAgent.MCTNode(board, my_pos, d, self.adv_pos, self.max_step, parent=self)
            self.children.append(child_node)
            return child_node

        def is_terminal_node(self):
            """
            This is used to check if the current node is terminal or not. 
            Terminal node is reached when the game is over.
            """
            return self.check_endgame(self.board, self.my_pos, self.adv_pos)[0]

        def default(self):
            """
            From the current board, entire game is simulated till there is an outcome for the game.
            This outcome of the game is returned. For example if it results in a win, the outcome is 1. 
            Otherwise it is -1 if it results in a loss. And it is 0 if it is a tie. If the entire game is
            randomly simulated, that is at each turn the move is randomly selected out of set of possible 
            moves, it is called light playout.
            """
            current_default_board = deepcopy(self.board)
            is_end, res = self.check_endgame(current_default_board, self.my_pos, self.adv_pos)
            my_pos = deepcopy(self.my_pos)
            adv_pos = deepcopy(self.my_pos)
            is_my_turn = True
            
            while not is_end:
                curr_pos = my_pos if is_my_turn else adv_pos
                opp_pos = adv_pos if is_my_turn else my_pos
                # get possible moves for the current player
                possible_moves = self.get_legal_actions(curr_pos, opp_pos)
                # choose random move
                random_action = self.default_policy(possible_moves)
                if is_my_turn:
                    my_pos = random_action[0]
                else:
                    adv_pos = random_action[0]
                # move changes board state
                self.move(random_action, current_default_board)
                is_end, res = self.check_endgame(current_default_board, my_pos, adv_pos)
                # alternate players
                is_my_turn = not is_my_turn
            return res

        def backpropagate(self, result):
            """
            Updates the states of the nodes untill the root is reached.
            result could be -1 for lose or 0 for tie or 1 for win.
            """
            self.number_of_visits += 1
            self.results[result] += 1
            if self.parent:
                self.parent.backpropagate(result)

        def is_fully_expanded(self):
            """
            All the actions are poped out of untried_actions one by one. 
            When it becomes empty, it is fully expanded.
            """
            return len(self.untried_actions) == 0

        def best_child(self, c_param=0.1):
            """
            Once fully expanded, it selects the best child out of the children array.
            The first term in the formula corresponds to exploitation and 
            the second term corresponds to exploration.
            """
            choices_weights = []
            for child in self.children:
                tmp = (child.q() / child.n()) + c_param * ((2 * np.log(self.n()) / child.n())) ** 0.5
                choices_weights.append(tmp)
            return self.children[np.argmax(choices_weights)]

        def default_policy(self, possible_moves):
            """
            Returns a move out of possible moves randomly.
            """
            return possible_moves[np.random.randint(len(possible_moves))]

        def tree_policy(self):
            """
            Returns a node using tree policy.
            """
            current_node = self
            # is_terminal_node uses check_end_game method to check if the game is over at current node.
            while not current_node.is_terminal_node():                
                # is_fully_expanded checks if the untried_actions list is empty
                if not current_node.is_fully_expanded():
                    # expand returns a child node.
                    return current_node.expand()
                else:
                    current_node = current_node.best_child()
            return current_node

        def get_legal_actions(self, my_pos, adv_pos): 
            '''
            Constructs a list of all possible actions from current board.
            Returns a list of tuple of pos, dir
            '''
            my_r, my_c = my_pos
            visited = [[False] * len(self.board) for i in range(len(self.board))]
            visited[adv_pos[0]][adv_pos[1]] = True
            legal_actions = []
            legal_actions = self.get_legal_actions_rec(legal_actions, self.max_step, my_r, my_c, visited)
            return legal_actions

        def get_legal_actions_rec(self, legal_actions, number_of_steps, r, c, visited):
            # if we can still continue further then we recurse
            visited[r][c] = True
            if not number_of_steps == 0:
                # explore each direction that doesn't have a wall
                if r + 1 < self.board_size and not self.board[r][c][2] and not visited[r+1][c]:
                    self.get_legal_actions_rec(legal_actions, number_of_steps - 1, r + 1, c, visited)
                if c + 1 < self.board_size and not self.board[r][c][1] and not visited[r][c+1]:
                    self.get_legal_actions_rec(legal_actions, number_of_steps - 1, r, c + 1, visited)
                if r - 1 >= 0 and not self.board[r][c][0] and not visited[r-1][c]:
                    self.get_legal_actions_rec(legal_actions, number_of_steps - 1, r - 1, c, visited)
                if c - 1 >= 0 and not self.board[r][c][3] and not visited[r][c-1]:
                    self.get_legal_actions_rec(legal_actions, number_of_steps - 1, r, c - 1, visited)
            # add all possible wall directions
            for d in range(0,4):
                if not self.board[r][c][d]:
                    legal_actions.append(((r, c), d))
            return legal_actions
        
        def move(self, action, board):
            '''
            Changes the board of your board with a new value.
            '''
            # action is a tuple of position, direction
            my_pos, d = action[0], action[1]
            my_r, my_c = my_pos[0], my_pos[1]
            # create a wall in the board
            board[my_r][my_c][d] = True
            # absence of the two lines below was a source of errors.
            move = StudentAgent.moves[d]
            board[my_r+move[0], my_c+move[1], StudentAgent.opposites[d]] = True
