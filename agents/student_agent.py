# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys


@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
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
        # dummy return
        """
        plan:
        tree implementation:
        dictionary : we would have to map each move to a key - we can use strings?
            have multiple dictionaries, one for the values, one for the list of children
        
        could also make a nested class to make code more readable
        simulating a node (default policy) - just make random moves, - fast simulations
        
        tree policy - choose an epsilon value (we can use 0.15 for now and change later)
            1-epsilon of the time, we choose node with best winrate(if n/a then choose random node)
            epsilon of the time, we choose random node
        max height of the tree? 10 for now
        how many simulations? 100-200 simulations total?
        how many total times do we expand nodes? 50?
        """
        return my_pos, self.dir_map["u"]
    class MCTNode():
        def __init__(self, board, parent=None, parent_action=None):
            # nested list
            self.board = board
            self.parent = parent
            self.parent_action = parent_action
            # all possible actions from the current node
            self.children = []
            self.number_of_visits = 0
            self.results = dict()
            self.results[1] = 0
            self.results[-1] = 0
            # list of all possible actions
            self.untrird_actions = None
            self.untrird_actions = self.untried_actions()
            return
        def untried_actions(self):
            """
            Returns the list of untried actions from a given state. For the first turn of our game there
            are 81 possible actions. For the second turn it is 8 or 9. This varies in our game.
            """
            self.untrird_actions = self.state.get_legal_actions()
            return self.untrird_actions
        def q(self):
            """
            Returns the difference of wins - losses
            """
            wins = self.results[1]
            loses = self.results[-1]
            return wins - loses
        def n(self):
            """
            Returns the number of times each node is visited.
            """
            return self.number_of_visits
        def expand(self):
            """
            From the present state, next state is generated depending on the action which is carried out.
            In this step all the possible child nodes corresponding to generated states are appended to
            the children array and the child_node is returned. The states which are possible from the 
            present state are all generated and the child_node corresponding to this generated state is returned.
            """
            action = self.untrird_actions.pop()
            next_state = self.state.move(action)
            child_node = MCTNode(next_state, parent=self, parent_action=action)
            self.children.append(child_node)
            return child_node
        def is_terminal_node(self):
            """
            This is used to check if the current node is terminal or not. Terminal node is reached when the game is over.
            """
            return self.state.is_game_over()

        def rollout(self):
            """
            From the current state, entire game is simulated till there is an outcome for the game.
            This outcome of the game is returned. For example if it results in a win, the outcome is 1. 
            Otherwise it is -1 if it results in a loss. And it is 0 if it is a tie. If the entire game is
            randomly simulated, that is at each turn the move is randomly selected out of set of possible 
            moves, it is called light playout.
            """
            current_rollout_state = self.state
            while not current_rollout_state.is_game_over():
                possible_moves = current_rollout_state.get_legal_actions()
                action = self.rollout_policy(possible_moves)
                current_rollout_state = current_rollout_state.move(action)
            return current_rollout_state.game_result()
        def backpropagate(self, result):
            """
            In this step all the statistics for the nodes are updated.
            Untill the parent node is reached, the number of visits for each node is incremented by 1. 
            If the result is 1, that is it resulted in a win, then the win is incremented by 1. 
            Otherwise if result is a loss, then loss is incremented by 1.
            """
            self.number_of_visits += 1.
            self.results[result] += 1.
            if self.parent:
                self.parent.backpropagate(result)
        def is_fully_expanded(self):
            """
            All the actions are poped out of untrird_actions one by one. 
            When it becomes empty, that is when the size is zero, it is fully expanded.
            """
            return len(self.untrird_actions) == 0
        def best_child(self, c_param=0.1):
            """
            Once fully expanded, this function selects the best child out of the children array.
            The first term in the formula corresponds to exploitation and the second term corresponds to exploration.
            """
            choices_weights = [(c.q() / c.n()) + c_param * np.sqrt((2 * np.log(self.n()) / c.n())) for c in self.children]
            return self.children[np.argmax(choices_weights)]
        def rollout_policy(self, possible_moves):
            """
            Randomly selects a move out of possible moves. This is an example of random playout.
            """
            return possible_moves[np.random.randint(len(possible_moves))]
        def _tree_policy(self):
            """
            Selects node to run rollout.
            """
            current_node = self
            while not current_node.is_terminal_node():
                
                if not current_node.is_fully_expanded():
                    return current_node.expand()
                else:
                    current_node = current_node.best_child()
            return current_node
        def best_action(self):
            """
            This is the best action function which returns the node corresponding to best possible move. 
            The step of expansion, simulation and backpropagation are carried out by the code above.
            """
            simulation_no = 100
            for i in range(simulation_no):
                v = self._tree_policy()
                reward = v.rollout()
                v.backpropagate(reward)
            return self.best_child(c_param=0.)
        def get_legal_actions(self): 
            '''
            Modify according to your game or
            needs. Constructs a list of all
            possible actions from current state.
            Returns a list.
            '''
            pass
        def is_game_over(self):
            '''
            Modify according to your game or 
            needs. It is the game over condition
            and depends on your game. Returns
            true or false
            '''
            pass
        def game_result(self):
            '''
            Modify according to your game or 
            needs. Returns 1 or 0 or -1 depending
            on your state corresponding to win,
            tie or a loss.
            '''
            pass
        def move(self,action):
            '''
            Modify according to your game or 
            needs. Changes the state of your 
            board with a new value. For a normal
            Tic Tac Toe game, it can be a 3 by 3
            array with all the elements of array
            being 0 initially. 0 means the board 
            position is empty. If you place x in
            row 2 column 3, then it would be some 
            thing like board[2][3] = 1, where 1
            represents that x is placed. Returns 
            the new state after making a move.
            '''
            pass
        def main():
            """
            This is the main() function. Initialize the root node and call the best_action function to get the best node.
             This is not a member function of the class. All the other functions are member function of the class.
            """
            root = MTCNode(state = initial_state)
            selected_node = root.best_action()
            return 
