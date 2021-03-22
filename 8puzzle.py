import math
import time

"""
An implementation of A*-Search in 8-Puzzle using Python.
Author: Mirza Baig
Year: 2020
"""

class Node:
    """
    Class to represent a puzzle configuration.
    """
    def __init__(self, parent, data, g, h, f):
        """
        Constructor to initialize a puzzle configuration.
        :param parent: the parent node of the current node
        :param data: the values of the puzzle as a list
        :param g: the depth of the node from the root
        :param h: the heuristic value of this node
        :param f: the cost of this node
        """
        self.parent = parent
        self.data = data
        self.g = g
        self.h = h
        self.f = f

# Dictionary of index and co-ordinates of 8-puzzle board
co_ords = {
    0 : (0,0),
    1 : (0,1),
    2 : (0,2),
    3 : (1,0),
    4 : (1,1),
    5 : (1,2),
    6 : (2,0),
    7 : (2,1),
    8 : (2,2)
}

def validate_input(start, goal):
    """
    Validates the entered start and goal state for
    an 8-puzzle game.
    :param start: the start configuration
    :param goal: the goal configuration
    :return: true if valid input state, else false
    """
    if len(start) != 9 or len(goal) != 9: # Only 8-Puzzle board allowed
        print("Incorrect state space length.")
        return False

    state_dict = {}
    for value in start:
        if value == "9":
            print("Value '9' out of bound.") # Value 9 is not in 8-Puzzle
            return False
        if not value.isdigit():
            print("Non-integer in state space.")
            return False
        if value in state_dict: # Check for repeated values
            print("Repeated value in state space.")
            return False
        state_dict[value] = 1

    for value in goal: # Check goal if is permutation of start
        if value not in state_dict:
            print("Goal state space does not match start state space.")
            return False
        state_dict[value] -= 1

    if "0" not in state_dict: # Check if one blank cell is present
        print("No empty cell in state space.")
        return False

    return True

def inversions(state):
    """
    Returns the number of inversions for a state.
    :param state: the node state configuration
    :return inversion_counter: number of state inversions
    """
    state_copy = state.copy()
    state_copy.remove(0) # Copy state to remove blank for inversion calculation
    inversion_counter = 0
    for index_one in range(len(state_copy)): # Iterate each element
        for index_two in range(index_one, len(state_copy)): # Check all succeding values
            if state_copy[index_one] > state_copy[index_two]:
                inversion_counter += 1 # Largrer succeding values increments counter
    return inversion_counter

def manhattan_distance(state, goal):
    """
    Calculates the manhattan distance between two nodes.
    :param state: the start state
    :param goal: the goal state
    :return hval: the manhattan distance of the state
    """
    hval = 0
    for index, value in enumerate(state):
        if value == 0: # Underestimate by excluding calculation of the blank tile
            continue
        abs_x = abs((co_ords[index])[0] - (co_ords[goal.index(value)])[0])
        abs_y = abs((co_ords[index])[1] - (co_ords[goal.index(value)])[1])
        hval += abs_x + abs_y
    return hval

def eucledian_distance(state, goal):
    """
    Calculates the eucledian distance between two node.
    :param state: the start state
    :param goal: the goal state
    :return hval: the eucledian distance of the state
    """
    hval = 0
    for index, value in enumerate(state):
        if value == 0: # Underestimate by excluding calculation of the blank tile
            continue
        sqr_x = ((co_ords[index])[0] - (co_ords[goal.index(value)])[0])**2
        sqr_y = ((co_ords[index])[1] - (co_ords[goal.index(value)])[1])**2
        hval += math.sqrt(sqr_x + sqr_y)
    return hval

def apply_move(cell, x, y):
    """
    Calculates the new co-ordinates of a cell by
    applying an (x, y) shift.
    :param cell: the cell to shift
    :param x: the x shift vector
    :param y: the y shift vector
    :return (x2, y2): the new co-ordinate of the cell
    """
    x2 = (co_ords[cell])[0] + x
    y2 = (co_ords[cell])[1] + y
    return (x2, y2)

def trace_path(node):
    """
    Recursively prints a trace of the path from node
    by cycling through parent nodes up to root node.
    :param node: the node path to trace
    :return:
    """
    if node.parent == None:
        print("---------")
        print("Move ",node.g)
        print(node.data[:3])
        print(node.data[3:6])
        print(node.data[6:9])
        return

    trace_path(node.parent)
    print("---------")
    print("Move ",node.g)
    print(node.data[:3])
    print(node.data[3:6])
    print(node.data[6:9])
    return

def children_nodes(state, goal, heuristic):
    """
    Generates a list of a children nodes for the
    current node, state.
    :param state: the current node state
    :param goal: the target node state
    :return children: the list of all children nodes
    """
    children = []
    blank_index = state.data.index(0)
    blank_coord = co_ords[blank_index]
    move_vector = [(1,0), (0,1), (-1,0), (0,-1)] # Movement vectors D, R, U, L

    for move in move_vector:
        cur_state = state.data.copy() # Copy the current state of the node
        x, y = move[0], move[1] # Get movement co-ordinates
        new_blank_coord = apply_move(blank_index, x, y) # Get the new position of the blank cell
        x2, y2 = new_blank_coord[0], new_blank_coord[1]

        if (x2 > 2 or y2 > 2) or (x2 < 0 or y2 < 0): # If out of bound move, skip this move
            continue

        for index, coord in co_ords.items(): # Get the new index of the blank cell
            if coord == new_blank_coord:
                new_blank_index = index # The index the blank cell will swap to

        temp_cell = cur_state[new_blank_index] # Get the new blank cell index' current value
        cur_state[new_blank_index] = 0 # Place blank index in new cell
        cur_state[blank_index] = temp_cell # Put temporary cell into old blank index

        if heuristic == 0:
            hval = manhattan_distance(cur_state, goal) # Calculate the heuristic value of the start node
        else:
            hval = eucledian_distance(cur_state, goal)
        new_node = Node(state, cur_state, state.g+1, hval, hval+state.g+1) # Create the child node object
        children.append(new_node)
    return children

def astar_search(start, goal, heuristic):
    """
    Performs an A* search for the goal node.
    :param start: the starting configuration
    :param goal: the goal configuration to search for
    :return len(closed): the number of nodes explored
    """
    open = []
    closed = []

    expanded = {} # Dictionary to check if a node has been expanded
    explored = {} # Dictionary to check if a node has been explored

    if heuristic == 0:
        hval = manhattan_distance(start, goal) # Calculate the heuristic value of the start node
    else:
        hval = eucledian_distance(start, goal)

    start_node = Node(None, start, 0, hval, hval+0) # Create the new node - root node so no parent
    open.append(start_node)
    expanded[tuple(start_node.data)] = True # Mark node as being expanded

    while len(open) != 0:
        open.sort(key=lambda x: x.f, reverse=False) # Sort f-values of nodes in ascending order
        cur_node = open[0] # Select the node with smallest f_value
        closed.append(cur_node) # Add the current node to closed as it being explored
        explored[tuple(cur_node.data)] = True # Mark node as being explored
        open.pop(0) # Remove current node from open list

        if cur_node.h == 0: # Checks if current node is a goal node
            trace_path(cur_node)
            return len(closed)

        children = children_nodes(cur_node, goal, heuristic) # Get a list of all children nodes
        for child in children:
            if child.h == 0: # If child node is a goal node then return a trace to root node
                trace_path(child)
                return len(closed)

            if tuple(child.data) not in expanded and tuple(child.data) not in explored:
                open.append(child) # Only expand node if not already expanded and explored
                expanded[tuple(child.data)] = True
    return False

def main():
    """
    Main function to execute program.
    """
    start = input("Enter the starting state: ").replace(" ", "") # Remove whitespace from states
    goal = input("Enter the goal state: ").replace(" ", "")
    if validate_input(start, goal) == False:
        return print("Invalid input.")

    start_list = [] # Convert into list for efficieny
    for i in start:
        start_list.append(int(i))

    goal_list = [] # Convert into list for efficieny
    for i in goal:
        goal_list.append(int(i))

    state_inversions = inversions(start_list)
    print("\nStart state inversions:", state_inversions)
    goal_inversions = inversions(goal_list)
    print("Goal state inversions:", goal_inversions)

    if state_inversions % 2 != goal_inversions % 2: # If polarity is uneven, not solvable
        print("Unsolvable configurations.\nPolarities do not match.")
        exit(1)

    heuristic = -1 # Set heuristic option for A* search
    while heuristic != 0 and heuristic != 1:
        try:
            heuristic = int(input("\nEnter heuristic function: \n0 for Manhattan\n1 for Eucledian\n= "))
        except (EOFError, KeyboardInterrupt):
            print('Terminating.')
            exit()
        except (KeyError, ValueError):
            print('Invalid option.')

    start_time = time.time() # Start search time
    nodes_searched = astar_search(start_list, goal_list, heuristic)
    end_time = time.time() # Stop search timer

    print("\nSolved in: {:.2f}".format(end_time-start_time), "seconds")
    print("Nodes explored:", nodes_searched)
    exit()

if __name__ == "__main__":
    main()
