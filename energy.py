# -*- coding: utf-8 -*-

"""# Imports of modules

Below we import some modules necessary to make this code work properly. 

You can add other modules here which you might need for your algorithm. 
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import random
from scipy.spatial import distance

"""# Definitions of constants

In the cell below we define some constants that we need in the meta-heuristic. 

The paths to the data files are also specified here. 
"""

# ILS parameters 

ALPHA = 0.02
MAX_ITER = 100     # maximum number of iterations (Stopping criterion of ILS)
MAX_ITER_NI = 20   # number of iterations without improvement of the objective function (Stopping criterion of ILS)
MAX_ITER_LS = 10  # maximum number of iterations of the local search operator (Outer loop)
MAX_SWAPS = 1      # maximum number of swaps of the local search operator (Inner loop)
NB_SWAPS = 3       # number of swaps in the perturbation operator

# Path to data file

INPUT_DATA = "data/InputDataEnergySmallInstance.xlsx"  # Small instance
#INPUT_DATA = "data/InputDataEnergyLargeInstance.xlsx"  # Large instance

"""# Functions to load and prepare the data

In the cell below, you find three functions:
- first, read_excel_data, which returns the values of a specific sheet in an Excel file,
- then, a function to calculate the length of an eadge A--B in the network based on the coordinates of nodes A and B,
- and finally, create_data_model which fills the data dictionary with the data from the various sheets of the Excel file, as well as some dependent data, which is calculated from the raw data.
"""

def read_excel_data(filename, sheet_name):
    data = pd.read_excel(filename, sheet_name=sheet_name, header=None)
    values = data.values
    return values


def edgesLengthCalculation(NodesCord):
    EdgesLength = np.zeros((len(NodesCord), len(NodesCord)))

    for i in range(0, len(NodesCord)):
        for j in range(0, len(NodesCord)):
            EdgesLength[i, j] = distance.euclidean((NodesCord[i]), (NodesCord[j]))
    return EdgesLength


def create_data_model():
    data = {}
    
    # This section contains all required data #
    data['SourceNum'] = read_excel_data(INPUT_DATA, "SourceNum")[0][0]
    
    # Nodes Cordinate Read From Excel#
    data['NodesCord'] = read_excel_data(INPUT_DATA, "NodesCord")
    
    # Building Cordinate Read From Excel#
    #EdgesDemand = read_excel_data(INPUT_DATA, "EdgesDemand")
    
    # Fixed Unit Cost
    data['FixedUnitCost'] = read_excel_data(INPUT_DATA, "FixedUnitCost")
    
    # Edges Length Calculation based the Nodes Cordinate
    data['EdgesLength'] = np.matrix.round(edgesLengthCalculation(data['NodesCord']))
    
    # Fixed Instalation cost
    data['cfix'] = data['EdgesLength'] * data['FixedUnitCost']
    
    # Number of Nodes
    data['node_num'] = len(data['NodesCord'])
    
    # Revenue of Fulfiling the Edges Demand
    data['crev'] = read_excel_data(INPUT_DATA, "crev(cijrev)")
    
    # Penalty of Unmet Demand
    data['pumd'] = read_excel_data(INPUT_DATA, "pumd(pijumd)")
    
    # Cost of Heat Production in the Sources
    data['cheat'] = read_excel_data(INPUT_DATA, "cheat(ciheat)")
    
    # Variable Cost of Heat Transferring through the Edges
    data['cvar'] = read_excel_data(INPUT_DATA, "cvar(cijvar)")
    
    # Variable Thermal Losses
    data['vvar'] = read_excel_data(INPUT_DATA, "vvar(thetaijvar)")
    # Reshape vvar because data is read in wrong, only for small instance
    if(INPUT_DATA == "data/InputDataEnergySmallInstance.xlsx"):
        temp = np.zeros((8,8))
        for i in range(7):
            for j in range(7):
                temp[i][j] = data['vvar'][i][j]
        data['vvar'] = temp

    # Fixed Thermal Losses
    data['vfix'] = read_excel_data(INPUT_DATA, "vfix(thetaijfix)")
    
    # Full Load Hours for the Sources
    data['Tflh'] = read_excel_data(INPUT_DATA, "Tflh(Tiflh)")
    
    # Concurrence Effect
    data['Betta'] = read_excel_data(INPUT_DATA, "Betta")
    
    # Connect Quota
    data['Lambda'] = read_excel_data(INPUT_DATA, "Lambda")
    
    # Annuity Factor for Investment Cost
    data['Alpha'] = read_excel_data(INPUT_DATA, "Alpha")
    if(INPUT_DATA == "data/InputDataEnergySmallInstance.xlsx"):
        data['Alpha'] = data['Alpha'][0][0]
    
    # Edges Peak Demand
    data['EdgesDemandPeak'] = read_excel_data(INPUT_DATA, "EdgesDemandPeak(dij)")
    
    # Edges Annual Demand
    data['EdgesDemandAnnual'] = read_excel_data(INPUT_DATA, "EdgesDemandAnnual(Dij)")
    
    # Edges Maximum Capacity
    data['Cmax'] = read_excel_data(INPUT_DATA, "Cmax(cijmax)")
    
    # Cost of Maintenance
    data['com'] = read_excel_data(INPUT_DATA, "com(cijom)")
    
    # Source Maximum Capacity
    data['SourceMaxCap'] = read_excel_data(INPUT_DATA, "SourceMaxCap(Qimax)")
    
    # Dependent Parameters
    data['kfix'] = data['cfix'] * data['Alpha'] * data['EdgesLength'] + data['com'] * data['EdgesLength']
    data['kvar'] = data['cvar'] * data['EdgesLength'] * data['Alpha']
    data['rheat'] = data['crev'] * data['EdgesDemandAnnual'] * data['Lambda']
    data['kheat'] = (data['Tflh'] * data['cheat']) / data['Betta']
    data['Etta'] = 1 - data['EdgesLength'] * data['vvar']
    data['Delta'] = data['EdgesDemandPeak'] * data['Betta'] * data['Lambda'] + data['EdgesLength'] * data['vfix']
    
    print("1",data['kfix'])
    print("2",data['kvar'])
    print("3",data['cfix'])
    print("4",data['Alpha'])
    print("5",data['EdgesLength'])
    print("6",data['com'])
    print("7",data['EdgesLength'])

    return data

"""# Functions to calculate the objective function from the solution representation

The cell below contains 2 functions to calculate the objective function of an individual: 
- first `prufer_to_tree` which transforms the Prüfer representation of a solution into a tree, 
- second, `compute_of` which calculates the fitness (or objective function) of an individual (or a solution).
"""

def prufer_to_tree(a):
    """Transform the Prüfer representation into a tree."""
    tree = []
    t = range(0, len(a)+2)

    # the degree of each node is how many times it appears in the sequence
    deg = [1]*len(t)
    for i in a:
        deg[i] += 1

    # for each node label i in a, find the first node j with degree 1 and add the edge (j, i) to the tree
    for i in a:
        for j in t:
            if deg[j] == 1:
                tree.append((i, j))
                # decrement the degrees of i and j
                deg[i] -= 1
                deg[j] -= 1
                break

    last = [x for x in t if deg[x] == 1]
    tree.append((last[0], last[1]))

    return tree


def compute_of(individual, data):
    """Calculate the objective function of the individual."""
    tree_edges = prufer_to_tree(individual)
    
    graph = nx.Graph(tree_edges)
    all_pairs_path = dict(nx.all_pairs_shortest_path(graph))
    path_from_source = all_pairs_path[data['SourceNum']]

    P_in = np.zeros((len(tree_edges)+1, len(tree_edges)+1))
    P_out = np.zeros((len(tree_edges)+1, len(tree_edges)+1))

    hubs = np.unique(individual)
    spokes = list(set([i for i in range(len(individual)+2)]) - set(hubs))

    for i in spokes:
        A = path_from_source[i]
        e = 0
        for k in range(0, len(A)-1):
            if e == 0:
                P_out[A[len(A)-k-2], A[len(A)-k-1]] = 0
                e = e + 1
            else:
                P_out[A[len(A)-k-2], A[len(A)-k-1]] = sum(P_in[A[len(A)-k-1]])

            P_in[A[len(A)-k-2], A[len(A)-k-1]] = (P_out[A[len(A)-k-2], A[len(A)-k-1]] + data['Delta'][A[len(A)-k-2], A[len(A)-k-1]])/data['Etta'][A[len(A)-k-2], A[len(A)-k-1]]

    fitness = 0
    metDemand = 0
    for i in range(len(tree_edges)):
        fitness = fitness + data['kfix'][tree_edges[i]] - data['rheat'][tree_edges[i]] + data['kvar'][tree_edges[i]] * (P_in[tree_edges[i][0], tree_edges[i][1]])
        metDemand = metDemand + 2 * data['EdgesDemandAnnual'][tree_edges[i]] * data['pumd'][tree_edges[i]]
    fitness = fitness + data['kheat'][data['SourceNum']] * sum(P_in[data['SourceNum']])
    fitness = fitness + 0.5*((data['EdgesDemandAnnual'] * data['pumd']).sum() - metDemand)

    return fitness

"""# Functions to create solutions or individuals

The cell below contains two functions regarding individuals:

- first, `generate_individual` to create a random individual, 
- second, `initial_solution` which returns this single randomly generated individual and its fitness.
"""

def generate_individual(node_num):
    """Generate a random individual."""
    individual = np.ndarray.tolist(np.random.randint(0, high=node_num-1, size=node_num-2, dtype='int'))
    return individual


def initial_solution(data):
    """Generate a random solution and calculate its fitness."""
    solution = []

    # here we are generating only one initial solution
    solution.append(generate_individual(data['node_num']))

    value_of = compute_of(solution[0], data)

    return solution, value_of

"""# Functions for the local search

Below you can find functions to perform a local search: 
- first the general high-level `local_search` function,
- second the `swap` function, which implements a swap operator, 
- third the `swap_neighborhood` function which generates the neighborhood based on the swap operator,
- and finally the `unique_pairs` function, used by `swap_neighborhood`, wich generates unique pairs indexes. 
"""

# This function is for the local search operator
def local_search(of, sol, data):

    """Perform a local search."""
    
    # number of iterations local search
    
    nb_iterations = 0

    best_of = of
    best_sol = sol

    # Main loop local search
    # Local search operators is repeated MAX_ITER_LS times
    
    while nb_iterations <= MAX_ITER_LS:

        nb_iterations += 1
        print("Local: ", nb_iterations)
        # use an operator to perform local search
        #TODO local search operator
        temp_of, temp_sol=swap_neighborhood(best_sol, best_of, data)
        if (temp_of < best_of):
            best_of = temp_of
            best_sol = temp_sol
        else:
            continue
    return best_of, best_sol

# The following function is a sub-function to do a single swap move on the given solution "parent" (i.e., changing the hubs at positions p1 and p2)

def swap(p1, p2, parent):

    """Swap operator."""
    #TODO swap
    K=list(parent[p1:p2])
    ## Swap the points at the indices
    swap_part=K[::-1]
    return parent[:p1] + swap_part + parent[p2:]

# The following function is a function to generate the neighbours of the given solution "sol"
# NOTE: A single swap will create a neighbour
# All pairs of possible swap moves are investigated

def swap_neighborhood(sol, best_of, data):
    """Neighborhood generation with a swap operator."""
    #TODO swap_neighborhood
    best_sol=[]
    for i,j in unique_pairs(len(sol)):
        n_sol=swap(i, j, sol)
        n_of = compute_of(n_sol, data)
        if n_of < best_of:
            best_of = n_of
            best_sol= n_sol
        else:
            continue      
    of = best_of
    n= best_sol
    return of, n


def unique_pairs(n):
    """Produce pairs of indexes in range(n)"""
    for i in range(n):
        for j in range(i + 1, n):
            yield i, j

"""# Functions for the perturbation of solutions"""

# This function is a sub-function to do a given number of random swaps
def random_swap(sol):

    """Random perturbation."""
    
    # ----> Put your code here <---
    #TODO random swap
    return sol

# This function is the main body of the perturbation operator, wherein the random_swap function is called
def perturb(sol, data):
    #TODO perturbation
    pert_sol=random_swap(sol)
    pert_of = compute_of(pert_sol, data)
    return pert_of, pert_sol

"""# Main

"""

if __name__ == "__main__":

    # *************************Initialisation***************************
    # initialise the data
    data = create_data_model()
    
    # init number of iterations
    nb_iterations = 0

    # find initial solution (just one) with a constructive heuristic
    best_sol, best_of = initial_solution(data)

    # ********************************************************************

    print("Random solution")
    print("Initial objective function value:", best_of)
    print("Solution:", best_sol)

    # **************************Local Search******************************

    best_of, best_sol = local_search(best_of, best_sol[0], data)
    best_known = best_sol
    best_of_known = best_of

    print("\nLocal Search")
    print("Objective function value:", best_of)
    print("Tour:", best_sol)

    best_solution = prufer_to_tree(best_sol)

    print(best_solution)

    # ********************************************************************

    # ******************************ILS***********************************
    flag_continue = True
    improve = 0

    while flag_continue and nb_iterations <= MAX_ITER and improve <= MAX_ITER_NI:

        nb_iterations += 1
        print("Global: ", nb_iterations)
        # ******************Perturbation**********************************
        pert_of, pert_sol = perturb(best_sol, data)
        # print(pert_of)

        # ******************Local Search***********************************
        best_of_pert, best_sol_pert = local_search(pert_of, pert_sol, data)
        # print(best_of_pert)

        # ******************Aceptance criterion***************************
        if(best_of_pert < best_of_known):
            best_known = best_sol_pert
            best_of_known = best_of_pert
            improve = 0
        else:
            improve += 1

        if (best_of_pert < best_of * (1 + ALPHA)):
            best_of = best_of_pert
            best_sol = best_sol_pert
        else:
            flag_continue = False

    print("\n")
    print("After", nb_iterations, " ILS iterations, the best solution is:")
    print(best_known)
    print("with total cost:", best_of_known)

    best_solution = prufer_to_tree(best_known)

    graph = nx.Graph(best_solution)
    plt.figure(2)
    nx.draw(graph, with_labels=True)
    plt.show()