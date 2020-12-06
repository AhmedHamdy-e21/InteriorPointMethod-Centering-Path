'''
##########################################
The Algorithm:
##########################################

##########################################
## Step 1
##########################################
    - Write the problem on the Standard form. And add slack variables.
    - Formulate the problem and retrieve A,S,I,X,Atranspose.
##########################################
## Step 2
##########################################
    - choose initial point according to the reference ( Heuristic) to start the algorithm.
        - Initial points for vector x and s.
##########################################
## Step 3
##########################################
    - set the centering parameter mu.
    - set the step size alpha

##########################################
## Step 4
##########################################
    - calculate the duality measure using the specified centering paramter.

##########################################
## Step 5
##########################################
    - Add the duality measure and centering parameter in the system of equations before solving it.
    - Solve the System of equations (It's nonlinear be careful - I think I need to apply taylor expansion-).
    - For the next version of the code I need to implement the Chelosky factorization.
    - But for now I think I'll using Newton-Raphson method. 

##########################################
## Step 6
##########################################
    - update the x,y,s vector using the obtained deltas from the previous step

Set the stopping criteria to be xtranspose*s<tolerance ; tolerance = 0.0001 for example
Then iterate on the above using while loop maybe and track the values at the end in order to draw some conclusions using 
the number of iterations and function values. 
'''
import numpy as np
from numpy.linalg import inv as inv
from sympy import *


#### First Trial:-d
# Constraints paramters + slack variables
A=np.array([[1],[1],[1]])

b=np.array([6])

## Initialize values for initial point
x=np.array([[5],[6],[1]])

s=np.array([[1],[1],[1]])

## Set algorithm paramters
Sigma=0.5
alpha=0.9

StoppingCriteria=x.T@s
print(StoppingCriteria)

mu=StoppingCriteria/x.shape[0]  ## This is the duality measure
Tolerance=0.01
while StoppingCriteria<Tolerance:


    ## Solve the System of Equation using built-in function for Newton Raphson










    ### This will be calculated at each iteration
    StoppingCriteria=x.T@s
    mu=StoppingCriteria/x.shape[0]