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


#### First Trial:-





