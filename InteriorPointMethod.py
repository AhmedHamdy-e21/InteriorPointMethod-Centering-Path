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

##### Newton Raphson method 
def newton(f,x0,epsilon,max_iter):
    '''
    Approximate solution of f(x)=0 by Newton's method.

    Parameters
    ----------
    f : function
        Function for which we are searching for a solution f(x)=0.
    Df : function
        Derivative of f(x).
    x0 : number
        Initial guess for a solution f(x)=0.
    epsilon : number
        Stopping criteria is abs(f(x)) < epsilon.
    max_iter : integer
        Maximum number of iterations of Newton's method.

    Returns
    -------
    xn : number
        Implement Newton's method: compute the linear approximation
        of f(x) at xn and find x intercept by the formula
            x = xn - f(xn)/Df(xn)
        Continue until abs(f(xn)) < epsilon and return xn.
        If Df(xn) == 0, return None. If the number of iterations
        exceeds max_iter, then return None.

    Examples
    --------
    >>> f = lambda x: x**2 - x - 1
    >>> Df = lambda x: 2*x - 1
    >>> newton(f,Df,1,1e-8,10)
    Found solution after 5 iterations.
    1.618033988749989
    '''

    xn = x0

    Df = f.diff(x)

    for n in range(0,max_iter):
        fxn = f(xn)
        if abs(fxn) < epsilon:
            print('Found solution after',n,'iterations.')
            return xn
        Dfxn = Df(xn)
        if Dfxn == 0:
            print('Zero derivative. No solution found.')
            return None
        xn = xn - fxn/Dfxn
    print('Exceeded maximum iterations. No solution found.')
    return None

### Function to evaluate the derivative using sympy
# define what is the variable
x = symbols('x')
# define the function
f = x**2-4*x-5
# find the first derivative
fderivative = f.diff(x)
fderivative


#### Get the value from here
# fderivative.evalf(subs= {x:0})
####


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