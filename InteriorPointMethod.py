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

def UpdateValues( x,y,s,AllDeltas):
    """
    docstring
    """
    x=x+alpha*AllDeltas[:x.shape[0]]
    y=y+alpha*AllDeltas[x.shape[0]:x.shape[0]+y.shape[0]]
    s=s+alpha*AllDeltas[x.shape[0]+y.shape[0]:x.shape[0]+s.shape[0]+y.shape[0]]
    Xmat=Matricize(x)
    Smat=Matricize(s)
    return x,y,s,Xmat,Smat


def Matricize( VectorToMatrix):
    """
    docstring
    """
    Mat=np.diagflat(VectorToMatrix)
    return Mat
    
def GenerateAugmentedSystem(Zero1, A,Identity, Zero2,Zero3,Smat, Zero4,Xmat):
    """
    docstring
    """
    AugmentedSubystem1=np.concatenate((Zero1, A.T,Identity), axis=1)
    AugmentedSubystem2=np.concatenate((A, Zero2,Zero3), axis=1)
    AugmentedSubystem3=np.concatenate((Smat, Zero4,Xmat), axis=1)
    AugmentedSystem=np.concatenate((AugmentedSubystem1,AugmentedSubystem2,AugmentedSubystem3),axis=0)
    return AugmentedSystem



def InitializeZerosAndIdentities(A,s,x):
    """
    docstring
    """
    Zero1=np.zeros((A.shape[1],A.shape[1])) ## This for the most left upper zeros in the augmented system
    Identity=np.eye(A.shape[1]) ## This is for the most right upper identity in the augmented system
    Zero2=np.zeros((1,1)) ## This is hard coded for now, but I need to generate it automatically later on
    Zero3=np.zeros((1,x.shape[0]))
    Zero4=np.zeros((s.shape[0],1))
    return Zero1,Identity,Zero2,Zero3,Zero4



def GeneratAugmentedB(A,b,c,s,x,y, Xmat):
    """
    docstring
    """
    rXSe=Xmat@Smat@np.ones((Xmat.shape[0],1))
    rMu=mu*Sigma*np.ones((Xmat.shape[0],1))
    rc=A.T@y+s-c
    rb=A@x-b
    rLast=-rXSe+rMu
    AugmentedB=np.concatenate((-rc,-rb,rLast),axis=0)
    return AugmentedB

#### First Trial:-d
# Constraints paramters + slack variables
A=np.array([[1,1,1]])
b=np.array([6])
## Initialize values for initial point 
## I tried multiple arbitrary initial points and it's working awesome ^_^ 
x=np.array([[3],[3],[3]]) 
s=np.array([[1],[1],[1]])  ## This is initialized as identity
y=np.ones((1,1))
Xmat=Matricize(x)
Smat=Matricize(s)
c=np.array([[-1.1],[1],[0]])
################# 
 ### This is the y vector but I didn't initialize it before, I THINK THE DIMENSION IS 1 HERE ACCORDING TO THE NUMBER OF CONSTRANTS

## Set algorithm paramters
Sigma=0.5
alpha=0.7
StoppingCriteria=x.T@s
print(StoppingCriteria[0])
Tolerance=0.01
### I'll solve the augmented system first and in the next version I'll implement Cholesky Factorization
Zero1,Identity,Zero2,Zero3,Zero4=InitializeZerosAndIdentities(A,s,x)
AugmentedSystem=GenerateAugmentedSystem(Zero1, A,Identity, Zero2,Zero3,Smat, Zero4,Xmat)
i=0

while StoppingCriteria[0]>Tolerance:
    i=i+1
    print(i)
    mu=StoppingCriteria/x.shape[0]
    AugmentedSystem=GenerateAugmentedSystem(Zero1, A,Identity, Zero2,Zero3,Smat, Zero4,Xmat)
    AugmentedB=GeneratAugmentedB(A,b,c,s,x,y, Xmat) ## This is not so corrected you need to initialize zeros and the 
    ## last values will be -XSe+ mu*sigma*e Because this is solution.
    ## But in terms of the deltas, they will come from the function return.
    ## Anyways I need to proceed of implying the solving function.
    AllDeltas = np.linalg.solve(AugmentedSystem,AugmentedB)
#####################################################################################################
## Solve the system and Update Vecttor Values
#####################################################################################################
    x,y,s,Xmat,Smat=UpdateValues(x,y,s,AllDeltas)
    StoppingCriteria=x.T@s
    mu=StoppingCriteria/x.shape[0]
    print('Vector x is \n',x,'\n Vector s is \n',s,'\n mu is ',mu)