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
import matplotlib.pyplot as plt

def UpdateValues( x,y,s,AllDeltas,alpha):
    """
    docstring
    """
    x=x+alpha*AllDeltas[:x.shape[0]]
    y=y+alpha*AllDeltas[x.shape[0]:x.shape[0]+y.shape[0]]
    s=s+alpha*AllDeltas[x.shape[0]+y.shape[0]:x.shape[0]+s.shape[0]+y.shape[0]]
    Xmat=Matricize(x)
    Smat=Matricize(s)
    return x,y,s,Xmat,Smat

def ObjectiveFunction1(x1,x2):
    """
    docstring
    """
    return -1.1*x1-x2
def ObjectiveFunction3(x1,x2):
    """
    docstring
    """
    return -5*x1-4*x2

def ObjectiveFunction2(x1,x2):
    """
    docstring
    """
    return -30*x1-20*x2


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
    Zero2=np.zeros((A.shape[0],A.shape[0])) ## This is hard coded for now, but I need to generate it automatically later on
    Zero3=np.zeros((A.shape[0],x.shape[0]))
    Zero4=np.zeros((s.shape[0],A.shape[0]))
    return Zero1,Identity,Zero2,Zero3,Zero4

def GeneratAugmentedB(A,b,c,s,x,y, Xmat,Smat,mu,Sigma):
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

def GeneratAugmentedBAffine(A,b,c,s,x,y, Xmat,Smat):

    """
    docstring
    """
    rXSe=Xmat@Smat@np.ones((Xmat.shape[0],1))
    rc=A.T@y+s-c
    rb=A@x-b
    rLast=-rXSe
    AugmentedB=np.concatenate((-rc,-rb,rLast),axis=0)
    return AugmentedB

def Iterate( A,b,c,s,x,y,Zero1,Identity, Zero2,Zero3,Smat, Zero4,Xmat,Sigma,alpha):
    """
    docstring
    """
    StoppingCriteria=x.T@s
    mu=StoppingCriteria/x.shape[0]
    AugmentedSystem=GenerateAugmentedSystem(Zero1, A,Identity, Zero2,Zero3,Smat, Zero4,Xmat)
    AugmentedB=GeneratAugmentedB(A,b,c,s,x,y, Xmat,Smat,mu,Sigma)
    AllDeltas = np.linalg.solve(AugmentedSystem,AugmentedB)
    x,y,s,Xmat,Smat=UpdateValues(x,y,s,AllDeltas,alpha)
    return A,b,c,s,x,y,Xmat,Smat,mu,StoppingCriteria

def IterateAffine( A,b,c,s,x,y,Zero1,Identity, Zero2,Zero3,Smat, Zero4,Xmat,Sigma,alpha):
    """
    docstring
    """
    StoppingCriteria=x.T@s
    mu=StoppingCriteria/x.shape[0]
    AugmentedSystem=GenerateAugmentedSystem(Zero1, A,Identity, Zero2,Zero3,Smat, Zero4,Xmat)
    AugmentedB=GeneratAugmentedBAffine(A,b,c,s,x,y, Xmat,Smat)
    AllDeltas = np.linalg.solve(AugmentedSystem,AugmentedB)
    x,y,s,Xmat,Smat=UpdateValues(x,y,s,AllDeltas,alpha)
    return b,c,s,x,y,Xmat,Smat,mu,StoppingCriteria,AllDeltas

def Plot(ihist,OFhist,X,Y,i):
    """
    docstring
    """
    fig= plt.figure(i)
    ax=fig.add_subplot(111)
    ax.plot(ihist,OFhist,'r-',label="fmax")
    ax.set_xlabel(X)
    ax.set_ylabel(Y)

    pass

def PlotAll( xhist,shist,ihist,OFhist):
    """
    docstring
    """
    ComplementaryCondition=xhist*shist
    ComplementaryConditionX1=ComplementaryCondition[:,0,:]
    ComplementaryConditionX2=ComplementaryCondition[:,1,:]

    Plot(ihist,OFhist,'Iterations','Objective Function',1)
    Plot(ComplementaryConditionX1,ComplementaryConditionX2,'X1S1','X2S2',2)
    Plot(xhist[:,0,:],xhist[:,1,:],'X1','X2',3)
    pass