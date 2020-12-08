import InteriorPointMethodCentralPath as IP
import numpy as np
import matplotlib.pyplot as plt
def AdaptiveStepSize( x,s,y,AllDeltas):
    """
    docstring
    """
    ds=AllDeltas[x.shape[0]+y.shape[0]:x.shape[0]+s.shape[0]+y.shape[0]]
    dx=x/AllDeltas[:x.shape[0]]

    print(dx,ds)
    alphaPrimal=min(1,-IP.np.min(x/dx))
    alphaDual=min(1,-IP.np.min(s/ds))
    print(alphaDual,alphaPrimal)
    return alphaPrimal,alphaDual

def AdaptiveCenteringParameter( x,s,y,AllDeltas,mu):
    """
    docstring
    """
    alphaPrimal,alphaDual=0.3,0.3
    mux=x+alphaPrimal*AllDeltas[:x.shape[0]]
    mus=s+alphaDual*AllDeltas[x.shape[0]+y.shape[0]:x.shape[0]+s.shape[0]+y.shape[0]]
    muaff=mux.T@mus/x.shape[0]
    print(alphaPrimal,alphaDual)
    Sigma=pow((muaff/mu),3)
    return Sigma,alphaPrimal,alphaDual


def GeneratAugmentedBCorrector(A,b,c,s,x,y, Xmat,Smat,mu,Sigma,AllDeltas):
    """
    docstring
    """
    
    DeltaSX=AllDeltas[:x.shape[0]].T@AllDeltas[x.shape[0]+y.shape[0]:x.shape[0]+s.shape[0]+y.shape[0]]*np.ones((Xmat.shape[0],1))

    rXSe=Xmat@Smat@np.ones((Xmat.shape[0],1))
    rMu=mu*Sigma*np.ones((Xmat.shape[0],1))
    rc=A.T@y+s-c
    rb=A@x-b
    rLast=-rXSe-DeltaSX+rMu
    AugmentedB=np.concatenate((-rc,-rb,rLast),axis=0)
    return AugmentedB


def CorrectorUpdate( x,y,s,AllDeltas,alphaPrimal,alphaDual):
    """
    docstring
    """
    x=x+alphaPrimal*AllDeltas[:x.shape[0]]
    y=y+alphaDual*AllDeltas[x.shape[0]:x.shape[0]+y.shape[0]]
    s=s+alphaDual*AllDeltas[x.shape[0]+y.shape[0]:x.shape[0]+s.shape[0]+y.shape[0]]
    Xmat=IP.Matricize(x)
    Smat=IP.Matricize(s)
    return x,y,s,Xmat,Smat



def IteratePredictorCorrector( A,b,c,s,x,y,Zero1,Identity, Zero2,Zero3,Smat, Zero4,Xmat,Sigma,alphaPrimal,alphaDual,AllDeltas):
    """
    docstring
    """
    StoppingCriteria=x.T@s
    mu=x.T@s/x.shape[0]
    AugmentedSystem=IP.GenerateAugmentedSystem(Zero1, A,Identity, Zero2,Zero3,Smat, Zero4,Xmat)
    AugmentedB=GeneratAugmentedBCorrector(A,b,c,s,x,y, Xmat,Smat,mu,Sigma,AllDeltas)
    AllDeltas = np.linalg.solve(AugmentedSystem,AugmentedB)
    x,y,s,Xmat,Smat= CorrectorUpdate( x,y,s,AllDeltas,alphaPrimal,alphaDual)
    return b,c,s,x,y,Xmat,Smat,mu,StoppingCriteria,AllDeltas

def ObjectiveFunction1(x1,x2):
    """
    docstring
    """
    return -30*x1-20*x2

def Plot( parameter_list):
    """
    docstring
    """
    ## plot 
    fig= plt.figure()
    ax=fig.add_subplot(111)
    # Plot the function
    ax.plot(x,f(x),'g-',label="f(x)")
    # Plot the minima
    xmins=np.array([xmin_global,xmin_local])
    ax.plot(xmins,f(xmins),'go',label="Minima")
    # Plot the roots
    roots=np.array([root.x,root2.x])
    ax.plot(roots,f(roots),'kv',label="Roots")
    ##zoom in around roots and mimima
    # ax.margins(x=-.02,y=5) 
    # ax.legend(loc='best')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.axhline(0,color="red")
    plt.show()
    pass
##############################################################################################
### First Case
##############################################################################################
# A=IP.np.array([[1,1,1]])
# b=IP.np.array([6])
# x=IP.np.array([[5],[6],[1]]) 
# s=IP.np.array([[1],[1],[1]]) 
# y=IP.np.ones((1,1))
# Xmat=IP.Matricize(x)
# Smat=IP.Matricize(s)
# c=IP.np.array([[-1.1],[1],[0]])
##############################################################################################

##############################################################################################
### Second Case
##############################################################################################
# Constraints paramters + slack variables
A=IP.np.array([[2,1,1,0],[1,3,0,1]])
print(A.shape[0])
b=IP.np.array([[8],[8]])
## Initialize values for initial point 
## I tried multiple arbitrary initial points and it's working awesome ^_^ 
x=IP.np.array([[3],[3],[0],[0]]) 
s=IP.np.array([[1],[1],[1],[1]])  ## This is initialized as identity
y=IP.np.ones((2,1))
c=IP.np.array([[-30],[-20],[0],[0]])
##############################################################################################
Xmat=IP.Matricize(x)
Smat=IP.Matricize(s)
Sigma=0.5
alpha=alphaPrimal=alphaDual=0.5
StoppingCriteria=x.T@s
Tolerance=0.1
### I'll solve the augmented system first and in the next version I'll implement Cholesky Factorization
Zero1,Identity,Zero2,Zero3,Zero4=IP.InitializeZerosAndIdentities(A,s,x)
i=0
# xhist=np.matrix((3,20))
xhist=[]
shist=[]
ihist=[]

while StoppingCriteria>Tolerance:
    ihist.append(i)
    i=i+1
    
    b,c,s,x,y,Xmat,Smat,mu,StoppingCriteria,AllDeltas=IP.IterateAffine( A,b,c,s,x,y,Zero1,Identity, Zero2,Zero3,Smat, Zero4,Xmat,Sigma,alpha)
    Sigma,alphaPrimal,alphaDual=AdaptiveCenteringParameter( x,s,y,AllDeltas,mu)
    b,c,s,x,y,Xmat,Smat,mu,StoppingCriteria,AllDeltas=IteratePredictorCorrector( A,b,c,s,x,y,Zero1,Identity, Zero2,Zero3,Smat, Zero4,Xmat,Sigma,alphaPrimal,alphaDual,AllDeltas)
    xhist.append(x)
    shist.append(s)
    print(x)
    print('\n',i,'\n')

xhist=np.array(xhist)
shist=np.array(shist)
ihist=np.array(ihist)
OFhist=ObjectiveFunction1(xhist[:,0,:],xhist[:,1,:])
ComplementaryCondition=xhist*shist
ComplementaryConditionX1=ComplementaryCondition[:,0,:]
ComplementaryConditionX2=ComplementaryCondition[:,1,:]
print(ihist)


fig= plt.figure()
ax=fig.add_subplot(111)
# # Plot the function
ax.plot(ihist,OFhist,'r-',label="fmax")
ax.set_xlabel('Iterations')
ax.set_ylabel('Objective Function')



# # Plot the minima
# xmins=np.array([xmin_global,xmin_local])
# ax.plot(xmins,f(xmins),'go',label="Minima")
# # Plot the roots
# roots=np.array([root.x,root2.x])
# ax.plot(roots,f(roots),'kv',label="Roots")
# ##zoom in around roots and mimima
# # ax.margins(x=-.02,y=5) 
# # ax.legend(loc='best')

# ax.axhline(0,color="red")
plt.show()



