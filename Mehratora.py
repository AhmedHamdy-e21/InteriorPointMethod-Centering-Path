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
    return -1.1*x1-x2

def ObjectiveFunction2(x1,x2):
    """
    docstring
    """
    return -30*x1-20*x2
def Plot(ihist,OFhist,X,Y,i):
    """
    docstring
    """
    fig= plt.figure(i)
    ax=fig.add_subplot(111)
    ax.plot(ihist,OFhist,'r-',label="fmax")
    ax.plot(ihist,OFhist,'go',label="fmax")
    
    ax.set_xlabel(X)
    ax.set_ylabel(Y)
    plt.savefig(str(i)+'Mehratora.png')
    plt.savefig(str(i)+'Mehratora.pdf')
    
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
