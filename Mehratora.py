import InteriorPointMethodCentralPath as IP
import numpy as np
def AdaptiveStepSize( x,s,y,AllDeltas):
    """
    docstring
    """
    alphaPrimal=min(1,-IP.np.min(x/AllDeltas[:x.shape[0]]))
    alphaDual=min(1,-IP.np.min(s/AllDeltas[x.shape[0]+y.shape[0]:x.shape[0]+s.shape[0]+y.shape[0]]))
    return alphaPrimal,alphaDual

def AdaptiveCenteringParameter( x,s,y,AllDeltas,mu):
    """
    docstring
    """
    alphaPrimal,alphaDual=AdaptiveStepSize( x,s,y,AllDeltas)
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
    mu=StoppingCriteria/x.shape[0]
    AugmentedSystem=IP.GenerateAugmentedSystem(Zero1, A,Identity, Zero2,Zero3,Smat, Zero4,Xmat)
    AugmentedB=GeneratAugmentedBCorrector(A,b,c,s,x,y, Xmat,Smat,mu,Sigma,AllDeltas)
    AllDeltas = np.linalg.solve(AugmentedSystem,AugmentedB)
    x,y,s,Xmat,Smat= CorrectorUpdate( x,y,s,AllDeltas,alphaPrimal,alphaDual)
    return b,c,s,x,y,Xmat,Smat,mu,StoppingCriteria,AllDeltas


  

##############################################################################################
### I think It's better first to choose fixed step size and check the predictor corrector algorithm. THEn edit the adaptive step size
##############################################################################################


##############################################################################################
### First Case
##############################################################################################
A=IP.np.array([[1,1,1]])
b=IP.np.array([6])
x=IP.np.array([[3],[3],[3]]) 
s=IP.np.array([[1],[1],[1]]) 
y=IP.np.ones((1,1))
Xmat=IP.Matricize(x)
Smat=IP.Matricize(s)
c=IP.np.array([[-1.1],[1],[0]])
##############################################################################################
Xmat=IP.Matricize(x)
Smat=IP.Matricize(s)
Sigma=0.0001
alpha=alphaPrimal=alphaDual=0.6
StoppingCriteria=x.T@s
Tolerance=0.01
### I'll solve the augmented system first and in the next version I'll implement Cholesky Factorization
Zero1,Identity,Zero2,Zero3,Zero4=IP.InitializeZerosAndIdentities(A,s,x)
i=0
b,c,s,x,y,Xmat,Smat,mu,StoppingCriteria,AllDeltas=IP.IterateAffine( A,b,c,s,x,y,Zero1,Identity, Zero2,Zero3,Smat, Zero4,Xmat,Sigma,alpha)
while StoppingCriteria>0.01 :
    print(x)
    i=i+1
    # Sigma,alphaPrimal,alphaDual=AdaptiveCenteringParameter( x,s,y,AllDeltas,mu)
    b,c,s,x,y,Xmat,Smat,mu,StoppingCriteria,AllDeltas=IteratePredictorCorrector( A,b,c,s,x,y,Zero1,Identity, Zero2,Zero3,Smat, Zero4,Xmat,Sigma,alphaPrimal,alphaDual,AllDeltas)
    print(x)
    print('\n',i,'\n',mu)
