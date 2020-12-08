import InteriorPointMethodCentralPath as IP
import matplotlib.pyplot as plt
import numpy as np
##############################################################################################
### First Case
##############################################################################################
def ObjectiveFunction1(x1,x2):
    """
    docstring
    """
    return -1.1*x1-x2
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

A=IP.np.array([[1,1,1]])
b=IP.np.array([6])
x=IP.np.array([[5],[6],[1]]) 
s=IP.np.array([[1],[1],[1]]) 
y=IP.np.ones((1,1))
Xmat=IP.Matricize(x)
Smat=IP.Matricize(s)
c=IP.np.array([[-1.1],[1],[0]])
Xmat=IP.Matricize(x)
Smat=IP.Matricize(s)
Sigma=0.5
alpha=0.5

Zero1,Identity,Zero2,Zero3,Zero4=IP.InitializeZerosAndIdentities(A,s,x)
i=0
StoppingCriteria=x.T@s
Tolerance=0.01
xhist=[]
shist=[]
ihist=[]
while min(StoppingCriteria)>Tolerance:
    i=i+1
    print(i)
    ihist.append(i)
    ## Affine Scalling with smaller step size without centering parameter
    A,b,c,s,x,y,Xmat,Smat,mu,StoppingCriteria=IP.Iterate( A,b,c,s,x,y,Zero1,Identity, Zero2,Zero3,Smat, Zero4,Xmat,Sigma,0.5)
    xhist.append(x)
    shist.append(s)
    print('Vector x is \n',x,'\n Vector s is \n',s,'\n mu is ',mu)


xhist=np.array(xhist)
shist=np.array(shist)
ihist=np.array(ihist)
OFhist=ObjectiveFunction1(xhist[:,0,:],xhist[:,1,:])
ComplementaryCondition=xhist*shist
ComplementaryConditionX1=ComplementaryCondition[:,0,:]
ComplementaryConditionX2=ComplementaryCondition[:,1,:]
print(ihist)


# fig= plt.figure()
# ax=fig.add_subplot(111)
# # # Plot the Objective function
# ax.plot(ihist,OFhist,'r-',label="fmax")
# ax.set_xlabel('Iterations')
# ax.set_ylabel('Objective Function')


Plot(ihist,OFhist,'Iterations','Objective Function',1)
Plot(ComplementaryConditionX1,ComplementaryConditionX2,'X1S1','X2S2',2)
Plot(xhist[0,:],xhist[1,:],'X1','X2',3)

plt.show()
