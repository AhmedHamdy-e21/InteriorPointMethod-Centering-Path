from Mehratora import *



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


# OFhist=ObjectiveFunction2(xhist[:,0,:],xhist[:,1,:]) ## Case Two

ComplementaryCondition=xhist*shist
ComplementaryConditionX1=ComplementaryCondition[:,0,:]
ComplementaryConditionX2=ComplementaryCondition[:,1,:]

Plot(ihist,OFhist,'Iterations','Objective Function',1)
Plot(ComplementaryConditionX1,ComplementaryConditionX2,'X1S1','X2S2',2)
Plot(xhist[:,0,:],xhist[:,1,:],'X1','X2',3)
print(ComplementaryConditionX1)
plt.show()
