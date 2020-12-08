from Mehratora import *
##############################################################################################
### First Case
##############################################################################################
A=IP.np.array([[1,1,1]])
b=IP.np.array([6])
x=IP.np.array([[5],[6],[1]]) 
s=IP.np.array([[1],[1],[1]]) 
y=IP.np.ones((1,1))
c=IP.np.array([[-1.1],[1],[0]])
##############################################################################################
Xmat=IP.Matricize(x)
Smat=IP.Matricize(s)
Sigma=0.5
alpha=alphaPrimal=alphaDual=0.5
StoppingCriteria=x.T@s
Tolerance=0.1
Zero1,Identity,Zero2,Zero3,Zero4=IP.InitializeZerosAndIdentities(A,s,x)
i=0
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
PlotAll(xhist,shist,ihist,OFhist)
plt.show()
