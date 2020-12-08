from InteriorPointMethodCentralPath import *
##############################################################################################
### First Case
##############################################################################################

A=np.array([[1,1,1]])
b=np.array([6])
x=np.array([[5],[6],[1]]) 
s=np.array([[1],[1],[1]]) 
y=np.ones((1,1))
Xmat=Matricize(x)
Smat=Matricize(s)
c=np.array([[-1.1],[1],[0]])
Xmat=Matricize(x)
Smat=Matricize(s)
Sigma=0.1
alpha=0.9

Zero1,Identity,Zero2,Zero3,Zero4=InitializeZerosAndIdentities(A,s,x)
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
    A,b,c,s,x,y,Xmat,Smat,mu,StoppingCriteria=Iterate( A,b,c,s,x,y,Zero1,Identity, Zero2,Zero3,Smat, Zero4,Xmat,Sigma,0.5)
    xhist.append(x)
    shist.append(s)
    print('Vector x is \n',x,'\n Vector s is \n',s,'\n mu is ',mu)


xhist=np.array(xhist)
shist=np.array(shist)
ihist=np.array(ihist)
OFhist=ObjectiveFunction1(xhist[:,0,:],xhist[:,1,:])
PlotAll(xhist,shist,ihist,-OFhist)
plt.show()

