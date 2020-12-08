import sys
import os
sys.path.append(os.path.abspath(sys.path[0] + '/..'))
from InteriorPointMethodCentralPath import *

##############################################################################################
### Second Case
##############################################################################################

# Constraints paramters + slack variables
A=np.array([[2,1,1,0],[1,3,0,1]])
print(A.shape[0])
b=np.array([[8],[8]])
## Initialize values for initial point 
## I tried multiple arbitrary initial points and it's working awesome ^_^ 
x=np.array([[3],[1],[0],[0]]) 
s=np.array([[1],[1],[1],[1]])  ## This is initialized as identity
y=np.ones((2,1))
c=np.array([[-30],[-20],[0],[0]])
##############################################################################################
Xmat=Matricize(x)
Smat=Matricize(s)
Sigma=0.5
alpha=0.7
Zero1,Identity,Zero2,Zero3,Zero4=InitializeZerosAndIdentities(A,s,x)
i=0
StoppingCriteria=x.T@s
Tolerance=0.01
xhist=[]
shist=[]
ihist=[]

while StoppingCriteria[0]>Tolerance:
    i=i+1
    print(i)
    ihist.append(i)
    A,b,c,s,x,y,Xmat,Smat,mu,StoppingCriteria=Iterate( A,b,c,s,x,y,Zero1,Identity, Zero2,Zero3,Smat, Zero4,Xmat,Sigma,alpha)
    xhist.append(x)
    shist.append(s)
    
    print('Vector x is \n',x,'\n Vector s is \n',s,'\n mu is ',mu)
 
xhist=np.array(xhist)
shist=np.array(shist)
ihist=np.array(ihist)
OFhist=ObjectiveFunction2(xhist[:,0,:],xhist[:,1,:])
PlotAll(xhist,shist,ihist,OFhist)
plt.show()
