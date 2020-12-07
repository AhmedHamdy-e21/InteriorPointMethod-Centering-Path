import InteriorPointMethodCentralPath as IP


##############################################################################################
### Second Case
##############################################################################################
# Constraints paramters + slack variables
A=IP.np.array([[2,1,1,0],[1,3,0,1]])
print(A.shape[0])
b=IP.np.array([[8],[8]])
## Initialize values for initial point 
## I tried multiple arbitrary initial points and it's working awesome ^_^ 
x=IP.np.array([[3],[1],[0],[0]]) 
s=IP.np.array([[1],[1],[1],[1]])  ## This is initialized as identity
y=IP.np.ones((2,1))
c=IP.np.array([[-30],[-20],[0],[0]])
##############################################################################################
Xmat=IP.Matricize(x)
Smat=IP.Matricize(s)
Sigma=0.5
alpha=0.7

### I'll solve the augmented system first and in the next version I'll implement Cholesky Factorization
Zero1,Identity,Zero2,Zero3,Zero4=IP.InitializeZerosAndIdentities(A,s,x)
i=0
StoppingCriteria=x.T@s
Tolerance=0.01
while StoppingCriteria[0]>Tolerance:
    i=i+1
    print(i)
    A,b,c,s,x,y,Xmat,Smat,mu,StoppingCriteria=IP.Iterate( A,b,c,s,x,y,Zero1,Identity, Zero2,Zero3,Smat, Zero4,Xmat,Sigma,alpha)
    
    print('Vector x is \n',x,'\n Vector s is \n',s,'\n mu is ',mu)
 
