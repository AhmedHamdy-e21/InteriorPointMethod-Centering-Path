import InteriorPointMethodCentralPath as IP
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
Xmat=IP.Matricize(x)
Smat=IP.Matricize(s)
Sigma=0.5
alpha=0.7

Zero1,Identity,Zero2,Zero3,Zero4=IP.InitializeZerosAndIdentities(A,s,x)
i=0
StoppingCriteria=x.T@s
Tolerance=0.01
while StoppingCriteria[0]>Tolerance:
    i=i+1
    print(i)
    ## Affine Scalling with smaller step size without centering parameter
    A,b,c,s,x,y,Xmat,Smat,mu,StoppingCriteria=IP.IterateAffine( A,b,c,s,x,y,Zero1,Identity, Zero2,Zero3,Smat, Zero4,Xmat,Sigma,0.5)
    
    print('Vector x is \n',x,'\n Vector s is \n',s,'\n mu is ',mu)

