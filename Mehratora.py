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
##############################################################################################
Xmat=IP.Matricize(x)
Smat=IP.Matricize(s)
Sigma=0.5
alpha=0.7
StoppingCriteria=x.T@s
Tolerance=0.01
### I'll solve the augmented system first and in the next version I'll implement Cholesky Factorization
Zero1,Identity,Zero2,Zero3,Zero4=IP.InitializeZerosAndIdentities(A,s,x)
i=0
