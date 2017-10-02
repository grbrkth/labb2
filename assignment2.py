from cvxopt.solvers import qp
from cvxopt.base import matrix
from itertools import product
import numpy , pylab , random , math

def linear_kernel(x,y):
    return sum([x_i*y_i for x_i,y_i in zip(x,y)])+1

def radial_kernel(x, y, sigma):

    gamma = (-1)/(2*(sigma**2))

    #k = gamma*sum([numpy.linalg.norm(math.pow((x_i-y_i),2)) for x_i,y_i in zip(x,y)])
    return math.exp(gamma*(numpy.linalg.norm(x-y)**2))
    #return math.exp(k)

def poly_kernel(x,y,p):
    return linear_kernel(x,y)**p

def sigmoid_kernel(x,y,k,delta):
    return numpy.tanh(k*(linear_kernel(x,y)-1)-delta)

def P_matrix(data, kernel,p=1):
    m = numpy.zeros(shape=(len(data),len(data)))
    for i,j in product(range(len(data)),repeat=2):
        m[i,j] = data[i][len(data[i])-1]*data[j][len(data[j])-1]*kernel(data[i][:len(data[i])-1],data[j][:len(data[j])-1],p)
    return m

# NO SLACK VARIABLES
def q_vector(size):
    return numpy.array([[-1.0]*size]).T

def h_vector(size):
    return numpy.zeros(shape=(1,size)).T

def G_matrix(size):
    G = numpy.zeros(shape=(size,size))
    for i in range(size):
        G[i,i] = -1
    return G

# WITH SLACK VARIABLES
def q_vector_slack(size):
    return numpy.array([[-1.0]*size]).T

def h_vector_slack(size,C):
    h = numpy.zeros(shape=(1,size*2))
    for i in range(size,size*2):
        h[0][i] = C
    return h.T


def G_matrix_slack(size):
    G = numpy.zeros(shape=(size*2,size))
    for i in range(size):
        G[i,i] = -1
    for i in range(size,size*2):
        G[i,i-size] = 1
    return G

def indicator(new_data,kernel,alpha,p=1):
    s = sum([alpha[i][1]*alpha[i][0][len(alpha[i][0])-1]*kernel(new_data,alpha[i][0][:len(alpha[i][0])-1],p=p) for i in range(len(alpha))])
    return s
    #return numpy.sign(s)

# Uncomment t h e l i n e bel ow t o g e n e r a t e
# t h e same d a t a s e t ove r and ove r ag a in .
numpy.random.seed(100)
classA = [(random.normalvariate(0.0 , 1),
random.normalvariate(0.0, 1) ,
1.0)
for i in range (5)] + \
[ (random.normalvariate( 0.0, 1) ,
random.normalvariate(0.0, 1) ,
1.0)
for i in range (5)]
classB = [ (random.normalvariate(0.0, 0.5) ,
random.normalvariate(0.0, 0.5) ,
-1.0)
for i in range (10) ]
data = classA + classB
random.shuffle(data)


def calc_alpha(data,kernel,threshold,p=1):
    size = len(data)
    P = matrix(P_matrix(data,kernel,p=p))
    q = matrix(q_vector(size))
    G = matrix(G_matrix(size))
    h = matrix(h_vector(size))
    return [[data[i],a] for i,a in enumerate(list(qp(P,q,G,h)['x'])) if a>threshold]

def calc_alpha_slack(data,kernel,threshold,p=1,C=0):
    size = len(data)
    P = matrix(P_matrix(data,kernel,p=p))
    q = matrix(q_vector_slack(size))
    G = matrix(G_matrix_slack(size))
    h = matrix(h_vector_slack(size,C))
    return [[data[i],a] for i,a in enumerate(list(qp(P,q,G,h)['x'])) if a>threshold]



alpha = calc_alpha_slack(data,poly_kernel,10**-5,4,C=0.1)
pylab.clf()
#pylab.hold(True)
pylab.plot([p[0] for p in classA] ,
[p[1] for p in classA] ,
'bo')
pylab.plot([p[0] for p in classB] ,
[p[1] for p in classB] ,
'ro')
#pylab.plot([a[0][0] for a in alpha], [a[0][1] for a in alpha], 'gH')
xrange=numpy.arange(-4, 4, 0.05)
yrange=numpy.arange(-4, 4, 0.05)
grid=matrix([[indicator((x, y),poly_kernel,alpha,4) for y in yrange] for x in xrange])
pylab.contour(xrange, yrange, grid,
(-1.0 ,0.0 , 1.0) ,
colors=('red','black','blue') ,
linewidths=(1,3,1))
pylab.show()
