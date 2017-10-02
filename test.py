from cvxopt.solvers import qp
from cvxopt.base import matrix
from itertools import product
import numpy , pylab , random , math

def P_matrix(data, kernel,p):
    m = numpy.zeros(shape=(len(data),len(data)))
    for i,j in product(range(len(data)),repeat=2):
        m[i,j] = data[i][len(data[i])-1]*data[j][len(data[j])-1]*kernel(data[i][:len(data[i])-1],data[j][:len(data[j])-1],p)
    return m

def linear_kernel(x,y,p=1):
    return sum([x_i*y_i for x_i,y_i in zip(x,y)])+1

def polynomial_kernel(x,y,p=1):
    return math.pow(sum([x_i*y_i for x_i,y_i in zip(x,y)])+1,p)

def radial_kernel(x, y, sigma):

    gamma = (-1)/(2*(sigma**2))

    k = gamma*sum([numpy.linalg.norm(math.pow((x_i-y_i),2)) for x_i,y_i in zip(x,y)])

    return math.exp(k)

def q_vector(size):
    return numpy.array([[-1.0]*size]).T

def h_vector(size):
    return numpy.zeros(shape=(1,size)).T

def G_matrix(size):
    G = numpy.zeros(shape=(size,size))
    for i in range(size):
        G[i,i] = -1
    return G

def calc_alpha(data,kernel,threshold,p=1):
    size = len(data)
    P = matrix(P_matrix(data,kernel,p))
    q = matrix(q_vector(size))
    G = matrix(G_matrix(size))
    h = matrix(h_vector(size))
    return [[data[i],a] for i,a in enumerate(list(qp(P,q,G,h)['x'])) if a>threshold]

def indicator(new_data,kernel,alpha,p=1):
    s = sum([alpha[i][1]*alpha[i][0][len(alpha[i][0])-1]*kernel(new_data,alpha[i][0][:len(alpha[i][0])-1],p) for i in range(len(alpha))])
    return s
    #return numpy.sign(s)

# Uncomment t h e l i n e bel ow t o g e n e r a t e
# t h e same d a t a s e t ove r and ove r ag a in .
numpy.random.seed(100)
classA = [(random.normalvariate(-1.5 , 1),
random.normalvariate(0.5, 1) ,1.0)
for i in range (5)] + \
[ (random.normalvariate( 1.5, 1) ,random.normalvariate(0.5, 1) ,1.0)
for i in range (5)]
classB = [ (random.normalvariate(0.0, 0.5) ,random.normalvariate(-0.5, 0.5) ,-1.0)
for i in range (10) ]
data = classA + classB
random.shuffle(data)

pylab.clf()
#pylab.hold(True)
pylab.plot([p[0] for p in classA] ,
[p[1] for p in classA] ,
'bo')
pylab.plot([p[0] for p in classB] ,
[p[1] for p in classB] ,
'ro')

alpha = calc_alpha(data,radial_kernel,10**-5,1)
xrange=numpy.arange(-4, 4, 0.05)
yrange=numpy.arange(-4, 4, 0.05)
grid=matrix([[indicator((x, y),radial_kernel,alpha,1) for y in yrange] for x in xrange])
pylab.contour(xrange, yrange, grid,
(-1.0 ,0.0 , 1.0) ,
colors=('red','black','blue') ,
linewidths=(1,3,1))
pylab.show()                