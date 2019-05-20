import numpy as np
import npp.SED as igp
import IPython as ip
np.set_printoptions(suppress=True, precision=4)

o = igp.opts()
y = [ np.random.rand(100,2) for t in range(3) ]
igp.initPriorsDataDependent(o, y)
x = igp.initXDataMeans(o, y)
alpha = 0.01

t = 0
z0, pi, theta0, E0 = igp.initPartsAssoc(o, y, x, alpha, tInit=t)

mL_t = -14*np.ones(y[t].shape[0])
z0_prime = igp.inferZ(o, y[t], pi, theta0, E0, x[t], mL_t)

print( np.sum(z0 == z0_prime) )

print(z0)
print(z0_prime)

ip.embed()
