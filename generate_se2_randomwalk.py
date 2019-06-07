import sys
sys.path.append('code')
import numpy as np, matplotlib.pyplot as plt
import du, du.stats, lie
import npp.SED as igp
import npp.drawSED as drawSED
# import igpSEN_relative as igp
from scipy.stats import multivariate_normal as mvn
from scipy.stats import invwishart as iw
import IPython as ip
import os


# path = 'data/synthetic/se2_randomwalk10'
path = 'data/synthetic/se2_randomwalk3'

# os.makedirs(path)
try: os.makedirs(path)
except: None

group = 'se2'
dy = 2

transX_parts_1sd = 0.5
transY_parts_1sd = 1.0
rotAngle_parts_1sd = 5.0 * np.pi/180.

transX_object_1sd = 1.5
transY_object_1sd = 3.0
rotAngle_object_1sd = 15.0 * np.pi/180.

T = 5
# K = 10
K = 3
Nt = [ 300 for t in range(T) ]
m = getattr(lie, group)
dxGf = m.n**2
dxGm = (m.n, m.n)
dxA = m.dof

df = 1000

# priors
H_Q = ('iw', df, df*np.diag((
  transX_object_1sd**2, transY_object_1sd**2, rotAngle_object_1sd**2
)))
H_x = ('mvnL', np.eye(dxA), .01*np.eye(dxA))
H_S = ('iw', df, df*np.diag((
  transX_parts_1sd**2, transY_parts_1sd**2, rotAngle_parts_1sd**2
)))

# be careful with initial part configuration prior
H_theta = ('mvnL', np.eye(dxA), np.diag((10**2, 10**2, 1**2)))
H_E = ('iw', df, df*np.diag((5, 1)))

# true parameters
zero = np.zeros(dxA)

Q = np.diag(np.diag(iw.rvs(*H_Q[1:])))
x = np.zeros((T,) + dxGm)
x[0] = m.expm(m.alg(mvn.rvs( m.algi(m.logm(H_x[1])), H_x[2] ) ))

for t in range(1,T):
  xVec = mvn.rvs(zero, Q)
  x[t] = x[t-1].dot(m.expm(m.alg(xVec)))

# part
E = np.stack([np.diag(np.diag(iw.rvs(*H_E[1:]))) for k in range(K)])
S = np.stack([iw.rvs(*H_S[1:]) for k in range(K)])
theta = np.zeros((T,K) + dxGm)

for k in range(K):
  theta[0,k] = m.expm(m.alg(mvn.rvs( m.algi(m.logm(H_theta[1])), H_theta[2] ) ))

for t in range(1, T):
  for k in range(K):
    s_tk = mvn.rvs(zero, S[k])
    theta[t,k] = theta[t-1,k].dot(m.expm(m.alg(s_tk)))

# obs
y = [ np.zeros((Nt[t], dy)) for t in range(T) ]
z = [ np.zeros(Nt[t], dtype=np.int) for t in range(T) ]
zeroY = np.zeros(dy)
for t in range(T):
  for k in range(K):
    idx = np.arange(k*Nt[t] // K, (k+1)*Nt[t] // K)
    ytk_part = mvn.rvs(zeroY, E[k], size=Nt[t] // K)
    T_world_part = x[t].dot(theta[t,k])
    y[t][idx] = igp.TransformPointsNonHomog(T_world_part, ytk_part)
    z[t][idx] = k

  # randomly permute y_t, z_t
  perm = np.random.permutation(range(Nt[t]))
  y[t] = y[t][perm]
  z[t] = z[t][perm]

# pi
pi = np.ones(K) / K

mins = np.array([ np.min(y[t], axis=0) for t in range(T) ])
maxs = np.array([ np.max(y[t], axis=0) for t in range(T) ])
xlim = np.array([ np.min(mins[:,0]), np.max( maxs[:,0] ) ])
ylim = np.array([ np.min(mins[:,1]), np.max( maxs[:,1] ) ])
cols = du.diffcolors(2, bgCols=[[1,1,1],[0,0,0]])

# # uncomment if we want to save new dataset
# du.save('%s/data' % path, {
#   'y': y, 'z': z,
#   'H_Q': H_Q, 'H_x': H_x,
#   'H_S': H_S, 'H_theta': H_theta,
#   'H_E': H_E,
#   'Q': Q, 'S': S, 'E': E,
#   'x': x, 'theta': theta, 'z': z,
#   'pi': pi
# })

oFake = igp.opts(lie='se2')
# filename = [ f'{path}/images/img-{t:05}.jpg' for t in range(T) ]
# title = [ f'Frame {t:05}' for t in range(T) ]
drawSED.draw(oFake, y=y, x=x, z=z, theta=theta, E=E)
plt.show()
