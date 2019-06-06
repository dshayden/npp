import numpy as np, lie
from npp import SED
from scipy.stats import invwishart as iw, multivariate_normal as mvn

def GenerateRandomDataset(T, K, N, grp):
  # Generate random dataset
  m = getattr(lie, grp)
  o = SED.opts(lie=grp)
  zeroAlgebra = np.zeros(o.dxA)
  zeroObs = np.zeros(o.dy)

  E = np.stack([iw.rvs(1000, 1000*np.diag([1.0, 5.0])) for k in range(K)])
  S = np.stack([iw.rvs(1000, 1000*np.diag([.2,.1,.01])) for k in range(K)])
  Q = iw.rvs(1000, 1000*np.diag([.5,.3,.01]))

  x = np.tile(np.eye(o.dy+1), [T,1,1])
  theta = np.tile(np.eye(o.dy+1), [T,K,1,1])
  y = [ [] for t in range(T) ]
  z = [ [] for t in range(T) ]
  for t in range(1,T):
    x[t] = x[t-1].dot(m.expm(m.alg(mvn.rvs(zeroAlgebra, Q))))
    for k in range(K):
      theta[t,k] = theta[t-1,k].dot(m.expm(m.alg(mvn.rvs(zeroAlgebra, Q))))

  for t in range(T):
    y[t] = np.vstack([SED.TransformPointsNonHomog(x[t].dot(theta[t,k]),
      mvn.rvs(zeroObs, E[k], size=N)) for k in range(K) ])
    z[t] = np.concatenate([k*np.ones(N, dtype=np.int) for k in range(K)])

  return o, x, theta, E, S, Q, y, z
