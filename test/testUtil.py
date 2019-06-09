import numpy as np, lie
from npp import SED
from scipy.stats import invwishart as iw, multivariate_normal as mvn

def GenerateRandomDataset(T, K, N, grp):
  # Generate random dataset
  m = getattr(lie, grp)
  o = SED.opts(lie=grp)
  zeroAlgebra = np.zeros(o.dxA)
  zeroObs = np.zeros(o.dy)

  if grp == 'se2':
    c = 1000
    E_scatter = np.diag([1.0, 5.0])
    S_scatter = np.diag([0.2, 0.1, 0.01])
    Q_scatter = 3*S_scatter

    # E = np.stack([iw.rvs(c, c*E_scatter) for k in range(K)])
    E = np.stack([np.diag(np.diag(iw.rvs(c, c*E_scatter))) for k in range(K)])
    S = np.stack([iw.rvs(c, c*S_scatter) for k in range(K)])
    Q = iw.rvs(c, c*Q_scatter)
  elif grp == 'se3':
    c = 1000
    E_scatter = np.diag([1.0, 5.0, 0.1])
    S_scatter = np.diag([.2, .1, .3, .1, .01, .001])
    Q_scatter = 3*S_scatter
    # E = np.stack([iw.rvs(c, c*E_scatter) for k in range(K)])
    E = np.stack([np.diag(np.diag(iw.rvs(c, c*E_scatter))) for k in range(K)])
    S = np.stack([iw.rvs(c, c*S_scatter) for k in range(K)])
    Q = iw.rvs(c, c*Q_scatter)
  else: assert False, 'Only support se(2) and se(3)'

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
