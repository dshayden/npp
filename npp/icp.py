from . import SED, drawSED
import du, du.stats
import lie
from scipy.special import logsumexp
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import chi2
import scipy.optimize
import matplotlib.pyplot as plt
import functools, numdifftools
import IPython as ip

def register(o, y_t1, y_t, x_t1, theta_t1, E, **kwargs):
  K = E.shape[0]
  m = getattr(lie, o.lie)

  # Initialize q_t
  muDiff = np.mean(y_t, axis=0) - np.mean(y_t1, axis=0)
  # muDiff = np.mean(y_t, axis=0)

  Q_t = SED.MakeRd(np.eye(o.dy), muDiff)
  # Q_t = SED.MakeRd(np.eye(o.dy), np.zeros(o.dy))
  q_t = m.algi(m.logm(Q_t))

  # QTx = np.stack([ Q_t @ theta_t1[k] for k in range(K) ])
  QTx = np.stack([ x_t1 @ Q_t @ theta_t1[k] for k in range(K) ])
  z_t = z_tPrev = _match(o, y_t, QTx, E)

  cost0 = _objective(o, y_t, theta_t1, E, z_tPrev, q_t)

  drawSED.draw_t(o, x=x_t1 @ Q_t, theta=theta_t1, E=E, y=y_t, z=z_t, reverseY=True)

  # drawSED.draw_t(o, x=Q_t, theta=theta_t1, E=E, y=y_t, z=z_t, reverseY=True)
  # drawSED.draw_t(o, x=np.eye(3), theta=QTx, E=E, y=y_t, z=z_t, reverseY=True)

  xlim, ylim = (kwargs.get('xlim', None), kwargs.get('ylim', None))
  if xlim is not None: plt.xlim(xlim)
  if ylim is not None: plt.ylim(ylim)

  # plt.xlim(-40, 40); plt.ylim(-40, 40)
  plt.title(f'cost0 {cost0:.2f}')

  maxIter = kwargs.get('maxIter', 1)
  for n in range(maxIter):
    f = functools.partial(_objective, o, y_t, theta_t1, E, z_tPrev)
    gradf = numdifftools.Gradient(f)
    res = scipy.optimize.minimize(f, q_t, method='BFGS', jac=gradf)
    q_t = res.x
    # print(res)
    cost = f(q_t)

    print(q_t, cost)

    Q_t = m.expm(m.alg(q_t))
    QTx = np.stack([ Q_t @ theta_t1[k] for k in range(K) ])

    # drawSED.draw_t(o, x=np.eye(3), theta=QTx, E=E, y=y_t, z=z_t)
    # plt.show()

    z_t = _match(o, y_t, QTx, E, max=True)
    
    z_tPrev = z_t

  plt.figure()
  
  drawSED.draw_t(o, x=x_t1 @ Q_t, theta=theta_t1, E=E, y=y_t, z=z_t, reverseY=True)
  # drawSED.draw_t(o, x=Q_t, theta=theta_t1, E=E, y=y_t, z=z_t, reverseY=True)
  # drawSED.draw_t(o, x=np.eye(3), theta=QTx, E=E, y=y_t, z=z_t, reverseY=True)
  if xlim is not None: plt.xlim(xlim)
  if ylim is not None: plt.ylim(ylim)
  # plt.xlim(-40, 40); plt.ylim(-40, 40)
  plt.title(f'cost {cost:.2f}')
  plt.show()

  print(q_t)

  # cost = _objective(o, y_t, theta_t1, E, z_t, q_t)
  # print(cost)

  # drawSED.draw_t(o, x=np.eye(3), theta=QTx, E=E, y=y_t, z=z_t)
  # plt.show()

def _mahalCentered(y, VI):
  dists = cdist(y, np.zeros((1, y.shape[1])), metric='mahalanobis', VI=VI)
  return dists.squeeze()

def _objective(o, y_t, Tx, E, z_t, q_t):
  m = getattr(lie, o.lie)

  # transform points to part coordinates
  N, K = ( y_t.shape[0], E.shape[0] )
  Q_t = m.expm(m.alg(q_t))
  QTx = [ Q_t @ Tx[k] for k in range(K) ]

  dists = np.zeros((N, K))
  for k in range(K): dists[:,k] = _dist_point2part(o, y_t, QTx[k], E[k])

  # part fitting cost
  smoothMinDists = -logsumexp(-dists, axis=1)
  
  # object center cost
  _, dq = m.Rt(Q_t)
  yMu = np.mean(y_t, axis=0)
  ctrCost = np.linalg.norm(dq - yMu)

  return np.sum(smoothMinDists) + ctrCost

  # dists = np.zeros(N)
  # for k in range(K):
  #   ztk = z_t == k
  #   dists[ztk] = _dist_point2part(o, y_t[ztk], QTx[k], E[k])
  # return np.sum(dists)

def _match(o, y_t, QTx, E, **kwargs):
  """ Sample assignments z_t to one of K parts (QTx[k], E[k]).

  INPUT
    o (Namespace): options
    y_t (ndarray, [N_t, dy]): Observations (in Reference coordinates) at time t
    QTx (ndarray, [(K,) + o.dxGm]): K Reference -> Part Transformations
    E (ndarray, [K, dy, dy]): K Part extents (all diagonal)
  """
  m = getattr(lie, o.lie)
  N, K = ( y_t.shape[0], E.shape[0] )
  dists = np.zeros((N, K))
  for k in range(K): dists[:,k] = _dist_point2part(o, y_t, QTx[k], E[k])
  log_pz = -0.5 * dists

  # sample 
  pz = np.exp( log_pz - logsumexp(log_pz, axis=1, keepdims=True) )
  if kwargs.get('max', False): z = np.argmax(pz, axis=1)
  else: z = du.stats.catrnd(pz)

  return z

def _dist_point2part(o, y_t, Tx, E_k):
  """ Determine Mahalanobis distance of points to y_t to part (T_x, E_k)

  INPUT
    o (Namespace): options
    y_t (N_t, dy): Observations (in world coordinates) at time t
    Tx (ndarray, o.dxGm): World -> Part Transformation
    E_k (ndarray, [dy, dy]): Part extent (diagonal)

  OUPUT
    dists (N_t,): Point distances in units of standard deviations
  """
  m = getattr(lie, o.lie)
  N = y_t.shape[0]

  # Now we have a centered, uncorrelated Gaussian
  yLocal = SED.TransformPointsNonHomog(m.inv(Tx), y_t)
  E_ki = np.diag(1/np.diag(E_k))
  dists = _mahalCentered(yLocal, E_ki)
  return dists

  # mahal distance and SD lineup
  # sd = 1.25
  # lhs = _mahalCentered(yLocal, E_ki)
  # baseColors = np.array([[1, 0, 0], [0, 1, 0]])
  # pointColors = baseColors[(lhs <= sd).astype(np.int)]
  # pts = du.stats.Gauss2DPoints(np.zeros(o.dy), E_k, deviations=sd)
  # plt.plot(*pts, color='k')
  # plt.scatter(yLocal[:,0], yLocal[:,1], s=2, color=pointColors)
  # plt.xlim(-50, 30)
  # plt.ylim(-50, 30)
  # plt.show()
  
  # chi2 test
  # lhs = _mahalCentered(yLocal, E_ki)
  # chi2_stat = chi2.ppf(1-0.01,df=2)
  # baseColors = np.array([[1, 0, 0], [0, 1, 0]])
  # pointColors = baseColors[(lhs < chi2_stat).astype(np.int)]
  # plt.scatter(y_t[:,0], y_t[:,1], s=5, color=pointColors)
  # plt.show()


  

  # define E as fcn of E_k and sd
  ## || Q^{1/2} x ||^2 <= b
