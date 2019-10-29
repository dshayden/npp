from . import SED, drawSED
from .SED_tf import np2tf, ex, alg, generators_tf, obs_t_tf, Ei_tf, Si_tf
import du, du.stats
import lie
from scipy.special import logsumexp
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import chi2
import scipy.optimize
import matplotlib.pyplot as plt
import functools, numdifftools
import IPython as ip, sys
import tensorflow as tf

def optimize_local(o, y_t, x_t, thetaPrev, E, S, **kwargs):
  m = getattr(lie, o.lie)
  K = thetaPrev.shape[0]

  yPart = [ obs_t_tf(np2tf(
    SED.TransformPointsNonHomog(m.inv(x_t @ thetaPrev[k]), y_t) ))
    for k in range(K)
  ]

  Si = Si_tf(o, S)
  Ei = Ei_tf(o, E)

  G = generators_tf(o)
  s_t = tf.Variable(np2tf( kwargs.get('s_t', np.zeros(o.dxA*K)) ))

  def objective(s_t): 
    s_t_sep = [ s_t[k*o.dxA : (k+1)*o.dxA] for k in range(K) ]

    # observation cost
    S_t_inv = [ ex(-alg(G, s_t_sep[k])) for k in range(K) ]

    # v is Nx3                          3x3, Nx3
    v = [ tf.transpose(tf.linalg.matmul(S_t_inv[k], yPart[k], transpose_b=True))
      for k in range(K) ]

    # mahalanobis distance of <v_n, 0>
    vE = [ tf.linalg.matmul(v[k], Ei[k]) for k in range(K) ]
    negDists = tf.stack([
      -tf.reduce_sum(tf.multiply(vE[k], v[k]), axis=1) for k in range(K)]
    )
    smooth_mins = -tf.math.reduce_logsumexp(negDists, axis=0)
    cost = tf.reduce_sum(smooth_mins)

    # dynamics cost   
    sS = [ tf.linalg.matvec(Si[k], s_t_sep[k]) for k in range(K) ]
    sSs = [ tf.tensordot(sS[k], s_t_sep[k], axes=1) for k in range(K) ]
    costDyn = tf.reduce_sum(sSs)

    return cost + costDyn

  def grad(s_t):
    cost = tf.Variable(0.0)
    with tf.GradientTape() as tape: cost = objective(s_t)
    return cost, tape.gradient(cost, s_t)

  steps = kwargs.get('opt_steps', 10000)
  opt = tf.train.AdamOptimizer(learning_rate=0.01)
  print('Running Local Optimization:')
  prevCost = 1e6
  for s in range(steps):
    cost, grads = grad(s_t)
    opt.apply_gradients([(grads, s_t)])
    print(f'{s:05}, cost: {cost.numpy():.2f}, s: {s_t.numpy()}')
    if np.abs(cost.numpy() - prevCost) < 1e-6: break
    else: prevCost = cost.numpy()

  S_t = [ ex(alg(G, s_t[k*o.dxA : (k+1)*o.dxA])).numpy() for k in range(K) ]
  return np.stack(S_t)

def optimize_global(o, y_t, xPrev, theta_t, E, **kwargs):
  m = getattr(lie, o.lie)
  K = theta_t.shape[0]

  yObj = obs_t_tf(np2tf( SED.TransformPointsNonHomog(m.inv(xPrev), y_t) ))

  # theta = np2tf(theta_t)
  theta_inv = [ np2tf(m.inv(theta_t[k])) for k in range(K) ]

  Ei = Ei_tf(o, E)
  G = generators_tf(o)
  q_t = tf.Variable(np2tf( kwargs.get('q_t', np.zeros(o.dxA)) ))

  def objective(q_t):
    Q_t_inv = ex(-alg(G,q_t))

    thetaQ = [ tf.linalg.matmul(theta_inv[k], Q_t_inv) for k in range(K) ]

    # have 3 x 3 and N x 3
    v = [ tf.transpose(tf.linalg.matmul(thetaQ[k], yObj, transpose_b=True))
      for k in range(K) ]

    # get mahalanobis distance of v, 0 parameterized by E_k
    vE = [ tf.matmul(v[k], Ei[k]) for k in range(K) ]

    negDists = tf.stack([
      -tf.reduce_sum(tf.multiply(vE[k], v[k]), axis=1) for k in range(K)]
    )

    smooth_mins = -tf.math.reduce_logsumexp(negDists, axis=0)
    cost = tf.reduce_sum(smooth_mins)
    return cost

  def grad(q_t):
    cost = tf.Variable(0.0)
    with tf.GradientTape() as tape: cost = objective(q_t)
    return cost, tape.gradient(cost, q_t)

  steps = kwargs.get('opt_steps', 500)
  opt = tf.train.AdamOptimizer(learning_rate=0.1)
  print('Running Global Optimization:')
  prevCost = 1e6
  for s in range(steps):
    cost, grads = grad(q_t)
    opt.apply_gradients([(grads, q_t)])
    # print(f'{s:05}, cost: {cost.numpy():.2f}, q: {q_t.numpy()}')
    if np.abs(cost.numpy() - prevCost) < 1e-4: break
    else: prevCost = cost.numpy()

  return ex(alg(G,q_t)).numpy()


def register(o, y_t1, y_t, x_t1, theta_t1, E, **kwargs):
  # y_t1, x_t1, theta_t1 are previous time

  K = E.shape[0]
  m = getattr(lie, o.lie)

  # Initialize q_t as relative mean
  muDiff = np.mean(y_t, axis=0) - np.mean(y_t1, axis=0)

  Q_t = SED.MakeRd(np.eye(o.dy), muDiff)
  q_t = m.algi(m.logm(Q_t))

  # get observations y_t in coordinates of x_{t-1}
  y_tObj = SED.TransformPointsNonHomog(m.inv(x_t1), y_t)

  QTx = np.stack([ Q_t @ theta_t1[k] for k in range(K) ])
  z_t0 = _match(o, y_tObj, QTx, E, max=True)
  cost0 = _objective(o, y_tObj, theta_t1, E, q_t)

  plot = kwargs.get('plot', False)
  if plot:
    # draw prev time with previous parts, ensure we start with reasonable fit
    QTx1 = np.stack([ x_t1 @ theta_t1[k] for k in range(K) ])
    z_t1 = _match(o, y_t1, QTx1, E, max=True)
    drawSED.draw_t(o, x=x_t1, theta=theta_t1, E=E, y=y_t1, z=z_t1, reverseY=True)
    plt.title('Previous time')

    # draw this time with initial q_t estimate and consequent labels
    plt.figure()
    drawSED.draw_t(o, x=Q_t, theta=theta_t1, E=E, y=y_tObj, z=z_t0, reverseY=True)
    plt.title(f'Current time, initial estimate, cost0 {cost0:.2f}')

  maxIter = kwargs.get('maxIter', 1)
  for n in range(maxIter):
    f = functools.partial(_objective, o, y_tObj, theta_t1, E)
    gradf = numdifftools.Gradient(f)
    res = scipy.optimize.minimize(f, q_t, method='BFGS', jac=gradf)
    q_t = res.x
    cost = f(q_t)

    Q_t = m.expm(m.alg(q_t))
    QTx = np.stack([ Q_t @ theta_t1[k] for k in range(K) ])
    z_t = _match(o, y_tObj, QTx, E, max=True)

  # draw this time with updated q_t estimate and labels
  if plot:
    plt.figure()
    drawSED.draw_t(o, x=Q_t, theta=theta_t1, E=E, y=y_tObj, z=z_t, reverseY=True)
    plt.title(f'Current time, final estimate, cost {cost:.2f}')
    plt.show()

  return Q_t


def _mahalCentered(y, VI):
  dists = cdist(y, np.zeros((1, y.shape[1])), metric='mahalanobis', VI=VI)
  return dists.squeeze()

def _objective(o, y_t, Tx, E, q_t):
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
  # ctrCost = np.linalg.norm(dq - yMu)
  ctrCost = 0.0

  return np.sum(smoothMinDists) + ctrCost

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
