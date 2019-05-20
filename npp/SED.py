from . import util

import numpy as np, argparse, lie, warnings
from scipy.linalg import block_diag
from sklearn.mixture import BayesianGaussianMixture as dpgmm 
from scipy.stats import multivariate_normal as mvn, invwishart as iw
from scipy.special import logsumexp
import du, du.stats # move useful functions from these to utils (catrnd)


import IPython as ip

def opts(**kwargs):
  """ Construct algorithm options

  KEYWORD INPUT
    H_* (tuple): Prior on * (* = Q, x, S, theta, E)
    lie (string): 'se2' or 'se3' (default: 'se2')
    alpha (float): Concentration parameter (default: 0.01)

  OUTPUT
    o (argparse.Namespace): Algorithm options
  """
  o = argparse.Namespace()

  # priors are parameter tuples, first element is string
  o.H_Q = kwargs.get('H_Q', None)
  o.H_x = kwargs.get('H_x', None)
  o.H_S = kwargs.get('H_S', None)
  o.H_theta = kwargs.get('H_theta', None)
  o.H_E = kwargs.get('H_E', None)

  o.lie = kwargs.get('lie', 'se2')
  if o.lie == 'se2':
    o.dxA = 3
    o.dy = 2
  elif o.lie == 'se3':
    o.dxA = 6
    o.dy = 3
  else:
    assert False, 'Only support se2 or se3'
  o.dxGm = (o.dy+1, o.dy+1)

  return o

def initPriorsDataDependent(o, y, **kwargs):
  """ Construct data-dependent priors from observations y. In-place modify o.

  Priors are constructed for
    H_* (tuple): (* = Q, x, S, theta, E)
  
  INPUT
    o (argparse.Namespace): Algorithm options
    y (list of ndarray, [ [N_1, dy], [N_2, dy], ..., [N_T, dy] ]): Observations

  KEYWORD INPUT
    dfQ (float): Degrees of freedom prior for Q
    rotQ (float): Expected body rotation magnitude (degrees).
    dfS (float): Degrees of freedom prior for S
    rotS (float): Expected part rotation magnitude (degrees).
    dfE (float): Degrees of freedom prior for E
    scaleE (float): Expected part rotation magnitude (degrees).
    rotX (float): Expected initial body rotation (degrees)
  """
  m = getattr(lie, o.lie)
  def d2r(d): return d * (np.pi / 180.)

  nRotAngles = 3 if o.lie == 'se3' else 1
  T = len(y)
  y_mu = np.stack( [ np.mean(y[t], axis=0) for t in range(T) ] )
  y_muAbsDiff = np.mean(np.abs(np.diff(y_mu, axis=0)), axis=0)
  y_meanVar = np.mean([np.var(y[t], axis=0) for t in range(T)], axis=0)
  
  # Q prior
  rotQ = kwargs.get('rotQ', 15.0)
  dfQ = kwargs.get('dfQ', 10.0)
  # expRotQ_radian = rotQ * (np.pi / 180.)
  expRotQ_radian = d2r(rotQ)
  o.H_Q = ('iw', dfQ, np.diag( np.concatenate((
    y_muAbsDiff * (dfQ - o.dxA - 1),
    expRotQ_radian*np.ones(nRotAngles) * (dfQ - o.dxA - 1)
  ))))

  # S prior
  rotS = kwargs.get('rotS', 1.5)
  dfS = kwargs.get('dfS', 10.0)
  # expRotS_radian = rotS * (np.pi / 180.)
  expRotS_radian = d2r(rotS)
  o.H_S = ('iw', dfS, np.diag( np.concatenate((
    y_muAbsDiff * (dfS - o.dxA - 1),
    expRotS_radian*np.ones(nRotAngles) * (dfS - o.dxA - 1)
  ))))

  # E prior
  dfE = kwargs.get('dfE', 10.0)
  scaleE = kwargs.get('scaleE', 0.1)
  o.H_E = ('iw', dfE, scaleE*np.diag(y_meanVar) * (dfE - o.dy - 1))

  # x prior
  muX = MakeRd(np.eye(o.dy), y_mu[0])
  rotX = kwargs.get('rotX', 180.0) # 1SD of x0 rotation
  xRotVar_radian = d2r(rotX)**2
  xTransVar = np.var(y[0], axis=0)
  SigmaX = np.diag( np.hstack([ xTransVar, xRotVar_radian]) )
  o.H_x = ('mvnL', muX, SigmaX)

  # theta prior
  muTheta = MakeRd(np.eye(o.dy), np.zeros(o.dy))
  rotTheta = kwargs.get('rotTheta', 180.0) # 1SD of theta0 rotation
  thetaRotVar_radian = d2r(rotTheta)**2
  thetaTransVar = np.var(y[0], axis=0)
  SigmaTheta = np.diag( np.hstack([ thetaTransVar, thetaRotVar_radian]) )
  o.H_theta = ('mvnL', muTheta, SigmaTheta)

def initXDataMeans(o, y):
  """ Initialize x with no rotation and translation as observed data means.

  INPUT
    o (argparse.Namespace): Algorithm options
    y (list of ndarray, [ [N_1, dy], [N_2, dy], ..., [N_T, dy] ]): Observations

  OUTPUT
    x (ndarray, [T,] + o.dxGm): Global latent dynamic.
  """
  T = len(y)
  y_mu = np.stack( [ np.mean(y[t], axis=0) for t in range(T) ] )
  x = np.tile( np.eye(o.dy+1), [T, 1, 1] )
  for t in range(T): x[t,:-1,-1] = y_mu[t]
  return x

def initPartsAssoc(o, y, x, alpha, **kwargs):
  """ Initialize parts (theta, E, S) and associations z using static DP.
  
  INPUT
    o (argparse.Namespace): Algorithm options
    y (list of ndarray, [ [N_1, dy], [N_2, dy], ..., [N_T, dy] ]): Observations
    x (ndarray, [T,] + o.dxGm): Global latent dynamic
    alpha (float): Concentration parameter

  KEYWORD INPUTS
    maxBreaks (int): Max parts under initialization (default: 20)
    nInit (int): Max seeds for initial clustering (default: 1)
    nIter (int): Max iterations per seed (default: 100)
    tInit (int or string): Time to initialize parts on, or 'random'

  OUTPUT
    theta (ndarray, [T, K, dt]): K-Part Local dynamic.
    E (ndarray, [K, dt, dt]): K-Part Local Extent
    S (ndarray, [K, dt, dt]): K-Part Local Dynamic
    z (list of ndarray, [ [N_1,], [N_2,], ..., [N_T,] ]): Associations
    pi (ndarray, [T, K+1]): stick weights, incl. unused portion
  """
  # Transform y to object frame of reference
  m = getattr(lie, o.lie)
  T = len(y)
  yObj = [ TransformPointsNonHomog(m.inv(x[t]), y[t]) for t in range(T) ]

  # Determine initial frame to initialize
  tInit = kwargs.get('tInit', 'random')
  if type(tInit) == str and tInit == 'random': tInit = np.random.randint(T)
  tInit = int(tInit)

  # fit at time tInit
  maxBreaks = kwargs.get('maxBreaks', 20)
  nInit = kwargs.get('nInit', 1)
  nIter = kwargs.get('nIter', 100)
  bgmm = dpgmm(maxBreaks, n_init=nInit, max_iter=nIter,
    weight_concentration_prior=alpha)
  with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    bgmm.fit(yObj[tInit])
  
  # get labels, means, covariances, and stick weights for tInit
  z0 = bgmm.predict(yObj[tInit])
  used = np.unique(z0)
  mu = bgmm.means_[used]
  Sigma = bgmm.covariances_[used]
  pi = bgmm.weights_[used]
  pi = np.concatenate((pi, [ 1 - np.sum(pi), ] ))
  if pi[-1] == 0:
    pi[-1] = alpha
    pi /= np.sum(pi)

  # relabel z0 from 0..len(used)-1
  zNew = -1*np.ones_like(z0)
  for cnt, u in enumerate(used): zNew[z0==u] = cnt
  nUsed = len(pi)-1
  z0 = zNew

  # get theta0, E0 from Sigma
  theta0 = np.zeros((nUsed, o.dy+1, o.dy+1))
  E0 = np.zeros((nUsed, o.dy, o.dy))
  for i in range(nUsed):
    _sigD, _R = util.eigh_proper_all(Sigma[i])
    theta0[i] = util.make_rigid_transform(_R[0], mu[i])
    E0[i] = np.diag(_sigD)

  # todo: build out theta trajectory from tInit
  #   for t in reversed(range(tInit)):
  #     initialize prediction
  #     associate
  #     backward filter
  #     associate
  #   for t in range(tInit, T):
  #     initialize prediction
  #     associate
  #     forward filter
  #     associate
  return z0, pi, theta0, E0

def inferZ(o, y_t, pi, theta_t, E, x_t, mL_t, **kwargs):
  """ Sample z_{tn} | y_{tn}, pi, theta_t, E

  INPUT
    o (argparse.Namespace): Algorithm options
    y_t (ndarray, [N_1, o.dy]): Observations at time t
    pi (ndarray, [K+1,]): Local association priors
    theta_t (ndarray, [K, [o.dxGm]]): K-Part local dynamics
    E (ndarray, [K, o.dy, o.dy]): K-Part observation covariance
    x_t (ndarray, o.dxGm): Global latent dynamic.
    mL_t (ndarray, [N_1,]): marginal LL

  KEYWORD INPUT
    max (bool): Max assignment rather than sampled assignment

  OUTPUT
    z (ndarray, [N_1,]): Associations
  """
  m = getattr(lie, o.lie)
  K = theta_t.shape[0]
  N = y_t.shape[0]
  assert K+1 == len(pi), 'pi must have K+1 components'
  assert np.all(pi > 0), 'pi must have all positive elements.'
  assert np.isclose(np.sum(pi), 1.0), 'pi must sum to 1.0'

  log_pz = np.zeros((N, K+1))
  zero = np.zeros(o.dy)
  logPi = np.log(pi)
  for k in range(K):
    yPart = TransformPointsNonHomog(m.inv(x_t.dot(theta_t[k])), y_t)
    log_pz[:,k] = logPi[k] + mvn.logpdf(yPart, zero, E[k], allow_singular=True)
  log_pz[:,-1] = logPi[-1] + mL_t

  # any association to K goes to base measure, set to -1
  pz = np.exp( log_pz - logsumexp(log_pz, axis=1, keepdims=True) )
  if kwargs.get('max', False): z = np.argmax(pz, axis=1)
  else: z = du.stats.catrnd(pz)
  z[z==K] = -1
  return z



# todo: move to utils
def MakeRd(R, d):
  """ Make SE(D) element from rotation matrix and translation vector.

  INPUT
    R (ndarray, [d, d]): Rotation matrix
    d (ndarray, [d,]): translation vector

  OUTPUT
    Rd (ndarray, [d+1, d+1]: Homogeneous Rotation + translation matrix
  """
  bottomRow = np.hstack( [np.zeros_like(d), np.array([1.])] )
  return np.block([ [R, d[:, np.newaxis]], [bottomRow] ])

# todo: move to utils
def TransformPointsNonHomog(T, y):
  """ Transform non-honogeneous points y with transformation T.

  INPUT
    T (ndarray, [dz, dz]): Homogeneous transformation matrix
    y (ndarray, [N, dz-1]): Non-homogeneous points

  OUTPUT
    yp (ndarray, [N, dz-1]): Transformed non-homogeneous points
  """
  R = T[:-1, :-1]
  d = T[:-1, -1][np.newaxis]
  if y.ndim == 1: y = y[np.newaxis]
  yp = np.squeeze(y.dot(R.T) + d)
  return yp
