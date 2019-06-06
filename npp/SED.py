from . import util, sample

import numpy as np, argparse, lie, warnings
from scipy.linalg import block_diag
from sklearn.mixture import BayesianGaussianMixture as dpgmm 
from scipy.stats import multivariate_normal as mvn, invwishart as iw
from scipy.special import logsumexp
import du, du.stats # move useful functions from these to utils (catrnd)
import functools

import IPython as ip

def opts(**kwargs):
  """ Construct algorithm options

  KEYWORD INPUT
    H_* (tuple): Prior on * (* = Q, x, S, theta, E)
    lie (string): 'se2' or 'se3' (default: 'se2')
    alpha (float): Concentration parameter (default: 0.01)
    Q_Diag (bool): Enforce diagonal Global Dynamic Noise Cov
    S_Diag (bool): Enforce diagonal Local Dynamic Noise Cov

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

  o.Q_Diag = kwargs.get('Q_Diag', True)
  o.S_Diag = kwargs.get('S_Diag', True)

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

def initPartsAndAssoc(o, y, x, alpha, mL, **kwargs):
  """ Initialize parts (theta, E, S) and associations z using static DP.
  
  INPUT
    o (argparse.Namespace): Algorithm options
    y (list of ndarray, [ [N_1, dy], [N_2, dy], ..., [N_T, dy] ]): Observations
    x (ndarray, [T,] + o.dxGm): Global latent dynamic
    alpha (float): Concentration parameter
    mL (list of ndarray, [N_1, ..., N_T]): Marginal Likelihoods

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
  K = len(pi)-1
  z0 = zNew

  # get theta0, E0 from Sigma
  theta0 = np.zeros((K, o.dy+1, o.dy+1))
  E0 = np.zeros((K, o.dy, o.dy))
  for i in range(K):
    _sigD, _R = util.eigh_proper_all(Sigma[i])
    theta0[i] = util.make_rigid_transform(_R[0], mu[i])
    E0[i] = np.diag(_sigD)

  S0 = [ iw.rvs(*o.H_S[1:]) for k in range(K) ]

  theta = np.zeros((T, K) + o.dxGm)
  theta[tInit] = theta0

  z = [ [] for t in range(T) ]
  z[tInit] = z0
  
  for t in reversed(range(tInit)):
    # propagate theta_{t+1} -> theta_t
    thetaPredict_t = theta[t+1]

    # associate
    zPredict = inferZ(o, y[t], pi, thetaPredict_t, E0, x[t], mL[t])

    # sample part rotation, translation
    for k in range(K):
      y_tk = y[t][zPredict==k]
      thetaPredict_t[k,:-1,:-1] = sampleRotationTheta(o, y_tk,
        thetaPredict_t[k], x[t], S0[k], E0[k], theta[t+1,k])
      thetaPredict_t[k,:-1,-1] = sampleTranslationTheta(o, y_tk,
        thetaPredict_t[k], x[t], S0[k], E0[k], theta[t+1,k])
    theta[t] = thetaPredict_t

    # re-associate
    z[t] = inferZ(o, y[t], pi, theta[t], E0, x[t], mL[t])

  for t in range(tInit+1, T):
    # propagate theta_{t-1} -> theta_t
    thetaPredict_t = theta[t-1]
    
    # associate
    zPredict = inferZ(o, y[t], pi, thetaPredict_t, E0, x[t], mL[t])

    # sample part rotation, translation
    for k in range(K):
      y_tk = y[t][zPredict==k]
      thetaPredict_t[k,:-1,:-1] = sampleRotationTheta(o, y_tk,
        thetaPredict_t[k], x[t], S0[k], E0[k], theta[t-1,k])
      thetaPredict_t[k,:-1,-1] = sampleTranslationTheta(o, y_tk,
        thetaPredict_t[k], x[t], S0[k], E0[k], theta[t-1,k])
    theta[t] = thetaPredict_t

    # re-associate
    z[t] = inferZ(o, y[t], pi, theta[t], E0, x[t], mL[t])

  
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

  # re-estimate S, E

    # theta (ndarray, [T, K, dt]): K-Part Local dynamic.
    # E (ndarray, [K, dt, dt]): K-Part Local Extent
    # S (ndarray, [K, dt, dt]): K-Part Local Dynamic
    # z (list of ndarray, [ [N_1,], [N_2,], ..., [N_T,] ]): Associations
    # pi (ndarray, [T, K+1]): stick weights, incl. unused portion
    #
  return theta, E0, S0, z, pi

def logpdf_t(o, y_t, z_t, x_t, theta_t, E, mL_t=None):
  """ Return time-t data log-likelihood, y_t | z_t, x_t, theta_t, E

  INPUT
    o (argparse.Namespace): Algorithm options
    y_t (ndarray, [N_t, dy]): Observations at time t
    z_t (ndarray, [N_t,]): Associations at time t
    x_t (ndarray, dxGm): Current global latent dynamic, time t
    theta_t (ndarray, [K, dxGm]): Part Dynamics, time t
    E (ndarray, [K, dy, dy]): Part Local Observation Noise Covs
    mL_t (ndarray, [N_t,]): marginal LL

  OUTPUT
    ll (float): log-likelihood for time t
  """
  m = getattr(lie, o.lie)
  K = theta_t.shape[0]
  zeroObs = np.zeros(o.dy)

  # current y_t observations
  ll = 0.0
  for k in range(K):
    z_tk = z_t == k
    Nk = np.sum(z_tk)
    if Nk == 0: continue
    y_tk_world = y_t[z_tk]
    T_part_world = m.inv( x_t.dot(theta_t[k]) )
    y_tk_part = TransformPointsNonHomog(T_part_world, y_tk_world)
    ll += np.sum(mvn.logpdf(y_tk_part, zeroObs, E[k], allow_singular=True))

  if mL_t is not None:
    z_t0 = z_t == -1
    ll += np.sum(mL_t[z_t0])

  return ll

def sampleTranslationX(o, y_t, z_t, x_t, theta_t, E, Q_tminus1, x_tminus1, **kwargs):
  """ Sample translation component d_x_t of x_t, allowing for correlation.

  INPUT
    o (argparse.Namespace): Algorithm options
    y_t (ndarray, [N_t, dy]): Observations at time t
    z_t (ndarray, [N_t,]): Associations at time t
    x_t (ndarray, dxGm): Current global latent dynamic, time t
    theta_t (ndarray, [K, dxGm]): Part Dynamics, time t
    E (ndarray, [K, dy, dy]): Part Local Observation Noise Covs
    Q_tminus1 (ndarray, [dxA, dxA]): Global Dynamic Noise Cov, time t-1
    x_tminus1 (ndarray, dxGm): Previous global latent dynamic, time t

  KEYWORD INPUTS
    x_tplus1 (ndarray, dxGm): Global latent dynamic, time t+1
    Q_tplus1 (ndarray, [dxA, dxA]): Global Dynamic Noise Cov, time t+1
    returnMuSigma (bool): Return mu, Sigma rather than sample

  OUTPUT
    d_x_t (ndarray, [dy,]): Sampled global k translation dynamic, time t
  """
  # assert o.Q_Diag, 'Method is only valid for Q_Diag = True'
  m = getattr(lie, o.lie)
  K = theta_t.shape[0]

  # Are we sampling from forward filter (past only) or full conditional?
  x_tplus1 = kwargs.get('x_tplus1', None)
  if x_tplus1 is None: noFuture = True
  else: noFuture = False

  # Check if we have different Q_tminus1, Q_tplus1
  Q_tplus1 = kwargs.get('Q_tplus1', Q_tminus1)

  # Separate R, d; get H, x2, b
  R_x_t, _ = m.Rt(x_t)
  R_x_tminus1, d_x_tminus1 = m.Rt(x_tminus1)

  V_x_tminus1_x_t_inv, x2 = m.getVi(m.inv(x_tminus1).dot(x_t), return_phi=True)
  if o.lie == 'se2': x2 = np.array([x2])
  H = V_x_tminus1_x_t_inv.dot(R_x_tminus1.T)
  b = -H.dot(d_x_tminus1)
  u = np.zeros(o.dxA)

  if not noFuture:
    R_x_tplus1, d_x_tplus1 = m.Rt(x_tplus1)
    V_x_t_x_tplus1_inv, x2_ = m.getVi(m.inv(x_t).dot(x_tplus1), return_phi=True)
    if o.lie == 'se2': x2_ = np.array([x2_])

    H_ = -V_x_t_x_tplus1_inv.dot(R_x_t.T)
    b_ = -H_.dot(d_x_tplus1)
    u_ = np.zeros(o.dxA)

    mu, Sigma = inferNormalConditionalNormal(x2, H, b, u, Q_tminus1,
      x2_, H_, b_, u_, Q_tplus1)
  else:
    mu, Sigma = inferNormalConditional(x2, H, b, u, Q_tminus1)

  Sigma_inv = np.linalg.inv(Sigma)
  for k in range(K):
    # if not isObserved[k]: continue
    R_theta_tk, d_theta_tk = m.Rt(theta_t[k])
    R2 = R_x_t.dot(R_theta_tk)
    obsCov = R2.dot(E[k]).dot(R2.T)
    obsCov_inv = R2.dot(np.linalg.inv(E[k])).dot(R2.T)
    bias = R_x_t.dot(d_theta_tk)

    z_tk = z_t == k
    Nk = np.sum(z_tk)
    if Nk == 0: continue
    SigmaNew_inv = Sigma_inv + Nk*obsCov_inv
    SigmaNew = np.linalg.inv(SigmaNew_inv)
    muNew = SigmaNew.dot( Sigma_inv.dot(mu) + obsCov_inv.dot(
      np.sum(y_t[z_tk], axis=0) - Nk*bias)
    )
    mu, Sigma, Sigma_inv = (muNew, SigmaNew, SigmaNew_inv)

  # ## incorporate \prod_n N( y_{tn} | ... )
  # ### pre-compute observation bias and covariance for each of K parts
  # obsCov = np.zeros((K, o.dy, o.dy))
  # bias = np.zeros((K, o.dy))
  # for k in range(K):
  #   # if not isObserved[k]: continue
  #   R_theta_tk, d_theta_tk = m.Rt(theta_t[k])
  #   R2 = R_x_t.dot(R_theta_tk)
  #   obsCov[k] = R2.dot(E[k]).dot(R2.T)
  #   bias[k] = R_x_t.dot(d_theta_tk)
  #
  # ### update with each observation
  # for n in range(y_t.shape[0]):
  #   k = z_t[n]
  #   if k >= 0:
  #     mu, Sigma = inferNormalNormal(y_t[n], obsCov[k], mu, Sigma, b=bias[k])

  if kwargs.get('returnMuSigma', False): return mu, Sigma
  else: return mvn.rvs(mu, Sigma)

def sampleTranslationTheta(o, y_tk, theta_tk, x_t, S_tminus1_k, E_k, theta_tminus1_k, **kwargs):
  """ Sample translation component d_theta_tk of theta_tk

  INPUT
     (argparse.Namespace): Algorithm options
    y_tk (ndarray, [N_tk, dy]): Observations associated to part k at time t
    theta_tk (ndarray, dxGm): Part k Local Dynamic (current estimate)
    x_t (ndarray, dxGm): Current global latent dynamic, time t
    S_tminus1_k (ndarray, [dxA, dxA]): Part k Local Dynamic Noise Cov 
    E_k (ndarray, [dy, dy]): Part k Extent
    theta_tminus1_k (ndarray, dxGm): Global latent dynamic, time t-1

  KEYWORD INPUTS
    theta_tplus1_k (ndarray, dxGm): Global latent dynamic, time t+1
    S_tplus1_k (ndarray, [dxA, dxA]): Part k Local Dynamic Noise Cov, time t+1
    returnMuSigma (bool): Return mu, Sigma rather than sample

  OUTPUT
    d_theta_tk (ndarray, [dy,]): Sampled part k translation dynamic, time t
  """
  m = getattr(lie, o.lie)

  # Are we sampling from forward filter (past only) or full conditional?
  theta_tplus1_k = kwargs.get('theta_tplus1_k', None)
  if theta_tplus1_k is None: noFuture = True
  else: noFuture = False

  # Check if we have different S_tminus1_k, S_tplus1_k
  S_tplus1_k = kwargs.get('S_tplus1_k', S_tminus1_k)

  # Separate R, d
  R_x_t, d_x_t = m.Rt(x_t)
  R_theta_tk, _ = m.Rt(theta_tk)
  R_theta_tminus1_k, d_theta_tminus1_k = m.Rt(theta_tminus1_k)

  # Get substitutions for prior
  V_theta_tminus1_k_theta_tk_inv, x2 = m.getVi(
    m.inv(theta_tminus1_k).dot(theta_tk), return_phi=True)
  if o.lie == 'se2': x2 = np.array([x2])
  H = V_theta_tminus1_k_theta_tk_inv.dot(R_theta_tminus1_k.T)
  b = -H.dot(d_theta_tminus1_k)
  u = np.zeros(o.dxA)

  if not noFuture:
    # setup
    R_theta_tplus1_k, d_theta_tplus1_k = m.Rt(theta_tplus1_k)
    V_theta_tk_theta_tplus1_k_inv, x2_ = m.getVi(
      m.inv(theta_tk).dot(theta_tplus1_k), return_phi=True)
    if o.lie == 'se2': x2_ = np.array([x2_])
    H_ = -V_theta_tk_theta_tplus1_k_inv.dot(R_theta_tk.T)
    b_ = -H_.dot(d_theta_tplus1_k)
    u_ = np.zeros(o.dxA)
    mu, Sigma = inferNormalConditionalNormal(x2, H, b, u, S_tminus1_k,
      x2_, H_, b_, u_, S_tplus1_k)
  else:
    mu, Sigma = inferNormalConditional(x2, H, b, u, S_tminus1_k)

  Nk = y_tk.shape[0]
  if Nk > 0:
    R2 = R_x_t.dot(R_theta_tk)
    H = R_x_t
    bias = d_x_t
    obsCov = R2.dot(E_k).dot(R2.T)
    obsCov_inv = R2.dot(np.linalg.inv(E_k)).dot(R2.T)
    
    Sigma_inv = np.linalg.inv(Sigma)
    SigmaNew_inv = Sigma_inv + Nk*H.T.dot(obsCov_inv).dot(H)
    SigmaNew = np.linalg.inv(SigmaNew_inv)
    muNew = SigmaNew.dot( Sigma_inv.dot(mu) + H.T.dot(obsCov_inv).dot(
      np.sum(y_tk, axis=0) - Nk*bias)
    )
    mu, Sigma = (muNew, SigmaNew)

    # ## incorporate \prod_n N( y_{tn} | z_{tn}==k, ... )
    # H = R_x_t
    # bias = d_x_t
    # R2 = R_x_t.dot(R_theta_tk)
    # obsCov = R2.dot(E_k).dot(R2.T)
    #
    # ## update with each observation
    # # TODO: can we do this without a loop?
    # for n in range(y_tk.shape[0]):
    #   mu, Sigma = inferNormalNormal(y_tk[n], obsCov, mu, Sigma, b=bias, H=H)

  # sample d_theta_tk
  # return mu, Sigma
  # return mu
  # return mvn.rvs(mu, Sigma)

  if kwargs.get('returnMuSigma', False): return mu, Sigma
  else: return mvn.rvs(mu, Sigma)

def sampleRotationX(o, y_t, z_t, x_t, theta_t, E, Q_tminus1, x_tminus1, **kwargs):
  """ Sample translation component d_x_t of x_t

  INPUT
    o (argparse.Namespace): Algorithm options
    y_t (ndarray, [N_t, dy]): Observations at time t
    z_t (ndarray, [N_t,]): Associations at time t
    x_t (ndarray, dxGm): Current global latent dynamic, time t
    theta_t (ndarray, [K, dxGm]): Part Dynamics, time t
    E (ndarray, [K, dy, dy]): Part Local Observation Noise Covs
    Q_tminus1 (ndarray, [dxA, dxA]): Global Dynamic Noise Cov, time t-1
    x_tminus1 (ndarray, dxGm): Previous global latent dynamic, time t

  KEYWORD INPUTS
    x_tplus1 (ndarray, dxGm): Global latent dynamic, time t+1
    Q_tplus1 (ndarray, [dxA, dxA]): Global Dynamic Noise Cov, time t+1
    nRotationSamples (int): length of markov chain for rotation slice sampler
    w (float): characteristic width of slice sampler, def: np.pi / 100
    P (float): max doubling iterations of slice sampler def: 10

  OUTPUT
    R_x_t (ndarray, [dy,]): Sampled global k translation dynamic, time t
  """
  m = getattr(lie, o.lie)
  K = theta_t.shape[0]

  nRotationSamples = kwargs.get('nRotationSamples', 10)
  if nRotationSamples == 0:
    R_x_t, _ = m.Rt(x_t)
    return R_x_t 

  # Get future values, if needed.
  x_tplus1 = kwargs.get('x_tplus1', None)
  if x_tplus1 is None:
    noFuture = True
  else:
    # Check if we have different Q_tminus1, Q_tplus1
    Q_tplus1 = kwargs.get('Q_tplus1', Q_tminus1)
    noFuture = False


  # Sample R_x_t | MB(R_x_t) using slice sampler
  if o.lie == 'se2': rotIdx = np.arange(2,3)
  elif o.lie == 'se3': rotIdx = np.arange(3,6) 

  ## setup
  zeroAlgebra = np.zeros(o.dxA)
  zeroObs = np.zeros(o.dy)

  # slice sampler parameters
  w = kwargs.get('w', np.pi / 100) # characteristic width
  P = kwargs.get('P', 10) # max doubling iterations

  x_t = x_t.copy()
  for idx in rotIdx:
    p0 = m.algi(m.logm(m.inv(x_tminus1).dot(x_t)))

    # inlined functional, making use of lots of closures
    def slice_log_x_t_full_conditional(idx, p0, a):
      p = p0.copy()
      p[idx] = a

      # full x_t from proposed angle a for given idx
      x_t_proposal = x_tminus1.dot(m.expm(m.alg(p)))

      ll = 0

      # prior
      ll += mvn.logpdf(
        m.algi(m.logm(m.inv(x_tminus1).dot(x_t_proposal))),
        zeroAlgebra,
        Q_tminus1,
        allow_singular=True
      )

      # future observations, if available
      if not noFuture:
        ll += mvn.logpdf(
          m.algi(m.logm(m.inv(x_t_proposal).dot(x_tplus1))),
          zeroAlgebra,
          Q_tplus1,
          allow_singular=True
        )

      ll += logpdf_t(o, y_t, z_t, x_t_proposal, theta_t, E)
      return ll

    # end full conditional functional
    llFunc = functools.partial(slice_log_x_t_full_conditional, idx, p0)
    x_t_sample_angles, _ll = sample.slice_univariate(llFunc, p0[idx],
      w=w, P=P, nSamples=nRotationSamples, xmin=-np.pi, xmax=np.pi)

    # Note: x_t_sample_angles are just univariate floats so we need to plug
    #       new sampled angle into matrix repr
    p = p0.copy()
    p[idx] = x_t_sample_angles[-1]
    x_t = x_tminus1.dot(m.expm(m.alg(p)))
  # end loop over rotation coordinates

  R_x_t, _ = m.Rt(x_t)
  return R_x_t 

def sampleRotationTheta(o, y_tk, theta_tk, x_t, S_tminus1_k, E_k, theta_tminus1_k, **kwargs):
  """ sample R_theta_tk | MB(R_theta_tk) using slice sampler

  INPUT
    o (argparse.Namespace): Algorithm options
    y_tk (ndarray, [N_tk, dy]): Observations associated to part k at time t
    theta_tk (ndarray, dxGm): Part k Local Dynamic (current estimate)
    x_t (ndarray, dxGm): Current global latent dynamic, time t
    S_tminus1_k (ndarray, [dxA, dxA]): Part k Local Dynamic Noise Cov 
    E_k (ndarray, [dy, dy]): Part k Extent
    theta_tminus1_k (ndarray, dxGm): Global latent dynamic, time t-1

  KEYWORD INPUTS
    theta_tplus1_k (ndarray, dxGm): Global latent dynamic, time t+1
    S_tplus1_k (ndarray, [dxA, dxA]): Part k Local Dynamic Noise Cov, time t+1
    nRotationSamples (int): length of markov chain for rotation slice sampler
    w (float): characteristic width of slice sampler, def: np.pi / 100
    P (float): max doubling iterations of slice sampler def: 10

  OUTPUT
    R_theta_tk (ndarray, [dy, dy]): Sampled rotation estimate.
  """
  m = getattr(lie, o.lie)
  nRotationSamples = kwargs.get('nRotationSamples', 10)
  if nRotationSamples == 0:
    R_theta_tk, _ = m.Rt(theta_tk)
    return R_theta_tk

  # Get future values, if needed.
  theta_tplus1_k = kwargs.get('theta_tplus1_k', None)
  if theta_tplus1_k is None: noFuture = True
  else: noFuture = False
  S_tplus1_k = kwargs.get('S_tplus1_k', S_tminus1_k)

  if o.lie == 'se2': rotIdx = np.arange(2,3)
  elif o.lie == 'se3': rotIdx = np.arange(3,6) 

  ## setup
  zeroAlgebra = np.zeros(o.dxA)
  zeroObs = np.zeros(o.dy)

  # slice sampler parameters
  w = kwargs.get('w', np.pi / 100) # characteristic width
  P = kwargs.get('P', 10) # max doubling iterations

  theta_tk = theta_tk.copy() # don't modify passed parameter
  z_tkFake = np.zeros(y_tk.shape[0], dtype=np.int) # fake z_t for logpdf_t
  for idx in rotIdx:
    p0 = m.algi(m.logm(m.inv(theta_tminus1_k).dot(theta_tk)))

    # inlined functional, making use of lots of closures
    def slice_log_theta_tk_full_conditional(idx, p0, a):
      p = p0.copy()
      p[idx] = a

      # full theta_tk from proposed angle a for given idx
      theta_tk_proposal = theta_tminus1_k.dot(m.expm(m.alg(p)))

      ll = 0

      # prior
      ll += mvn.logpdf(
        m.algi(m.logm(m.inv(theta_tminus1_k).dot(theta_tk_proposal))),
        zeroAlgebra,
        S_tminus1_k,
        allow_singular=True
      )

      # future observations, if available
      if not noFuture:
        ll += mvn.logpdf(
          m.algi(m.logm(m.inv(theta_tk_proposal).dot(theta_tplus1_k))),
          zeroAlgebra,
          S_tplus1_k,
          allow_singular=True
        )
     
      ll += logpdf_t(o, y_tk, z_tkFake, x_t, theta_tk_proposal[np.newaxis],
        E_k[np.newaxis])
      # if y_tk.shape[0] > 0:
      #   y_tk_world = y_tk
      #   T_part_world = m.inv(x_t.dot(theta_tk_proposal))
      #   y_tk_part = TransformPointsNonHomog(T_part_world, y_tk_world)
      #   ll += np.sum(mvn.logpdf(y_tk_part, zeroObs, E_k, allow_singular=True))

      return ll
    # end full conditional functional

    llFunc = functools.partial(slice_log_theta_tk_full_conditional, idx, p0)
    theta_tk_sample_angles, _ll = sample.slice_univariate(llFunc, p0[idx],
      w=w, P=P, nSamples=nRotationSamples, xmin=-np.pi, xmax=np.pi)

    # Note: theta_tk_sample_angles are just univariate floats so we need to plug
    #       new sampled angle into matrix repr
    p = p0.copy()
    p[idx] = theta_tk_sample_angles[-1]
    theta_tk = theta_tminus1_k.dot(m.expm(m.alg(p)))
  # end loop over rotation coordinates

  R_theta_tk, _ = m.Rt(theta_tk)
  return R_theta_tk

def inferZ(o, y_t, pi, theta_t, E, x_t, mL_t, **kwargs):
  """ Sample z_{tn} | y_{tn}, pi, theta_t, E

  INPUT
    o (argparse.Namespace): Algorithm options
    y_t (ndarray, [N_1, o.dy]): Observations at time t
    pi (ndarray, [K+1,]): Local association priors
    theta_t (ndarray, [K, [o.dxGm]]): K-Part local dynamics
    E (ndarray, [K, o.dy, o.dy]): K-Part observation covariance
    x_t (ndarray, o.dxGm): Global latent dynamic.
    mL_t (ndarray, [N_t,]): marginal LL

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

def getComponentCounts(o, z, pi):
  """ Count number of observations associated to each component.

  INPUT
    o (argparse.Namespace): Algorithm options
    z (ndarray, [N_1,]): Associations
    pi (ndarray, [K+1,]): Local association priors for each time

  OUTPUT
    Nk (ndarray, [K+1,]): Observation counts
  """
  K1 = len(pi)
  Nk = np.zeros(K1, dtype=np.int)
  uniq, cnts = np.unique(z[t], return_counts=True)
  for u, c in zip(uniq, cnts):
    uni = u if u != -1 else -1
    Nk[uni] = c
  return Nk

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

def inferNormalNormal(y, SigmaY, muX, SigmaX, H=None, b=None):
  """ Conduct a conjugate Normal-Normal update on the below system:

  p(x | y) \propto p(x)               p(y | x)
                 = N(x | muX, SigmaX) N(y | Hx + b, SigmaY)

  INPUT
    y (ndarray, [dy,]): Observation
    SigmaY (ndarray, [dy, dy]): Observation covariance
    muX (ndarray, [dx,]): Prior mean
    SigmaX (ndarray, [dx,dx]): Prior covariance
    H (ndarray, [dy,dx]): Multiplicative dynamics
    b (ndarray, [dy,]): Additive dynamics

  OUPUT
    mu (ndarray, [dx,]): Posterior mean
    Sigma (ndarray, [dx,dx]): Posterior covariance
  """
  (dy, dx) = (y.shape[0], muX.shape[0])
  if H is None: H = np.eye(dy,dx)
  if b is None: b = np.zeros(dy)

  SigmaXi = np.linalg.inv(SigmaX)
  SigmaYi = np.linalg.inv(SigmaY)

  SigmaI = SigmaXi + H.T.dot(SigmaYi).dot(H)
  Sigma = np.linalg.inv(SigmaI)
  mu = Sigma.dot(H.T.dot(SigmaYi).dot(y-b) + SigmaXi.dot(muX))

  return mu, Sigma

def inferNormalConditionalNormal(x2, H, b, u, S, x2_, H_, b_, u_, S_):
  """ Return mu, Sigma in x1 | x2, H, b, u, S, x2_, H_, b_, u_, S_, where
    
       x1 ~ N(x1 | mu, Sigma)
     propto N( [ H x1 + b; x2 ] | u, S ) * N( [ H_ x1 + b_; x2_ ] | u_, S_ )

  INPUT
    x2 (ndarray, [d2,])
    H (ndarray, [d1, d1])
    b (ndarray, [d1,])
    u (ndarray, [d1+d2,])
    S (ndarray, [d1+d2,d1+d2])
    x2_ (ndarray, [d2,])
    H_ (ndarray, [d1, d1])
    b_ (ndarray, [d1,])
    u_ (ndarray, [d1+d2,])
    S_ (ndarray, [d1+d2,d1+d2])

  OUTPUT
    mu (ndarray, [d1,])
    Sigma (ndarray, [d1,d1])
  """
  d1, d2 = (len(b), len(x2))

  # Get block matrices
  S11 = S[:d1,:d1]  # d1 x d1
  S12 = S[:d1,d1:]  # d1 x d2
  S21 = S12.T       # d2 x d1
  S22 = S[d1:,d1:]  # d2 x d2
  u1 = u[:d1]       # d1
  u2 = u[d1:]       # d2

  S11_ = S_[:d1,:d1]  # d1 x d1
  S12_ = S_[:d1,d1:]  # d1 x d2
  S21_ = S12_.T       # d2 x d1
  S22_ = S_[d1:,d1:]  # d2 x d2
  u1_ = u_[:d1]       # d1
  u2_ = u_[d1:]       # d2

  # Get necessary inverses
  S22inv = np.linalg.inv(S22)
  SoverS22inv = np.linalg.inv( S11 - S12.dot(S22inv).dot(S21) )

  S22_inv = np.linalg.inv(S22_)
  S_overS22_inv = np.linalg.inv( S11_ - S12_.dot(S22_inv).dot(S21_) )

  # Simplifying substitutions
  lam = u1 + S12.dot(S22inv).dot(x2 - u2) - b
  lam_ = u1_ + S12_.dot(S22_inv).dot(x2_ - u2_) - b_

  # Final calculations
  SigmaI = H.T.dot(SoverS22inv).dot(H) + \
           H_.T.dot(S_overS22_inv).dot(H_)
  Sigma = np.linalg.inv(SigmaI)
  mu = Sigma.dot( H.T.dot(SoverS22inv).dot(lam) +
                  H_.T.dot(S_overS22_inv).dot(lam_))
  return mu, Sigma

def inferNormalConditional(x2, H, b, u, S):
  """ Return mu, Sigma in x1 | x2, H, b, u, S, where
    
       x1 ~ N(x1 | mu, Sigma)
     propto N( [ H x1 + b; x2 ] | u, S )

  INPUT
    x2 (ndarray, [d2,])
    H (ndarray, [d1, d1])
    b (ndarray, [d1,])
    u (ndarray, [d1+d2,])
    S (ndarray, [d1+d2,d1+d2])

  OUTPUT
    mu (ndarray, [d1,])
    Sigma (ndarray, [d1,d1])
  """
  d1, d2 = (len(b), len(x2))

  # Get block matrices
  S11 = S[:d1,:d1]  # d1 x d1
  S12 = S[:d1,d1:]  # d1 x d2
  S21 = S12.T       # d2 x d1
  S22 = S[d1:,d1:]  # d2 x d2
  u1 = u[:d1]       # d1
  u2 = u[d1:]       # d2

  # Get necessary inverses
  S22inv = np.linalg.inv(S22)

  SoverS22 = S11 - S12.dot(S22inv).dot(S21)

  # Simplifying substitutions
  lam = u1 + S12.dot(S22inv).dot(x2 - u2) - b

  # Finally
  Sigma = SoverS22
  mu = lam
  
  H_i = np.linalg.inv(H)
  mu = H_i.dot(mu)
  Sigma = H_i.dot(Sigma).dot(H_i.T)

  return mu, Sigma
