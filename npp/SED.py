from . import util, sample

import numpy as np, argparse, lie, warnings
from scipy.linalg import block_diag
from sklearn.mixture import BayesianGaussianMixture as dpgmm 
from sklearn.mixture import GaussianMixture as gmm
from scipy.stats import multivariate_normal as mvn, invwishart as iw
from scipy.stats import dirichlet, beta as Be
from scipy.special import logsumexp
import du, du.stats # move useful functions from these to utils (catrnd)
import functools

import IPython as ip

def opts(**kwargs):
  r''' Construct algorithm options

  KEYWORD INPUT
    H_* (tuple): Prior on * (* = Q, x, S, theta, E)
    lie (string): 'se2' or 'se3' (default: 'se2')
    alpha (float): Concentration parameter (default: 0.01)

  OUTPUT
    o (argparse.Namespace): Algorithm options
  ''' 
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
  r''' Construct data-dependent priors from observations y. In-place modify o.

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
  '''
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
  if o.lie == 'se2':
    SigmaX = np.diag( np.hstack([ xTransVar, xRotVar_radian]) )
  else:
    SigmaX = np.diag( np.hstack([ xTransVar, xRotVar_radian*np.ones(3)]) )
  # SigmaX = np.diag( np.hstack([ xTransVar, xRotVar_radian]) )
  o.H_x = ('mvnL', muX, SigmaX)

  # theta prior
  muTheta = MakeRd(np.eye(o.dy), np.zeros(o.dy))
  rotTheta = kwargs.get('rotTheta', 180.0) # 1SD of theta0 rotation
  thetaRotVar_radian = d2r(rotTheta)**2
  thetaTransVar = np.var(y[0], axis=0)

  if o.lie == 'se2':
    SigmaTheta = np.diag( np.hstack([ thetaTransVar, thetaRotVar_radian]) )
  else:
    SigmaTheta = np.diag( np.hstack([ thetaTransVar, thetaRotVar_radian*np.ones(3)]) )
  # SigmaTheta = np.diag( np.hstack([ thetaTransVar, thetaRotVar_radian]) )
  o.H_theta = ('mvnL', muTheta, SigmaTheta)

def initXDataMeans(o, y):
  r''' Initialize x with no rotation and translation as observed data means.

  INPUT
    o (argparse.Namespace): Algorithm options
    y (list of ndarray, [ [N_1, dy], [N_2, dy], ..., [N_T, dy] ]): Observations

  OUTPUT
    x (ndarray, [T,] + o.dxGm): Global latent dynamic.
  '''
  T = len(y)
  y_mu = np.stack( [ np.mean(y[t], axis=0) for t in range(T) ] )
  x = np.tile( np.eye(o.dy+1), [T, 1, 1] )
  for t in range(T): x[t,:-1,-1] = y_mu[t]
  return x

def initPartsAndAssoc(o, y, x, alpha, mL, **kwargs):
  r''' Initialize parts (theta, E, S) and associations z using static DP.
  
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
    fixedBreaks (bool): Force number of components to maxBreaks

  OUTPUT
    theta (ndarray, [T, K, dxA]): K-Part Local dynamic.
    E (ndarray, [K, dxA, dxA]): K-Part Local Extent
    S (ndarray, [K, dxA, dxA]): K-Part Local Dynamic
    z (list of ndarray, [ [N_1,], [N_2,], ..., [N_T,] ]): Associations
    pi (ndarray, [T, K+1]): stick weights, incl. unused portion
  '''
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

  if not kwargs.get('fixedBreaks', False):
    # nonparametric init
    bgmm = dpgmm(maxBreaks, n_init=nInit, max_iter=nIter,
      weight_concentration_prior=alpha)
  else:
    # parametric init
    bgmm = gmm(maxBreaks, n_init=nInit, max_iter=nIter)

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
  if pi[-1] <= 0: pi[-1] = alpha

  # renormalize pi
  logPi = np.log(pi)
  pi = np.exp(logPi - logsumexp(logPi))

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
  

  # sample S from prior
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

  # infer S, E
  S = np.array([ inferSk(o, theta[:,k]) for k in range(K) ])
  E = inferE(o, x, theta, y, z)

  # have to do this after E, S because they don't exist yet
  z, pi, theta, E, S = consolidatePartsAndResamplePi(o, z, pi, alpha, theta,
    E, S)
  K = len(pi) - 1


  # z, theta, E, S = consolidateExtantParts(o, z, pi, theta, E, S)
  # Nk = np.sum([getComponentCounts(o, z[t], pi) for t in range(T)], axis=0)
  # pi = inferPi(o, Nk, alpha)

  return theta, E, S, z, pi

def logpdf_data_t(o, y_t, z_t, x_t, theta_t, E, mL_t=None):
  r''' Return time-t data log-likelihood, y_t | z_t, x_t, theta_t, E

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
  '''
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

def logpdf_assoc_t(o, z_t, pi):
  r''' Return log-likelihood of z_t | pi.

  INPUT
    o (argparse.Namespace): Algorithm options
    z_t (ndarray, [N_t,]): Associations at time t
    pi (ndarray, [T, K+1]): stick weights, incl. unused portion

  OUTPUT
    ll (float): log-likelihood
  '''
  logPi = np.log(pi)
  ll = np.sum( logPi[z_t] )

  # ztNonNeg = z[t][z[t]>=0]

  # ll = 0.0

  # ll += np.sum( logPi[ztNonNeg] )
  # ll += mL * np.sum(z[t]==-1)

  return ll

  None

def logpdf_parameters(o, alpha, pi, E, S, Q):
  r''' Return log-likelihood, E, S, Q, pi | alpha, H_*

  INPUT
    o (argparse.Namespace): Algorithm options
    alpha (float): Concentration parameter (default: 0.01)
    pi (ndarray, [T, K+1]): stick weights, incl. unused portion
    E (ndarray, [K, dxA, dxA]): K-Part Local Extent
    S (ndarray, [K, dxA, dxA]): K-Part Local Dynamic
    Q (ndarray, [dxA, dxA]): Latent Dynamic

  OUTPUT
    ll (float): log-likelihood for parameters
  '''
  K = len(pi)-1
  ll = 0.0

  # pi
  betaPrime = np.zeros_like(pi)
  betaPrime[0] = np.minimum(1.0-1e-16, np.maximum(0.0, pi[0]))
  for k in range(1,len(pi)):
    betaPrime[k] = pi[k] / np.prod( 1 - betaPrime[:k-1] )
    betaPrime[k] = np.minimum(1.0-1e-16, np.maximum(0.0, betaPrime[k]))
  ll += np.sum(Be.logpdf(betaPrime, 1.0, alpha))

  # E, S, Q
  ll += np.sum( [iw.logpdf(E[k], *o.H_E[1:]) for k in range(K)] )
  ll += np.sum( [iw.logpdf(S[k], *o.H_S[1:]) for k in range(K)] )
  ll += iw.logpdf(Q, *o.H_Q[1:])
  return ll

def logpdf_x_t(o, x_t, x_tminus1, Q):
  r''' Calculate logpdf of x_t | x_{t-1}, Q.

  INPUT
    o (argparse.Namespace): Algorithm options
    x_t (ndarray, o.dxGm): Global latent dynamic, time t
    x_tminus1 (ndarray, o.dxGm): Global latent dynamic, time t-1
    Q (ndarray, [dxA, dxA]): Latent Dynamic
  
  OUTPUT
    ll (float): log-likelihood
  '''
  return mvnL_logpdf(o, x_t, x_tminus1, Q)

def logpdf_theta_tk(o, theta_tk, theta_tminus1_k, Sk):
  r''' Calculate logpdf of theta_tk | theta_{(t-1)k}, S_k

  INPUT
    o (argparse.Namespace): Algorithm options
    theta_tk (ndarray, o.dxGm): Part latent dynamic, time t
    theta_tminus1_k (ndarray, o.dxGm): Part latent dynamic, time t-1
    Sk (ndarray, [dxA, dxA]): Latent Part Dynamic Noise
  
  OUTPUT
    ll (float): log-likelihood
  '''
  return mvnL_logpdf(o, theta_tk, theta_tminus1_k, Sk)

def logJoint(o, y, z, x, theta, E, S, Q, alpha, pi, mL=None):
  r''' Calculate joint log-likelihood of below:

      p(y, z, x, theta, E, S, Q, pi | H_*, alpha)

  INPUT
    o (argparse.Namespace): Algorithm options
    y (list of ndarray, [ [N_1, dy], [N_2, dy], ..., [N_T, dy] ]): Observations
    z (list of ndarray, [ [N_1,], [N_2,], ..., [N_T,] ]): Associations
    x (ndarray, [T,] + o.dxGm): Global latent dynamic.
    theta (ndarray, [T, K, dxA]): K-Part Local dynamic.
    E (ndarray, [K, dxA, dxA]): K-Part Local Extent
    S (ndarray, [K, dxA, dxA]): K-Part Local Dynamic
    Q (ndarray, [dxA, dxA]): Latent Dynamic
    alpha (float): Concentration parameter (default: 0.01)
    pi (ndarray, [T, K+1]): stick weights, incl. unused portion
    mL (list of ndarray, [N_1, ..., N_T]): Marginal Likelihoods

  OUTPUT
    ll (float): joint log-likelihood
  '''
  ll = 0.0

  T,K = theta.shape[:2]
  if mL is None: mL = [ None for t in range(T) ]

  # E, S, Q, pi
  ll += logpdf_parameters(o, alpha, pi, E, S, Q)

  # time-dependent
  ## y
  ll += np.sum( [logpdf_data_t(o, y[t], z[t], x[t], theta[t], E, mL[t])
    for t in range(T) ])
  
  ## z
  ll += np.sum( [logpdf_assoc_t(o, z[t], pi) for t in range(T)] )

  ## x
  ll += logpdf_x_t(o, x[1], o.H_x[1], o.H_x[2])
  ll += np.sum( [logpdf_x_t(o, x[t], x[t-1], Q) for t in range(1,T)] )

  ## theta
  for k in range(K):
    ll += logpdf_theta_tk(o, theta[0,k], o.H_theta[1], o.H_theta[2])
    ll += np.sum( [logpdf_theta_tk(o, theta[t,k], theta[t-1,k], S[k])
      for t in range(1,T)] )

  return ll

def sampleTranslationX(o, y_t, z_t, x_t, theta_t, E, Q_tminus1, x_tminus1, **kwargs):
  r''' Sample translation component d_x_t of x_t, allowing for correlation.

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
  '''
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

  if kwargs.get('returnMuSigma', False): return mu, Sigma
  else: return mvn.rvs(mu, Sigma)

def sampleTranslationTheta(o, y_tk, theta_tk, x_t, S_tminus1_k, E_k, theta_tminus1_k, **kwargs):
  r''' Sample translation component d_theta_tk of theta_tk

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
  '''
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

  if kwargs.get('returnMuSigma', False): return mu, Sigma
  else: return mvn.rvs(mu, Sigma)

def sampleRotationX(o, y_t, z_t, x_t, theta_t, E, Q_tminus1, x_tminus1, **kwargs):
  r''' Sample translation component d_x_t of x_t

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
    handleSymmetries (bool): MCMC mode hop as a final step

  OUTPUT
    R_x_t (ndarray, [dy,]): Sampled global k translation dynamic, time t
  '''
  m = getattr(lie, o.lie)
  K = theta_t.shape[0]
  d_x_t = x_t[:-1,-1].copy()

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

      ll += logpdf_data_t(o, y_t, z_t, x_t_proposal, theta_t, E)
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

  # set us back to original translation
  ## todo: check if this is necessary
  x_t[:-1,-1] = d_x_t

  # handle rotation symmetries
  if kwargs.get('handleSymmetries', True):
    def dynamicsLL(x_t_candidate):
      ll = mvnL_logpdf(o, x_t_candidate, x_tminus1, Q_tminus1)
      if not noFuture: ll += mvnL_logpdf(o, x_tplus1, x_t_candidate, Q_tplus1)
      return ll
    
    # build all possible x_t_candidates from rotation symmetries
    x_t_candidates = util.rotation_symmetries(x_t)
    lls = [ dynamicsLL(x_t_candidate) for x_t_candidate in x_t_candidates ]
    x_t = x_t_candidates[np.argmax(lls)]

  R_x_t, _ = m.Rt(x_t)
  return R_x_t 

def sampleRotationTheta(o, y_tk, theta_tk, x_t, S_tminus1_k, E_k, theta_tminus1_k, **kwargs):
  r''' sample R_theta_tk | MB(R_theta_tk) using slice sampler

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
    handleSymmetries (bool): MCMC mode hop as a final step

  OUTPUT
    R_theta_tk (ndarray, [dy, dy]): Sampled rotation estimate.
  '''
  m = getattr(lie, o.lie)
  d_theta_tk = theta_tk[:-1,-1]

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
  z_tkFake = np.zeros(y_tk.shape[0], dtype=np.int) # fake z_t for logpdf_data_t
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
     
      ll += logpdf_data_t(o, y_tk, z_tkFake, x_t, theta_tk_proposal[np.newaxis],
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

  theta_tk[:-1,-1] = d_theta_tk
  # handle rotation symmetries
  if kwargs.get('handleSymmetries', True):
    def dynamicsLL(theta_tk_candidate):
      ll = mvnL_logpdf(o, theta_tk_candidate, theta_tminus1_k, S_tminus1_k)
      if not noFuture:
        ll += mvnL_logpdf(o, theta_tplus1_k, theta_tk_candidate, S_tplus1_k)
      return ll
    
    # build all possible x_t_candidates from rotation symmetries
    theta_tk_candidates = util.rotation_symmetries(theta_tk)
    lls = [ dynamicsLL(theta_tk_candidate) for theta_tk_candidate in
      theta_tk_candidates ]
    theta_tk = theta_tk_candidates[np.argmax(lls)]

  R_theta_tk, _ = m.Rt(theta_tk)
  return R_theta_tk

def inferZ(o, y_t, pi, theta_t, E, x_t, mL_t, **kwargs):
  r''' Sample z_{tn} | y_{tn}, pi, theta_t, E

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
  '''
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

def getComponentCounts(o, z_t, pi):
  r''' Count number of observations associated to each component.

  INPUT
    o (argparse.Namespace): Algorithm options
    z_t (ndarray, [N_t,]): Associations, -1,..,K-1
    pi (ndarray, [K+1,]): Local association priors for each time

  OUTPUT
    Nk (ndarray, [K+1,]): Observation counts
  '''
  K1 = len(pi)
  Nk = np.zeros(K1, dtype=np.int)
  uniq, cnts = np.unique(z_t, return_counts=True)
  for u, c in zip(uniq, cnts):
    uni = u if u != -1 else -1
    Nk[uni] = c
  return Nk

def consolidateExtantParts(o, z, pi, theta, E, S, **kwargs):
  r''' Remove parts with no associations, renumber existing parts to 0..(K-1).

  After calling this, pi is no longer valid. Must run inferPi.

  INPUT
    o (argparse.Namespace): Algorithm options
    z (list of ndarray, [ [N_1,], [N_2,], ..., [N_T,] ]): Associations
    pi (ndarray, [K+1,]): stick-breaking weights.
    theta (ndarray, [T, K,] + o.dxGm): K-Part Local dynamic.
    E (ndarray, [K, dy, dy]): K-Part Local Extent
    S (ndarray, [K, dxA, dxA]): K-Part Local Dynamic

  KEYWORD INPUT
    Nk (ndarray, [K+1,]): Pre-computed counts
    return_alive (bool): Return boolean array of which components survived.

  OUTPUT
    z (list of ndarray, [ [N_1,], [N_2,], ..., [N_T,] ]): Relabeled assoc.
    theta (ndarray, [T, K', dxA]): K'-Part Local dynamic.
    E (ndarray, [K', dxA, dxA]): K'-Part Local Extent
    S (ndarray, [K', dxA, dxA]): K'-Part Local Dynamic
  '''
  K, T = ( len(pi)-1, len(z) )
  Nk = kwargs.get('Nk', np.sum(
    [ getComponentCounts(o, z[t], pi) for t in range(T) ],
    axis=0
  ))
  # last entry of Nk is the number of unassigned observations

  # nonzero(Nk) gives the indices of nonzero elements
  missing = np.setdiff1d(range(K), np.nonzero(Nk)[0])
  uniq = np.setdiff1d(range(K), missing)

  if len(missing) > 0:
    # establish relabeling
    mapping = np.arange(K)
    for m in missing: mapping[m] = -1
    alive = mapping >= 0
    mapping[alive] = np.arange(len(uniq))

    # relabel z
    z_ = [ -1*np.ones_like(z[t]) for t in range(T) ]
    for k in uniq:
      for t in range(T):
        z_[t][z[t]==k] = mapping[k]

    # reorder parts
    theta = theta[:, alive]
    E = E[alive]
    S = S[alive]
    
  else:
    alive = np.ones(K, dtype=np.bool)
    z_ = z

  if kwargs.get('return_alive', False): return z_, theta, E, S, alive
  else: return z_, theta, E, S

def mergeComponents(o, theta, E, S, theta_star, E_star, S_star):
  """ Merge components (theta, E, S) and (theta_star, E_star, S_star)
  
  INPUT
    o (argparse.Namespace): Algorithm options
    theta (ndarray, [T, K, dt]): K-Part Local dynamic.
    E (ndarray, [K, dt, dt]): K-Part Local Extent
    S (ndarray, [K, dt, dt]): K-Part Local Dynamic
    theta_star (ndarray, [T, K_new, dt]): K_new Local dynamic.
    E_star (ndarray, [K_new, dt, dt]): K_new Local Extent
    S_star (ndarray, [K_new, dt, dt]): K_new Local Dynamic

  OUTPUT
    theta_merge (ndarray, [T, K + K_new, dt]): K-Part Local dynamic.
    E_merge (ndarray, [K + K_new, dt, dt]): K-Part Local Extent
    S_merge (ndarray, [K + K_new, dt, dt]): K-Part Local Dynamic
  """
  theta = np.concatenate((theta, theta_star), axis=1)
  E = np.concatenate((E, E_star), axis=0)
  S = np.concatenate((S, S_star), axis=0)
  return theta, E, S

def inferPi(o, Nk, alpha):
  r''' Sample pi

  INPUT
    o (argparse.Namespace): Algorithm options
    Nk (ndarray, [K+1,]): Observation counts, including unassigned counts.
    alpha (float): Concentration parameter for infinite sample

  OUTPUT
    pi (ndarray, [K+1]): mixture weights for K+1 parts
  '''
  cnts = Nk.astype(np.float)

  # todo: check if alpha should be added even if there are unassigned counts.
  if cnts[-1] == 0: cnts[-1] = alpha
  pi = dirichlet.rvs(cnts)[0]

  # enforce no zero entries
  if np.any(pi == 0):
    pi[pi==0] += 1e-32

    # renormalize in log domain
    logPi = np.log(pi)
    pi = np.exp(logPi - logsumexp(logPi))

  return pi

def inferE(o, x, theta, y, z):
  r''' Sample E_k from inverse wishart posterior, for all parts k

  INPUT
    o (argparse.Namespace): Algorithm options
    x (ndarray, [T, dxGf]): Global latent dynamics
    theta (ndarray, [T, K,] + dxGm): K-Part Local dynamic.
    y (list of ndarray, [ [N_1, dy], [N_2, dy], ..., [N_T, dy] ]): Observations
    z (list of ndarray, [ [N_1,], [N_2,], ..., [N_T,] ]): Associations

  OUTPUT
    E (ndarray, [K, dy, dy]): K-Part Local Extent (diagonal)
  '''
  m = getattr(lie, o.lie)
  T, K = theta.shape[:2]
  E = np.zeros((K, o.dy, o.dy))
  for k in range(K):
    v_E, S_E = o.H_E[1:]
    S_E = S_E.copy()
    for t in range(T):
      ztk = z[t]==k
      if np.sum(ztk) == 0: continue
      ytk_world = y[t][ztk]
      T_part_world = m.inv(x[t].dot(theta[t,k]))
      ytk_part = TransformPointsNonHomog(T_part_world, ytk_world)

      v_E, S_E = inferNormalInvWishart(v_E, S_E, ytk_part)
    E[k] = np.diag(np.diag(iw.rvs(v_E, S_E)))
  return E

def inferEk(o, x, theta, y, z, k):
  r''' Sample E_k from inverse wishart posterior

  INPUT
    o (argparse.Namespace): Algorithm options
    x (ndarray, [T, dxGf]): Global latent dynamics
    theta (ndarray, [T, K,] + dxGm): K-Part Local dynamic.
    y (list of ndarray, [ [N_1, dy], [N_2, dy], ..., [N_T, dy] ]): Observations
    z (list of ndarray, [ [N_1,], [N_2,], ..., [N_T,] ]): Associations
    k (int): index of part to infer

  OUTPUT
    E (ndarray, [K, dy, dy]): K-Part Local Extent (diagonal)
  '''
  m = getattr(lie, o.lie)
  T = theta.shape[0]
  E_k = np.zeros((o.dy, o.dy))

  v_E, S_E = o.H_E[1:]
  S_E = S_E.copy()
  for t in range(T):
    ztk = z[t]==k
    if np.sum(ztk) == 0: continue
    ytk_world = y[t][ztk]
    T_part_world = m.inv(x[t].dot(theta[t,k]))
    ytk_part = TransformPointsNonHomog(T_part_world, ytk_world)

    v_E, S_E = inferNormalInvWishart(v_E, S_E, ytk_part)
    E_k = np.diag(np.diag(iw.rvs(v_E, S_E)))
  return E_k

def inferSk(o, theta_k):
  r''' Sample S_k from inverse wishart posterior.

  INPUT
    o (argparse.Namespace): Algorithm options
    theta_k (ndarray, [T,] + dxGm): Part k latent dynamic.

  OUTPUT
    S_k (ndarray, [dxA, dxA]): Part k Local Dynamic
  '''
  m = getattr(lie, o.lie)
  T = len(theta_k)
  vS, GammaS = o.H_S[1:]
  vSkpost = vS + (T-1)

  theta_tkVec = np.zeros((T-1, o.dxA))
  for t in range(1, T):
    theta_tkVec[t-1] = m.algi(
      m.logm( m.inv(theta_k[t-1]).dot(theta_k[t]) )
    )

  GammaSkpost = GammaS + du.scatter_matrix(theta_tkVec, center=False)
  S_k = iw.rvs(vSkpost, GammaSkpost)

  # handle poorly conditioned S_k so we can invert it later
  if np.linalg.cond(S_k) >= 1.0 / np.finfo(S_k.dtype).eps:
    S_k += np.eye(o.dxA)*1e-6
  return S_k

def inferQ(o, x):
  r''' Sample Q from inverse wishart posterior.

  INPUT
    o (argparse.Namespace): Algorithm options
    x (ndarray, [T,] + o.dxGm): Global latent dynamic.

  OUTPUT
    Q (ndarray, [dxA, dxA]): Latent Dynamic
  '''
  m = getattr(lie, o.lie)
  vQ, GammaQ = o.H_Q[1:]
  T = len(x)
  vQpost = vQ + (T-1)
  xVec = np.zeros((T-1, o.dxA))

  xM = du.asShape(x, (T,) + o.dxGm)
  for t in range(1,T):
    xVec[t-1] = m.algi(m.logm(m.inv(xM[t-1]).dot(xM[t])))

  GammaQpost = GammaQ + du.scatter_matrix(xVec, center=False)
  Q = iw.rvs(vQpost, GammaQpost)

  # handle poorly conditioned Q so we can invert it later
  if np.linalg.cond(Q) >= 1.0 / np.finfo(Q.dtype).eps:
    Q += np.eye(o.dxA)*1e-6

  return Q

def samplePartFromPrior(o, T):
  r''' Sample new part from prior.
    
    Samples the following:
      (theta*_1, E*, S*) ~ ( H_theta, H_E, H_S )
      for t=2:T
        theta*_t ~ T_theta(theta*_{t-1}, S*)

  INPUT
    o (argparse.Namespace): Algorithm options
    T (int): Number timesteps

  OUTPUT
    theta (ndarray, [T,] + dxGm): Local dynamic.
    E (ndarray, [dxA, dxA]): Local Extent
    S (ndarray, [dxA, dxA]): Local Dynamic
  '''
  m = getattr(lie, o.lie)

  ## sample extent
  if o.H_E[0] == 'iw': E = np.diag(np.diag(iw.rvs(*o.H_E[1:])))
  else: assert False, 'only support H_E[0] == iw'

  ## sample dynamic
  if o.H_S[0] == 'iw': S = iw.rvs(*o.H_S[1:])
  else: assert False, 'only support H_S[0] == iw'

  ## sample transformations
  if o.H_theta[0] == 'mvnL':
    theta1 = m.expm(m.alg(mvn.rvs(m.algi(m.logm(o.H_theta[1])), o.H_theta[2])))
  else: assert False, 'only support H_theta[0] == mvnL'

  zero = np.zeros(o.dxA)
  theta = np.tile(theta1, (T, 1, 1))
  for t in range(1,T):
    s_tk = m.expm(m.alg(mvn.rvs(zero, S)))
    theta[t] = theta[t-1].dot(s_tk)
  return theta, E, S

def sampleKPartsFromPrior(o, T, K):
  r''' Sample K new parts from prior.
    
    Samples the following:
      (theta*_1, E*, S*) ~ ( H_theta, H_E, H_S )
      for t=2:T
        theta*_t ~ T_theta(theta*_{t-1}, S*)

  INPUT
    o (argparse.Namespace): Algorithm options
    T (int): Number timesteps
    K (int): integer >= 0

  OUTPUT
    theta (ndarray, [T, K, dt]): Local dynamic.
    E (ndarray, [K, dt, dt]): Local Extent
    S (ndarray, [K, dt, dt]): Local Dynamic
  '''
  if K > 1:
    parts = [ samplePartFromPrior(o,T) for k in range(K) ]
    theta, E, S = zip(*parts)
    E = np.stack(E)
    S = np.stack(S)
    theta = np.ascontiguousarray(np.swapaxes(np.stack(theta), 0, 1))
  elif K == 1:
    theta, E, S = samplePartFromPrior(o,T)
    E = E[np.newaxis]
    S = S[np.newaxis]
    theta = theta[:,np.newaxis]
  else:
    _theta, _E, _S = samplePartFromPrior(o,T)
    E = np.zeros( (0,) + _E.shape )
    S = np.zeros( (0,) + _S.shape )
    theta = np.zeros( (_theta.shape[0], 0, _theta.shape[1]) )
  return theta, E, S

def logMarginalPartLikelihoodMonteCarlo(o, y, x, theta, E, S):
  r''' Given x, Monte Carlo estimate log marginal likelihood of each y_{tn}

      {theta, E, S} are assumed to be a set of particles sampled from the prior
        p(theta_{1:T}, E, S) = p(E) p(S) \prod_t p(theta_t | theta_{t-1})

  INPUT
    o (argparse.Namespace): Algorithm options
    y (list of ndarray, [ [N_1, dy], [N_2, dy], ..., [N_T, dy] ]): Observations
    x (ndarray, [T,] + dxGm]): Global latent dynamic.
    theta (ndarray, [T,K*] + dxGm): Local dynamic.
    E (ndarray, [K*, dy, dy]): Local Extent
    S (ndarray, [K*, dxA, dxA]): Local Dynamic

  OUTPUT
    mL (list of ndarray, [ [N_1,], [N_2,], ..., [N_T,] ]): marginal LL of obs
  '''
  m = getattr(lie, o.lie)
  T, K = theta.shape[:2]

  zeroObs = np.zeros(o.dy)
  def obsLL(k):
    """ Return data ll (ndarray, [N_1, ..., N_T]) for part k. """
    ll = [ None for t in range(T) ]
    for t in range(T):
      y_tk_world = y[t]
      T_part_world = m.inv( x[t].dot(theta[t][k]) )
      y_tk_part = TransformPointsNonHomog(T_part_world, y_tk_world)
      ll[t] = mvn.logpdf(y_tk_part, zeroObs, E[k], allow_singular=True)
    return ll

  # K-list of T-list of N_t
  ## Average in log domain:
  ##   log( (1/K)*\sum_k a_k ) = lse( exp(log(a_{1:K})) ) - log(K)
  mL_all = [ obsLL(k) for k in range(K) ]

  mL = [ [] for t in range(T) ]
  for t in range(T):
    arr = np.array( [ mL_all[k][t] for k in range(K) ] ) # K x N_t
    mL[t] = logsumexp(arr, axis=0) - np.log(K) # N_t

  return mL

def saveSample(filename, o, alpha, z, pi, theta, E, S, x, Q, ll=np.nan):
  r''' Save sample to disk.

  INPUT
    filename (str): file to load
    o (argparse.Namespace): Algorithm options
    alpha (float): Concentration parameter, default 1e-3
    z (list of ndarray, [ [N_1,], [N_2,], ..., [N_T,] ]): Associations
    pi (ndarray, [T, K+1]): Local association priors for each time
    theta (ndarray, [T, K,] + dxGm): K-Part Local dynamic.
    E (ndarray, [K, dy, dy: K-Part Local Extent
    S (ndarray, [K, dxA, dxA]): K-Part Local Dynamic
    x (ndarray, [T,] + dxGm): Global latent dynamic.
    Q (ndarray, [K, dxA, dxA]): Latent Dynamic
    ll (float): joint log-likelihood
  '''
  dct = dict(o=o, alpha=alpha, z=z, pi=pi, theta=theta, E=E, S=S, x=x, Q=Q,
    ll=ll)
  du.save(filename, dct)

def loadSample(filename):
  r''' Load sample from disk.

  INPUT
    filename (str): file to load

  OUTPUT
    o (argparse.Namespace): Algorithm options
    alpha (float): Concentration parameter, default 1e-3
    z (list of ndarray, [ [N_1,], [N_2,], ..., [N_T,] ]): Associations
    pi (ndarray, [T, K+1]): Local association priors for each time
    theta (ndarray, [T, K,] + dxGm): K-Part Local dynamic.
    E (ndarray, [K, dy, dy]): K-Part Local Extent
    S (ndarray, [K, dxA, dxA]): K-Part Local Dynamic
    x (ndarray, [T,] + dxGm): Global latent dynamic.
    Q (ndarray, [K, dxA, dxA]): Latent Dynamic
    ll (float): joint log-likelihood
  ''' 
  d = du.load(filename)
  o, alpha, z, pi, theta, E, S, x, Q, ll = \
    (d['o'], d['alpha'], d['z'], d['pi'], d['theta'], d['E'], d['S'], d['x'],
     d['Q'], d['ll'])
  return o, alpha, z, pi, theta, E, S, x, Q, ll

def sampleNewPart(o, y, alpha, z, pi, theta, E, S, x, Q, mL, **kwargs):
  r''' Attempt to sample new part, initialize if sampled.

  INPUT
    o (argparse.Namespace): Algorithm options
    y (list of ndarray, [ [N_1, dy], [N_2, dy], ..., [N_T, dy] ]): Observations
    alpha (float): Concentration parameter, default 1e-3
    z (list of ndarray, [ [N_1,], [N_2,], ..., [N_T,] ]): Associations
    pi (ndarray, [T, K+1]): Local association priors for each time
    theta (ndarray, [T, K,] + dxGm): K-Part Local dynamic.
    E (ndarray, [K, dy, dy: K-Part Local Extent
    S (ndarray, [K, dxA, dxA]): K-Part Local Dynamic
    x (ndarray, [T,] + dxGm): Global latent dynamic.
    Q (ndarray, [K, dxA, dxA]): Latent Dynamic
    mL (list of ndarray, [N_1, ..., N_T]): Marginal Likelihoods

  KEYWORD INPUT
    minNonAssoc (int): Minimum # non-associated points to instantiate new part

  OUTPUT
    z (list of ndarray, [ [N_1,], [N_2,], ..., [N_T,] ]): Associations
    pi (ndarray, [T, K+1]): Local association priors for each time
    theta (ndarray, [T, K,] + dxGm): K-Part Local dynamic.
    E (ndarray, [K, dy, dy: K-Part Local Extent
    S (ndarray, [K, dxA, dxA]): K-Part Local Dynamic
  '''
  T, K = theta.shape[:2]
  minNonAssoc = kwargs.get('minNonAssoc', 1)
  assert minNonAssoc > 0

  # Any observations associated to base measure? If not, return
  nonAssoc = [ z[t] == -1 for t in range(T) ]
  nNonAssoc = np.array([ np.sum(nonAssoc[t]) for t in range(T) ])
  anyNonAssoc = np.sum(nNonAssoc) >= minNonAssoc
  if not anyNonAssoc: return z, pi, theta, E, S
  
  # Randomly select an observation to initialize on
  tInit = np.random.choice(np.where(nNonAssoc)[0])
  idx = np.random.choice(np.where(nonAssoc[tInit])[0])
  z[tInit][idx] = K

  # Make new part k, merge with existing parts. For time tInit, set new part to
  # identity rotation & translation of point it was initialized on
  k = K
  theta, E, S = mergeComponents(o, theta, E, S, *sampleKPartsFromPrior(o, T, 1))
  theta[tInit, k] = util.make_rigid_transform(np.eye(o.dy), y[tInit][idx])

  # Add new part to pi
  logPi = np.log(pi)
  logPi = np.concatenate((pi, np.array([np.log(alpha),])))
  pi = np.exp(logPi - logsumexp(logPi))
  assert np.isclose(np.sum(pi), 1.0)

  # propagate new part fwd, back, only change z and new part
  for t in reversed(range(tInit)):
    # propagate theta_{t+1} -> theta_t
    thetaPredict_t = theta[t+1]

    # associate
    zPredict = inferZ(o, y[t], pi, thetaPredict_t, E, x[t], mL[t])

    # sample new part rotation, translation
    y_tk = y[t][zPredict==k]
    thetaPredict_t[k,:-1,:-1] = sampleRotationTheta(o, y_tk,
      thetaPredict_t[k], x[t], S[k], E[k], theta[t+1,k])
    thetaPredict_t[k,:-1,-1] = sampleTranslationTheta(o, y_tk,
      thetaPredict_t[k], x[t], S[k], E[k], theta[t+1,k])
    theta[t] = thetaPredict_t

    # re-associate
    z[t] = inferZ(o, y[t], pi, theta[t], E, x[t], mL[t])

  for t in range(tInit+1, T):
    # propagate theta_{t-1} -> theta_t
    thetaPredict_t = theta[t-1]
    
    # associate
    zPredict = inferZ(o, y[t], pi, thetaPredict_t, E, x[t], mL[t])

    # sample part rotation, translation
    y_tk = y[t][zPredict==k]
    thetaPredict_t[k,:-1,:-1] = sampleRotationTheta(o, y_tk,
      thetaPredict_t[k], x[t], S[k], E[k], theta[t-1,k])
    thetaPredict_t[k,:-1,-1] = sampleTranslationTheta(o, y_tk,
      thetaPredict_t[k], x[t], S[k], E[k], theta[t-1,k])
    theta[t] = thetaPredict_t

    # re-associate
    z[t] = inferZ(o, y[t], pi, theta[t], E, x[t], mL[t])

  # infer S_k, E_k
  S[k] = inferSk(o, theta[:,k])
  E[k] = inferEk(o, x, theta, y, z, k)

  # # Remove any dead parts
  z, pi, theta, E, S = consolidatePartsAndResamplePi(o, z, pi, alpha, theta,
    E, S)

  assert theta.shape[1] == len(pi)-1
  assert E.shape[0] == len(pi)-1
  assert S.shape[0] == len(pi)-1
  assert np.isclose(np.sum(pi), 1.0)

  return z, pi, theta, E, S

def sampleStepFC(o, y, alpha, z, pi, theta, E, S, x, Q, mL, **kwargs):
  r''' Perform full gibbs ssampling pass. Input values will be overwritten.

  INPUT
    o (argparse.Namespace): Algorithm options
    y (list of ndarray, [ [N_1, dy], [N_2, dy], ..., [N_T, dy] ]): Observations
    alpha (float): Concentration parameter, default 1e-3
    z (list of ndarray, [ [N_1,], [N_2,], ..., [N_T,] ]): Associations
    pi (ndarray, [T, K+1]): Local association priors for each time
    theta (ndarray, [T, K,] + dxGm): K-Part Local dynamic.
    E (ndarray, [K, dy, dy: K-Part Local Extent
    S (ndarray, [K, dxA, dxA]): K-Part Local Dynamic
    x (ndarray, [T,] + dxGm): Global latent dynamic.
    Q (ndarray, [K, dxA, dxA]): Latent Dynamic
    mL (list of ndarray, [N_1, ..., N_T]): Marginal Likelihoods

  KEYWORD INPUT
    newPart (bool): Allow new part sampling

  OUTPUT
    z (list of ndarray, [ [N_1,], [N_2,], ..., [N_T,] ]): Associations
    pi (ndarray, [T, K+1]): Local association priors for each time
    theta (ndarray, [T, K,] + dxGm): K-Part Local dynamic.
    E (ndarray, [K, dy, dy: K-Part Local Extent
    S (ndarray, [K, dxA, dxA]): K-Part Local Dynamic
    x (ndarray, [T,] + dxGm): Global latent dynamic.
    Q (ndarray, [K, dxA, dxA]): Latent Dynamic
    ll (float): joint log-likelihood
  '''
  T, K = ( len(y), len(pi)-1 )

  # sample new part, propagate
  if kwargs.get('newPart', True):
    z, pi, theta, E, S = sampleNewPart(o, y, alpha, z, pi, theta, E, S, x, Q,
      mL, **kwargs)
    K = len(pi) - 1

  for t in range(T):
    for k in range(K):
      # sample theta_tk
      if t==0: thetaPrev, SPrev = ( o.H_theta[1], o.H_theta[2] )
      else: thetaPrev, SPrev = ( theta[t-1,k], S[k] )
      if t==T-1: thetaNext, SNext = ( None, None )
      else: thetaNext, SNext = ( theta[t+1,k], S[k] )
      thetaPredict = theta[t,k]
      y_tk = y[t][z[t]==k]

      thetaPredict[:-1,:-1] = sampleRotationTheta(o, y_tk, thetaPredict, x[t],
        SPrev, E[k], thetaPrev, theta_tplus1_k=thetaNext, S_tplus1_k=SNext)
      thetaPredict[:-1,-1] = sampleTranslationTheta(o, y_tk, thetaPredict, x[t],
        SPrev, E[k], thetaPrev, theta_tplus1_k=thetaNext, S_tplus1_k=SNext)

      theta[t,k] = thetaPredict

    # sample x_t
    if t==0: xPrev, QPrev = ( o.H_x[1], o.H_x[2] )
    else: xPrev, QPrev = ( x[t-1], Q )
    if t==T-1: xNext, QNext = ( None, None )
    else: xNext, QNext = ( x[t+1], Q )
    xPredict = x[t]

    xPredict[:-1,:-1] = sampleRotationX(o, y[t], z[t], xPredict, theta[t], E,
      QPrev, xPrev, x_tplus1=xNext, Q_tplus1=QNext)
    xPredict[:-1,-1] = sampleTranslationX(o, y[t], z[t], xPredict, theta[t], E,
      QPrev, xPrev, x_tplus1=xNext, Q_tplus1=QNext)
    x[t] = xPredict

    # sample z_t
    z[t] = inferZ(o, y[t], pi, theta[t], E, x[t], mL[t])

  z, pi, theta, E, S = consolidatePartsAndResamplePi(o, z, pi, alpha, theta,
    E, S)
  K = len(pi) - 1

  # sample E, S, Q
  Q = inferQ(o, x)

  if K > 0: S = np.array([ inferSk(o, theta[:,k]) for k in range(K) ])
  else: S = np.zeros((0, o.dxA, o.dxA))
  E = inferE(o, x, theta, y, z)

  # compute log-likelihood
  ll = logJoint(o, y, z, x, theta, E, S, Q, alpha, pi, mL)

  return z, pi, theta, E, S, x, Q, ll

def consolidatePartsAndResamplePi(o, z, pi, alpha, theta, E, S):
  """ Remove unassociated parts, relabel/sort z, theta, E, S and resample pi

  INPUT
    z (list of ndarray, [ [N_1,], [N_2,], ..., [N_T,] ]): Associations
    pi (ndarray, [T, K+1]): stick weights, incl. unused portion
    alpha (float): Concentration parameter (default: 0.01)
    theta (ndarray, [T, K, dxA]): K-Part Local dynamic.
    E (ndarray, [K, dxA, dxA]): K-Part Local Extent
    S (ndarray, [K, dxA, dxA]): K-Part Local Dynamic

  OUTPUT
    z (list of ndarray, [ [N_1,], [N_2,], ..., [N_T,] ]): Associations
    pi (ndarray, [T, K'+1]): stick weights, incl. unused portion
    theta (ndarray, [T, K', dxA]): K-Part Local dynamic.
    E (ndarray, [K', dxA, dxA]): K-Part Local Extent
    S (ndarray, [K', dxA, dxA]): K-Part Local Dynamic
  """
  T = len(z)
  Nk = np.sum([getComponentCounts(o, z[t], pi) for t in range(T)], axis=0)
  z, theta, E, S, alive = consolidateExtantParts(o, z, pi, theta, E, S,
    return_alive=True)
  alive = np.concatenate((alive, np.array([True,])))
  Nk = Nk[alive]
  pi = inferPi(o, Nk, alpha)
  K = len(pi) - 1
  return z, pi, theta, E, S

# todo: move to utils
def MakeRd(R, d):
  r''' Make SE(D) element from rotation matrix and translation vector.

  INPUT
    R (ndarray, [d, d]): Rotation matrix
    d (ndarray, [d,]): translation vector

  OUTPUT
    Rd (ndarray, [d+1, d+1]: Homogeneous Rotation + translation matrix
  '''
  bottomRow = np.hstack( [np.zeros_like(d), np.array([1.])] )
  return np.block([ [R, d[:, np.newaxis]], [bottomRow] ])

# todo: move to utils
def TransformPointsNonHomog(T, y):
  r''' Transform non-honogeneous points y with transformation T.

  INPUT
    T (ndarray, [dz, dz]): Homogeneous transformation matrix
    y (ndarray, [N, dz-1]): Non-homogeneous points

  OUTPUT
    yp (ndarray, [N, dz-1]): Transformed non-homogeneous points
  '''
  R = T[:-1, :-1]
  d = T[:-1, -1][np.newaxis]
  if y.ndim == 1: y = y[np.newaxis]
  yp = np.squeeze(y.dot(R.T) + d)
  return yp

def inferNormalNormal(y, SigmaY, muX, SigmaX, H=None, b=None):
  r''' Conduct a conjugate Normal-Normal update on the below system:

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
  '''
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
  r''' Return mu, Sigma in x1 | x2, H, b, u, S, x2_, H_, b_, u_, S_, where
    
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
  '''
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
  r''' Return mu, Sigma in x1 | x2, H, b, u, S, where
    
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
  '''
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

def inferNormalInvWishart(df, S, y):
  r''' Conduct a conjugate Normal-Inverse-Wishart update on the below system:

  p(Sigma | df, S, y) \propto p(Sigma | df, S)   p(y | Sigma)
                            = IW(Sigma | df, S)  \prod_i N(y_i | 0, Sigma)

  INPUT
    df (float): prior degrees of freedom
    S (ndarray, [dy, dy]): prior scale matrix
    y (ndarray, [N, dy]): centered observations

  OUTPUT
    df_prime (float): posterior degrees of freedom
    S_prime (ndarray, [dy, dy]): posterior scale matrix
  ''' 
  df_prime = df + y.shape[0]
  S_prime = S + du.scatter_matrix(y, center=False)
  return (df_prime, S_prime)

def mvnL_logpdf(o, x, mu, Sigma):
  r''' Evaulate log N_L(x | mu, Sigma) = N( log(mu^{-1} x) | 0, Sigma )

  INPUT
    o (argparse.Namespace): Algorithm options
    x (ndarray, o.dxGm): Manifold input (as matrix)
    mu (ndarray, o.dxGm): Manifold mean (as matrix)
    Sigma (ndarray, o.dxGm): Covariance (in tangent plane)

  OUTPUT
    ll (float): log N_L(x | mu, Sigma)
  ''' 
  m = getattr(lie, o.lie)
  return mvn.logpdf(m.algi(m.logm(m.inv(mu).dot(x))), np.zeros(o.dxA), Sigma,
    allow_singular=True)
