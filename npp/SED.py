from . import util, sample
from . import hmm

import numpy as np, argparse, lie, warnings
from scipy.linalg import block_diag
from sklearn.mixture import BayesianGaussianMixture as dpgmm 
from sklearn.mixture import GaussianMixture as gmm
from scipy.stats import multivariate_normal as mvn, invwishart as iw
from scipy.stats import dirichlet, beta as Be
from scipy.special import logsumexp, gammaln
import du, du.stats # move useful functions from these to utils (catrnd)
import functools, itertools

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

  omega = np.tile(np.eye(o.dy+1), (K, 1, 1))
  E = inferE(o, x, theta, omega, y, z)

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

def logpdf_assoc(o, z, pi, alpha):
  r''' Return log-likelihood of p(pi, z | alpha) = p(pi | z, alpha) p(z | alpha)

  INPUT
    o (argparse.Namespace): Algorithm options
    z (list of ndarray, [ [N_1,], [N_2,], ..., [N_T,] ]): Associations
    pi (ndarray, [K+1,]): stick weights, incl. unused portion
    alpha (float): Concentration parameter

  OUTPUT
    ll (float): log-likelihood
  '''
  # get counts
  ll = 0.0
  K = len(pi)-1
  T = len(z)

  # pi | z, alpha
  Nk = []
  idx = []
  for k in range(K):
    s = 0
    for t in range(T): s += np.sum(z[t]==k)
    if s > 0:
      Nk.append(s)
      idx.append(k)
  s = 0
  for t in range(T): s += np.sum(z[t]==-1)
  if s > 0: Nk.append(s)
  else: Nk.append(alpha)
  idx.append(K)

  piInputUnnormalized = pi[idx]
  logPiInputUnnormalized = np.log(piInputUnnormalized)
  piInput = np.exp(logPiInputUnnormalized - logsumexp(logPiInputUnnormalized))

  assert len(piInput) == len(Nk)
  assert np.isclose(np.sum(piInput), 1.0)
  assert np.all(piInput>0)
  ll += dirichlet.logpdf(piInput, Nk)

  # z | alpha
  ll += K * np.log(alpha)
  ll += gammaln(alpha)
  ll += np.sum( gammaln(Nk) )
  ll -= gammaln( np.sum(Nk) + alpha )

  return ll

def logpdf_parameters(o, E, S, Q):
  r''' Return log-likelihood, E, S, Q, pi | alpha, H_*

  INPUT
    o (argparse.Namespace): Algorithm options
    E (ndarray, [K, dxA, dxA]): K-Part Local Extent
    S (ndarray, [K, dxA, dxA]): K-Part Local Dynamic
    Q (ndarray, [dxA, dxA]): Latent Dynamic

  OUTPUT
    ll (float): log-likelihood for parameters
  '''
  K = E.shape[0]
  ll = 0.0

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

def logJoint(o, y, z, x, theta, E, S, Q, alpha, pi, omega, mL=None):
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

  # E, S, Q | H_E, H_S, H_Q
  ll += logpdf_parameters(o, E, S, Q)

  # z, pi | alpha
  ll += logpdf_assoc(o, z, pi, alpha)

  # time-dependent
  ## y | z, theta, x, E
  if omega is None:
    ll += np.sum( [logpdf_data_t(o, y[t], z[t], x[t], theta[t], E, mL[t])
      for t in range(T) ])
  else:
    # make simulated parts
    for t in range(T):
      theta_t = np.stack([ omega[k] @ theta[t,k] for k in range(K) ])
      ll += np.sum(logpdf_data_t(o, y[t], z[t], x[t], theta_t, E, mL[t]))
  
  if omega is not None:
    I = np.eye(o.dy+1)
    cov = 1e6 * np.eye(o.dxA)
    ll += np.sum( [ mvnL_logpdf(o, omega[k], I, cov) for k in range(K) ] )

  ## x | Q, H_x
  ll += logpdf_x_t(o, x[1], o.H_x[1], o.H_x[2])

  ll += np.sum( [logpdf_x_t(o, x[t], x[t-1], Q) for t in range(1,T)] )

  ## theta | S, H_theta
  for k in range(K):
    ll += logpdf_theta_tk(o, theta[0,k], o.H_theta[1], o.H_theta[2])
    ll += np.sum( [logpdf_theta_tk(o, theta[t,k], theta[t-1,k], S[k])
      for t in range(1,T)] )

  return ll

def translationObservationPosterior(o, ys, R_U, lhs, rhs, E, mu, Sigma):
  r''' Compute Gaussian posterior on translation d_U for given observation model

  This function assumes the following observation model (where points are
  interchangeably viewed in homogenous coordinates or not for simplicity):

    \prod_n
    N( ys[n] | lhs * U * rhs * 0, (lhs * U * rhs) * E * (lhs * U * rhs)^T )

  = \prod_n
    N( lhs^{-1} ys[n] | U * rhs * 0, (U * rhs) * E * (U * rhs)^T )

  = \prod_n
    N( lhs^{-1} ys[n] | U * phi, (U * rhs) * E * (U * rhs)^T )
  
  ( for phi = lhs * 0 )

  = \prod_n
    N( lhs^{-1} ys[n] | d_U + R_U phi, (U * rhs) * E * (U * rhs)^T )

  ( for SE(D) transformation U = (R_U, d_U) )

  Given prior mu, Sigma on dU, this is a linear Gaussian system with bias
  parameter R_U * phi.

  INPUT
    o (argparse.Namespace): Algorithm options
    ys (ndarray, [N, dy]): Set of observations
    R_U (ndarray, [dy, dy]): Rotation component of U
    lhs (ndarray, dxGm): Global transformation to the left of U
    rhs (ndarray, dxGm): Global transformation to the right of U
    E (ndarray, [dy, dy]): Part Local Observation Noise Covs
    mu (ndarray, [dy,]): Prior mean of d_U, the translation of U
    Sigma (ndarray, [dy, dy]): Prior covariance of d_U, the translation of U

  OUTPUT
    mu (ndarray, [dy,]): Posterior mean of d_U, the translation of U
    Sigma (ndarray, [dy, dy]): Posterior covariance of d_U, the translation of U
  '''
  m = getattr(lie, o.lie)
  ys = np.atleast_2d(ys)
  N = ys.shape[0]
  if N == 0: return mu, Sigma

  zero = np.zeros(o.dy)
  y_lhs = TransformPointsNonHomog(m.inv(lhs), ys)
  phi = TransformPointsNonHomog(rhs, zero)
  bias = TransformPointsNonHomog(MakeRd(R_U, zero), phi)

  R_rhs, _ = m.Rt(rhs)
  R2 = R_U @ R_rhs
  obsCov = R2 @ E @ R2.T
  E_inv = np.diag(1. / np.diag(E))
  obsCov_inv = R2 @ E_inv @ R2.T

  Sigma_inv = np.linalg.inv(Sigma)
  SigmaNew_inv = Sigma_inv + N * obsCov_inv
  SigmaNew = np.linalg.inv(SigmaNew_inv)
  muNew = SigmaNew.dot( Sigma_inv.dot(mu) + obsCov_inv.dot(
    np.sum(y_lhs, axis=0) - N*bias)
  )

  return muNew, SigmaNew


def sampleOmega(o, y, z, x, theta, E, R_omega):
  r''' Sample Canonical part transformation for each part.

  INPUT
    R_omega (ndarray, [K, dy, dy]): Initial omega rotations

  OUTPUT
    omega (ndarray, [K, dxGm]): Resampled omega
  '''
  T, K = theta.shape[:2]

  # prior parameters; todo: set into opts
  prevU = np.eye(o.dy+1)
  prevS = 1e6 * np.eye(o.dxA)

  omega = np.zeros((K,) + o.dxGm)

  for k in range(K):
    # sample translation

    ## dynamics from prior
    mu, Sigma = translationDynamicsPosterior(o, R_omega[k], prevU, prevS)

    ## observations across time
    y_ks = tuple( y[t][z[t]==k] for t in range(T) )
    lhs = tuple( x[t] for t in range(T) )
    rhs = tuple( theta[t,k] for t in range(T) )
    for t in range(T):
      mu, Sigma = translationObservationPosterior(o, y_ks[t], R_omega[k],
        lhs[t], rhs[t], E[k], mu, Sigma)
    d_omega_k = mvn.rvs(mu, Sigma)

    # sample rotation
    E_k = ( E[k], ) * T
    R_omega_k = sampleRotation(o, y_ks, d_omega_k, lhs, rhs, E_k, prevU, prevS)

    omega[k] = MakeRd(R_omega_k, d_omega_k)

  return omega

def thetaTranslationDynamicsPosterior(o, R_U, prevU, prevS, A, B, **kwargs): 
  m = getattr(lie, o.lie)
  if o.lie == 'se2': mRot = lie.so2
  else: mRot = lie.so3

  zero, eye = ( np.zeros(o.dy), np.eye(o.dy) )

  # handle previous
  ## Get p( V^{-1} d_{theta_tk} | R_{theta_tk} )
  R_prev, d_prev = m.Rt(prevU)

  # new
  H = B
  b = A @ d_prev
  phi = np.atleast_1d(mRot.algi(mRot.logm(R_prev.T @ R_U)))

  jointMu = np.concatenate(( b, np.zeros(o.dxA - o.dy) ))
  jointSigma = prevS.copy() # 22 stays the same
  jointSigma[:o.dy,:o.dy] = H @ jointSigma[:o.dy,:o.dy] @ H.T # 11
  jointSigma[:o.dy,o.dy:] = H @ jointSigma[:o.dy,o.dy:] # 12
  jointSigma[o.dy:,:o.dy] = jointSigma[:o.dy,o.dy:].T # 21

  # prior on d_theta_tk
  mu, Sigma = inferNormalConditional(phi, eye, zero,
    jointMu, jointSigma)

  # handle future or return
  nextU = kwargs.get('nextU', None)
  if nextU is None: return mu, Sigma
  R_nextU, d_nextU = m.Rt(nextU)
  nextS = kwargs.get('nextS', None)
  if nextS is None: nextS = prevS

  Ai = np.linalg.inv(A)
  H = -Ai @ B
  b = Ai @ d_nextU

  _, phi = m.getVi(R_U.T @ R_nextU, return_phi=True)
  if o.lie == 'se2': phi = np.array([phi])

  jointMu = np.concatenate((
    b,
    np.zeros(o.dxA - o.dy)
  ))
  jointSigma = nextS.copy() # 22 stays the same
  jointSigma[:o.dy,:o.dy] = H @ jointSigma[:o.dy,:o.dy] @ H.T # 11
  jointSigma[:o.dy,o.dy:] = H @ jointSigma[:o.dy,o.dy:] # 12
  jointSigma[o.dy:,:o.dy] = jointSigma[:o.dy,o.dy:].T # 21

  # treat this conditional mu as an observation of d_theta_tk with observation
  # covariance Sigma_
  obs, Sigma_ = inferNormalConditional(phi, eye, zero,
    jointMu, jointSigma)

  # combine prior and observation
  mu, Sigma = inferNormalNormal(obs, Sigma_, mu, Sigma)
  return mu, Sigma

def translationDynamicsPosterior(o, R_U, prevU, prevS, **kwargs):
  r''' Compute Gaussian posterior on translation d_U for given dynamics model

  This function assumes the following dynamics model

    p(d_U | R_U) \propto N_L(U | prevU, prevS) * N_L(nextU | U, nextS)

  ( for SE(D) transformation U = (R_U, d_U) )

  INPUT
    o (argparse.Namespace): Algorithm options
    R_U (ndarray, [dy, dy]): Rotation component of U
    prevU (ndarray, [dxGm]): Previous dynamics
    prevS (ndarray, [dxA, dxA]): Previous dynamics covariance

  Keywords
    nextU (ndarray, [dxGm]): Next dynamics
    nextS (ndarray, [dxA, dxA]): Next dynamics covariance

  OUTPUT
    mu (ndarray, [dy,]): Posterior mean of d_U, the translation of U
    Sigma (ndarray, [dy, dy]): Posterior covariance of d_U, the translation of U
  '''
  m = getattr(lie, o.lie)
  R_prevU, d_prevU = m.Rt(prevU)

  nextU = kwargs.get('nextU', None)
  if nextU is not None:
    useFuture = True
    R_nextU, d_nextU = m.Rt(nextU)
    nextS = kwargs.get('nextS', None)
    if nextS is None: nextS = prevS

  else:
    useFuture = False

  # get V^{-1}
  prevV_inv, phi = m.getVi(R_prevU.T @ R_U, return_phi=True)
  if o.lie == 'se2': phi = np.array([phi])

  H = prevV_inv @ R_prevU.T
  b = -H @ d_prevU
  u = np.zeros(o.dxA)

  if useFuture:
    nextV_inv, phi_ = m.getVi(R_U.T @ R_nextU, return_phi=True)
    if o.lie == 'se2': phi_ = np.array([phi_])
    H_ = -nextV_inv @ R_U.T
    b_ = -H_ @ d_nextU
    u_ = u
    mu, Sigma = inferNormalConditionalNormal(phi, H, b, u, prevS, phi_, H_, b_,
      u_, nextS)
  else:
    mu, Sigma = inferNormalConditional(phi, H, b, u, prevS)

  return mu, Sigma


def sampleRotation(o, ys, d_U, lhs, rhs, E, prevU, prevS, **kwargs):
  r''' Sample rotation component R_U of SE(D) transformation under given model

  Assumed model is:

                                            optional
                                    { - - - - - - - - - -  }
     ( V_inv  d_U    |          )    
    N(               | 0, prevS ) * N_L ( U | nextU, nextS )
     ( phi           |          )    
    *
    \prod_n
    N(ys[n] | lhs * U * rhs * 0, (lhs * U * rhs) * E * (lhs * U * rhs)^T
  
  Where V_inv, phi depend on prevU, rotation R_U
  '''
  m = getattr(lie, o.lie)
  nSamples = kwargs.get('nSamples', 10)

  if type(lhs) is not tuple: lhs = ( lhs, )
  if type(rhs) is not tuple: rhs = ( rhs, )
  if type(ys) is not tuple: ys = ( ys, )
  if type(E) is not tuple: E = ( E, )
  K = len(ys)
  assert K == len(E) # ys, E must be same length
  if len(rhs) == 1 and K > 1: rhs = rhs * K
  if len(lhs) == 1 and K > 1: lhs = lhs * K

  nextU = kwargs.get('nextU', None)
  if nextU is not None:
    useFuture = True
    nextS = kwargs.get('nextS', None)
    if nextS is None: nextS = prevS
    R_nextU, d_nextU = m.Rt(nextU)
  else:
    useFuture = False

  # Sample R_U | MB(R_U) using slice sampler
  if o.lie == 'se2':
    dof = 1
    mRot = lie.so2
  elif o.lie == 'se3':
    dof = 3
    mRot = lie.so3

  A = kwargs.get('A', None)
  B = kwargs.get('B', None)
  if A is not None:
    assert B is not None
    Bi = np.linalg.inv(B)
    controlled = True

  ## setup
  zeroAlgebra = np.zeros(o.dxA)
  zeroObs = np.zeros(o.dy)
  R_prevU, d_prevU = m.Rt(prevU)

  # slice sampler parameters
  w = kwargs.get('w', np.pi / 100) # characteristic width
  P = kwargs.get('P', 10) # max doubling iterations

  rotIdx = np.arange(dof)
  phiEst = np.zeros(dof)
  for idx in rotIdx:

    # inlined functional, making use of closures
    def slice_conditional(idx, phiEst, a):
      ll = 0.0

      # rotation in lie Algebra
      phi = phiEst.copy() 
      phi[idx] = a
        
      # rotation in lie Group 
      R_U = R_prevU @ mRot.expm(mRot.alg(phi))

      # Construct group transformation U
      U = MakeRd(R_U, d_U)

      if controlled:
        m_t = Bi @ (d_U - A @ d_prevU)
        val = np.concatenate((m_t, phi))
        ll += mvn.logpdf(val, zeroAlgebra, prevS)
        if useFuture:
          m_t1 = Bi @ (d_U - A @ d_prevU)
          phi_ = np.atleast_1d(mRot.algi(mRot.logm(R_U.T @ R_nextU)))
          val_ = np.concatenate((m_t1, phi_))
          ll += mvn.logpdf(val_, zeroAlgebra, nextS)
      else:
        # Linear operator on translation d_U, a fcn of R_{prevU}^{-1} * R_U
        # slightly wasteful, but keep for code clarity
        V_inv = m.getVi( R_prevU.T @ R_U )

        # evaluate first time
        d_prevU_inv_U = V_inv @ R_prevU.T @ (d_U - d_prevU)
        tangent_vector = np.concatenate((V_inv @ d_prevU_inv_U, phi))
        ll += mvn.logpdf(tangent_vector, zeroAlgebra, prevS)

        # evaluate second term (if provided)
        # todo: I think U, nextU should be flipped?
        if useFuture: ll += mvnL_logpdf(o, U, nextU, nextS) 

      # evaluate observations (if available)
      for k in range(K):
        ys_k = ys[k]
        if ys_k.shape[0] == 0: continue
        
        # ip.embed()
        yLocal_k = TransformPointsNonHomog(m.inv(lhs[k] @ U @ rhs[k]), ys_k)
        ll += np.sum(mvn.logpdf(yLocal_k, zeroObs, E[k]))
      return ll

    # end full conditional functional
    llFunc = functools.partial(slice_conditional, idx, phiEst)
    samples, _ll = sample.slice_univariate(llFunc, phiEst[idx],
      w=w, P=P, nSamples=nSamples, xmin=-np.pi, xmax=np.pi)

    # samples are univariate floats so we need to plug new sampled algebra
    # coordinates into phiEst
    phiEst[idx] = samples[-1]
  # end loop over rotation coordinates
  
  # construct rotation from phiEst and d_U
  R_U = R_prevU @ mRot.expm(mRot.alg(phiEst))
  U = MakeRd(R_U, d_U)

  # handle rotation symmetries
  if kwargs.get('handleSymmetries', True):
    if controlled:
      def dynamicsLL(U_candidate):
        R_cand, d_cand = m.Rt(U_candidate)
        m_t = Bi @ (d_cand - A @ d_prevU)
        phi = np.atleast_1d(mRot.algi(mRot.logm(R_prevU.T @ R_cand)))
        val = np.concatenate((m_t, phi))
        ll = mvn.logpdf(val, zeroAlgebra, prevS)
        if useFuture:
          m_t1 = Bi @ (d_nextU - A @ d_cand)
          phi_ = np.atleast_1d(mRot.algi(mRot.logm(R_cand.T @ R_nextU)))
          val_ = np.concatenate((m_t1, phi_))
          ll += mvn.logpdf(val_, zeroAlgebra, nextS)
        return ll
    else:
      def dynamicsLL(U_candidate):
        ll = mvnL_logpdf(o, U_candidate, prevU, prevS)
        if useFuture: ll += mvnL_logpdf(o, nextU, U_candidate, nextS)
        return ll
    
    # build all possible x_t_candidates from rotation symmetries
    U_candidates = util.rotation_symmetries(U)

    lls = [ dynamicsLL(U_candidate) for U_candidate in U_candidates ]
    U = U_candidates[np.argmax(lls)]

  R_U, _ = m.Rt(U)
  return R_U


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

# not being called anymore, todo: delete and refactor tests
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

def inferE(o, x, theta, omega, y, z):
  r''' Sample E_k from inverse wishart posterior, for all parts k

  INPUT
    o (argparse.Namespace): Algorithm options
    x (ndarray, [T, dxGf]): Global latent dynamics
    theta (ndarray, [T, K,] + dxGm): K-Part Local dynamic.
    omega (ndarray, [K,] + dxGm): K-Part Canonical dynamic.
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
      T_part_world = m.inv(x[t] @ omega[k] @ theta[t,k])
      ytk_part = TransformPointsNonHomog(T_part_world, ytk_world)
      v_E, S_E = inferNormalInvWishart(v_E, S_E, ytk_part)
    E[k] = np.diag(np.diag(iw.rvs(v_E, S_E)))
  return E

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

def saveSample(filename, o, alpha, z, pi, theta, E, S, x, Q, omega, mL, ll,
  subsetIdx=None, dataset=None):
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
    theta (ndarray, [K,] + dxGm): K-Part Canonical dynamic.
    mL
    ll (float): joint log-likelihood
    subsetIdx (list of ndarray, [ [N_1,], ... [N_T,] ]): Observation subset
    dataset (string): Path to dataset
  '''
  dct = dict(o=o, alpha=alpha, z=z, pi=pi, theta=theta, E=E, S=S, x=x, Q=Q,
    omega=omega, mL=mL, ll=ll, subsetIdx=subsetIdx, dataset=dataset)
  du.save(filename, dct)
  # dct = dict(o=o, alpha=alpha, z=z, pi=pi, theta=theta, E=E, S=S, x=x, Q=Q,
  #   ll=ll)
  # du.save(filename, dct)

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
    mL
    ll (float): joint log-likelihood
    subsetIdx (list of ndarray, [ [N_1,], ... [N_T,] ]): Observation subset
    dataset (string): Path to dataset
  ''' 
  d = du.load(filename)
  o, alpha, z, pi, theta, E, S, x, Q, omega, mL, ll = \
    (d['o'], d['alpha'], d['z'], d['pi'], d['theta'], d['E'], d['S'], d['x'],
     d['Q'], d['omega'], d['mL'], d['ll'])
  subsetIdx, dataset = ( d.get('subsetIdx', None), d.get('dataset', None) )
  return o, alpha, z, pi, theta, E, S, x, Q, omega, mL, ll, subsetIdx, dataset

def sampleStepFC(o, y, alpha, z, pi, theta, E, S, x, Q, omega, mL, **kwargs):
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
    omega (ndarray, [K,] + dxGm): K-Part Canonical dynamic.
    mL (list of ndarray, [N_1, ..., N_T]): Marginal Likelihoods

  KEYWORD INPUT
    newPart (bool): Allow new part sampling
    dontSampleX (bool): Don't allow x, Q resampling

  OUTPUT
    z (list of ndarray, [ [N_1,], [N_2,], ..., [N_T,] ]): Associations
    pi (ndarray, [T, K+1]): Local association priors for each time
    theta (ndarray, [T, K,] + dxGm): K-Part Local dynamic.
    E (ndarray, [K, dy, dy: K-Part Local Extent
    S (ndarray, [K, dxA, dxA]): K-Part Local Dynamic
    x (ndarray, [T,] + dxGm): Global latent dynamic.
    Q (ndarray, [K, dxA, dxA]): Latent Dynamic
    omega (ndarray, [K,] + dxGm): K-Part Canonical dynamic.
    ll (float): joint log-likelihood
  '''
  m = getattr(lie, o.lie)
  T, K = ( len(y), len(pi)-1 )
  dontSampleX = kwargs.get('dontSampleX', False)

  # # sample new part, propagate
  # if kwargs.get('newPart', True):
  #   z, pi, theta, E, S = sampleNewPart(o, y, alpha, z, pi, theta, E, S, x, Q,
  #     mL, **kwargs)
  #   K = len(pi) - 1

  I = np.eye(o.dy+1)
  for t in range(T):
    for k in range(K):
      # sample theta_tk
      if t==0: thetaPrev, SPrev = ( o.H_theta[1], o.H_theta[2] )
      else: thetaPrev, SPrev = ( theta[t-1,k], S[k] )
      if t==T-1: thetaNext, SNext = ( None, None )
      else: thetaNext, SNext = ( theta[t+1,k], S[k] )

      y_tk = y[t][z[t]==k]
      R_theta_tk = m.Rt(theta[t,k])[0]
      rhs = I
      lhs = x[t] @ omega[k]
      # lhs = x[t] 

      # sample translation | rotation
      mu, Sigma = thetaTranslationDynamicsPosterior( o, R_theta_tk, thetaPrev,
        SPrev, nextU=thetaNext, A, B, nextS=SNext)
      mu, Sigma = translationObservationPosterior(o, y_tk, R_theta_tk,
        lhs, rhs, E[k], mu, Sigma)
      d_theta_tk = mvn.rvs(mu, Sigma)

      R_theta_tk = sampleRotation(o, y_tk, d_theta_tk, lhs, rhs, E[k],
        thetaPrev, SPrev, nextU=thetaNext, nextS=SNext, A=A, B=B)

      # old style
      # mu, Sigma = translationDynamicsPosterior(
      #   o, R_theta_tk, thetaPrev, SPrev, nextU=thetaNext, nextS=SNext)
      # mu, Sigma = translationObservationPosterior(o, y_tk, R_theta_tk,
      #   lhs, rhs, E[k], mu, Sigma)
      # d_theta_tk = mvn.rvs(mu, Sigma)

      # sample rotation | translation
      # R_theta_tk = sampleRotation(o, y_tk, d_theta_tk, lhs, rhs, E[k],
      #   thetaPrev, SPrev, nextU=thetaNext, nextS=SNext)
      
      # set new sample
      theta[t,k] = MakeRd(R_theta_tk, d_theta_tk)

    if not dontSampleX:
      # sample x_t
      if t==0: xPrev, QPrev = ( o.H_x[1], o.H_x[2] )
      else: xPrev, QPrev = ( x[t-1], Q )
      if t==T-1: xNext, QNext = ( None, None )
      else: xNext, QNext = ( x[t+1], Q )

      R_x_t = m.Rt(x[t])[0]
      lhs = I

      # sample translation | rotation
      mu, Sigma = translationDynamicsPosterior(
        o, R_x_t, xPrev, QPrev, nextU=xNext, nextS=QNext)
      for k in range(K):
        y_tk = y[t][z[t]==k]
        rhs = omega[k] @ theta[t,k]
        # rhs = theta[t,k]

        mu, Sigma = translationObservationPosterior(o, y_tk, R_x_t,
          lhs, rhs, E[k], mu, Sigma)
      d_x_t = mvn.rvs(mu, Sigma)
      
      # sample rotation | translation
      ## gather all y_tk
      yParts = tuple( y[t][z[t]==k] for k in range(K) )
      EParts = tuple( E[k] for k in range(K) )
      rhs = tuple( omega[k] @ theta[t,k] for k in range(K) )
      # rhs = tuple( theta[t,k] for k in range(K) )

      R_x_t = sampleRotation(o, yParts, d_x_t, lhs, rhs, EParts,
        xPrev, QPrev, nextU=xNext, nextS=QNext)

    # sample z_t
    z[t] = inferZ(o, y[t], pi, theta[t], E, x[t], mL[t])

  z, pi, theta, E, S = consolidatePartsAndResamplePi(o, z, pi, alpha, theta,
    E, S)
  K = len(pi) - 1

  # sample E, S, Q
  if not dontSampleX: Q = inferQ(o, x)

  if K > 0: S = np.array([ inferSk(o, theta[:,k]) for k in range(K) ])
  else: S = np.zeros((0, o.dxA, o.dxA))

  # sample omega
  R_omega = omega[:,:o.dy,:o.dy]
  omega = sampleOmega(o, y, z, x, theta, E, R_omega)
  # todo: sample omega rotation

  E = inferE(o, x, theta, omega, y, z)

  # compute log-likelihood
  ll = logJoint(o, y, z, x, theta, E, S, Q, alpha, pi, omega, mL)

  return z, pi, theta, E, S, x, Q, omega, ll

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
  K = len(pi) - 1

  Nk = []
  idx = []
  for k in range(K):
    s = 0
    for t in range(T): s += np.sum(z[t]==k)
    if s > 0:
      Nk.append(s)
      idx.append(k)

  # last entry of Nk is the number of unassigned observations
  s = 0
  for t in range(T): s += np.sum(z[t]==-1)
  if s > 0: Nk.append(s)
  else: Nk.append(alpha)
  idx.append(K)

  # resample pi
  piInputUnnormalized = pi[idx]
  logPiInputUnnormalized = np.log(piInputUnnormalized)
  piInput = np.exp(logPiInputUnnormalized - logsumexp(logPiInputUnnormalized))
  assert len(piInput) == len(Nk)
  assert np.isclose(np.sum(piInput), 1.0)
  assert np.all(piInput>0)
  pi = dirichlet.rvs(Nk)[0]

  # remove components that don't have associatios and relabel z
  missing = np.setdiff1d(range(K), idx[:-1])
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

  return z_, pi, theta, E, S

def try_switch(o, y, z, x, theta, E, S, Q, alpha, pi, mL, ks=None):
  m = getattr(lie, o.lie)
  T, K = ( len(y), len(pi)-1 )
  if K < 2: return False, z, x, theta, E, S, Q, alpha, pi, mL

  # sample two targets, set swap window up
  # if ks is None: ks = np.random.choice(range(K), size=2, replace=False)
  if ks is None: ks = np.random.choice(range(K), size=4, replace=False)

  t0 = np.zeros(len(ks))
  ts = list(range(1,T))
  theta0 = [ theta[0, k] for k in ks ]

  # build val, costxx, costxy functions
  zeroObs = np.zeros(o.dy)
  def val(t, k):
    z_tk = z[t] == k
    if np.sum(z_tk) == 0: y_tk, z_tk = (None, None)
    else: y_tk = y[t][z_tk]
    return theta[t,k], y_tk, z_tk
  def costxx(t1, thetaPrev, t2, theta, k):
    return logpdf_theta_tk(o, theta, thetaPrev, S[k] )

  def costxy(t, k, theta, y_tk_world):
    if y_tk_world is None: return 0.0
    T_part_world = m.inv( x[t].dot(theta) )
    y_tk_part = TransformPointsNonHomog(T_part_world, y_tk_world)
    return np.sum(mvn.logpdf(y_tk_part, zeroObs, E[k], allow_singular=True))
  
  perms, piHmm, Psi, psi = hmm.build(t0, theta0, ts, ks, val, costxx, costxy)
  states = hmm.ffbs(piHmm, Psi, psi)

  ## visualization part 1 edit
  # states[0] = 1
  # states[0] = 4
  ## end visualization part 1 edit

  # no swaps, auto accept
  # if np.all(states == 0): return True, z, x, theta, E, S, Q, alpha, pi, mL
  if np.all(np.asarray(states) == 0):
    return False, z, x, theta, E, S, Q, alpha, pi, mL

  theta_ = theta.copy()
  z_ = [ z[t].copy() for t in range(T) ]
  
  # decode states
  # k1, k2 = ks
  ks = np.asarray(ks)
  for s, t in zip(states, ts):
    if s == 0: continue

    # for kOld, kNew in zip(ks, ks[perms[s]]):
    perms_s = np.asarray(perms[s])
    for kOld, kNew in zip(ks, ks[perms_s]):
      # kOld -> kNew
      zkOld = z[t] == kOld
      
      z_[t][zkOld] = kNew
      theta_[t,kNew] = theta[t,kOld].copy()


    # s == 1 => switch corresponding theta[t,k] and z_tk
    # zt_k1 = z_[t] == k1
    # zt_k2 = z_[t] == k2
    #
    # theta_[t,k1], theta_[t,k2] = ( theta[t,k2], theta[t,k1] )
    #
    # z_[t][zt_k1] = k2
    # z_[t][zt_k2] = k1

  p_new = logJoint(o, y, z_, x, theta_, E, S, Q, alpha, pi, mL)
  p_old = logJoint(o, y, z, x, theta, E, S, Q, alpha, pi, mL)
  logRatio = p_new - p_old
  if np.any(np.asarray(states) != 0):
    print(f'Switch, p_new: {p_new:.2f}, p_old: {p_old:.2f}, logA: {logRatio:.2f}, states: {states}')
    # print(f'Switch ({k1:02}, {k2:02}), p_new: {p_new:.2f}, p_old: {p_old:.2f}, logA: {logRatio:.2f}, states: {states}')

  ## visualization part 2 edit
  # from . import drawSED 
  # import matplotlib.pyplot as plt, sys
  # idx = 0
  # t = ts[idx]
  # plt.figure()
  # drawSED.draw_t(o, y=y[t], z=z[t], x=x[t], theta=theta[t], E=E)
  # plt.title('Original')
  # plt.figure()
  # drawSED.draw_t(o, y=y[t], z=z_[t], x=x[t], theta=theta_[t], E=E)
  # perms_s = np.asarray(perms[states[idx]])
  # plt.title(f'Swap {ks} -> {ks[perms_s]}')
  #
  # # plt.title(f'Swap {k1}, {k2}')
  # # plt.title(f'Swap {ks}')
  # plt.show()
  # ip.embed()
  # sys.exit()
  ## end visualization part 2 edit

  if logRatio >= 0 or np.random.rand() < np.exp(logRatio):
    return True, z_, x, theta_, E, S, Q, alpha, pi, mL
  else:
    return False, z, x, theta, E, S, Q, alpha, pi, mL


def try_birth(o, y, z, x, theta, E, S, Q, alpha, pi, mL, pBirth, pDeath):
  # p(new)   g(old | new)   
  # ------ * ------------ * Jacobian
  # p(old)   g(new | old)
  m = getattr(lie, o.lie)
  T = len(y)
  K_new = len(pi)

  # log g(old | new)
  g_old_new = np.log(pDeath) + np.log(1 / K_new)

  # log g(new | old)
  ## sample all new random variables for proposal
  beta = Be.rvs(1,1)
  E_k = np.diag(np.diag(iw.rvs(*o.H_E[1:])))
  S_k = iw.rvs(*o.H_S[1:])
  theta1_rv_param = (m.algi(m.logm(o.H_theta[1])), o.H_theta[2])
  theta1_rv = mvn.rvs(*theta1_rv_param)
  theta1_ = m.expm(m.alg(theta1_rv))

  ## proposal logpdf 
  g_new_old = np.log(pBirth)
  g_new_old += Be.logpdf(beta, 1, 1)
  g_new_old += iw.logpdf(E_k, *o.H_E[1:])
  g_new_old += iw.logpdf(S_k, *o.H_S[1:])
  g_new_old += mvn.logpdf(theta1_rv, *theta1_rv_param)

  # log Jacobian
  logJ = np.log(pi[-1])

  # deterministic transformation
  pi_k = pi[-1] * beta
  pi_inf = pi[-1] * (1 - beta)
  pi_ = np.concatenate(( pi[:-1], np.array([pi_k, pi_inf]) ))
  E_ = np.concatenate((E, E_k[np.newaxis]), axis=0)
  S_ = np.concatenate((S, S_k[np.newaxis]), axis=0)
  theta_ = np.concatenate((
      theta,
      np.tile( theta1_[np.newaxis, np.newaxis], [T, 1, 1, 1] )
    ), axis=1)

  # resample labels z[t] as Gibbs-within-Metropolis step
  # Note: this doesn't figure into ratio because all involved terms cancel
  z_ = [ inferZ(o, y[t], pi_, theta_[t], E_, x[t], mL[t]) for t in range(T) ]

  # p(new)
  p_new = logJoint(o, y, z_, x, theta_, E_, S_, Q, alpha, pi_, mL)

  # p(old)
  p_old = logJoint(o, y, z, x, theta, E, S, Q, alpha, pi, mL)

  logRatio = p_new + g_old_new + logJ - p_old - g_new_old
  # print(f'BIRTH logRatio: {logRatio:.2f}')
  # print(f'  log p(new): {p_new:.2f}, log p(old): {p_old:.2f}')
  # print(f'  log g(old | new): {g_old_new:.2f}, log g(new | old): {g_new_old:.2f}')
  # print(f'  log Jacobian: {logJ:.2f}')
  # print(f'  nNewAssoc: {np.sum(z_ == len(pi_)-2)}')

  # accept / reject
  if logRatio >= 0 or np.random.rand() < np.exp(logRatio):
    return True, z_, x, theta_, E_, S_, Q, alpha, pi_, mL
  else:
    return False, z, x, theta, E, S, Q, alpha, pi, mL

def try_death(o, y, z, x, theta, E, S, Q, alpha, pi, mL, pDeath, pBirth):
  T = len(y)
  K_old = len(pi) - 1

  # sample random state
  C = np.random.randint(K_old) # component to kill

  # deterministic transformation
  newIdx = np.setdiff1d(np.arange(K_old+1), C)
  theta_, E_, S_ = ( theta[:,newIdx[:-1]], E[newIdx[:-1]], S[newIdx[:-1]] )
  pi_ = pi[newIdx]
  pi_[-1] += pi[C]  ## add deleted weight to unassigned

  # resample labels z[t] as Gibbs-within-Metropolis step
  # Note: this doesn't figure into ratio because all involved terms cancel
  z_ = [ inferZ(o, y[t], pi_, theta_[t], E_, x[t], mL[t]) for t in range(T) ]

  # construct ratio
  ## g(old | new) : reverse birth move
  beta_ = pi[C] / pi_[-1]
  g_old_new = np.log(pBirth)
  g_old_new += Be.logpdf(beta_, 1, 1)
  g_old_new += iw.logpdf(E[C], *o.H_E[1:])
  g_old_new += iw.logpdf(S[C], *o.H_S[1:])
  g_old_new += mvnL_logpdf(o, theta[0,C], *o.H_theta[1:])

  ## g(new | old) : death move
  g_new_old = np.log(pDeath) + np.log(1 / K_old)

  logJ = -np.log(pi[-1])

  # p(new)
  p_new = logJoint(o, y, z_, x, theta_, E_, S_, Q, alpha, pi_, mL)

  # p(old)
  p_old = logJoint(o, y, z, x, theta, E, S, Q, alpha, pi, mL)

  logRatio = p_new + g_old_new + logJ - p_old - g_new_old
  # print(f'DEATH logRatio: {logRatio:.2f}')
  # print(f'  log p(new): {p_new:.2f}, log p(old): {p_old:.2f}')
  # print(f'  log g(old | new): {g_old_new:.2f}, log g(new | old): {g_new_old:.2f}')
  # print(f'  log Jacobian: {logJ:.2f}')
  # print(f'  nNewAssoc: {np.sum(z_ == len(pi_)-2)}')

  # accept / reject
  if logRatio >= 0 or np.random.rand() < np.exp(logRatio):
    return True, z_, x, theta_, E_, S_, Q, alpha, pi_, mL
  else:
    return False, z, x, theta, E, S, Q, alpha, pi, mL

def sampleRJMCMC(o, y, alpha, z, pi, theta, E, S, x, Q, omega, mL, pBirth, pDeath, pSwitch, **kwargs):
  # RJMCMC birth/death 
  K = len(pi) - 1
  pGibbs = 1 - (pBirth + pDeath + pSwitch)

  if K == 0:
    rjmcmc_probs = np.array([pBirth, pGibbs])
    rjmcmc_probs /= np.sum(rjmcmc_probs)
    moves = ['birth', 'gibbs']
  elif K >= 2:
    rjmcmc_probs = np.array([pBirth, pDeath, pSwitch, pGibbs])
    rjmcmc_probs /= np.sum(rjmcmc_probs)
    moves = ['birth', 'death', 'switch', 'gibbs']
  else:
    rjmcmc_probs = np.array([pBirth, pDeath, pGibbs])
    rjmcmc_probs /= np.sum(rjmcmc_probs)
    moves = ['birth', 'death', 'gibbs']
  assert np.isclose(np.sum(rjmcmc_probs), 1.0)

  move = du.stats.catrnd(rjmcmc_probs[np.newaxis])[0]
  if moves[move] == 'birth':
    accept, z, x, theta, E, S, Q, alpha, pi, mL = try_birth(o, y,
      z, x, theta, E, S, Q, alpha, pi, mL, pBirth, pDeath)
  elif moves[move] == 'death': # death
    accept, z, x, theta, E, S, Q, alpha, pi, mL = try_death(o, y,
      z, x, theta, E, S, Q, alpha, pi, mL, pDeath, pBirth)
  elif moves[move] == 'switch':
    import random
    combs = list(itertools.combinations(range(K), 2))
    # combs = itertools.combinations(range(K), 3)
    random.shuffle(combs)
    for comb in combs:
      ks = comb
      accept, z, x, theta, E, S, Q, alpha, pi, mL = try_switch(o, y,
        z, x, theta, E, S, Q, alpha, pi, mL, ks=ks)
  elif moves[move] == 'gibbs':
    accept = True
    dontSampleX = kwargs.get('dontSampleX', False)

    z, pi, theta, E, S, x, Q, omega, ll = sampleStepFC(o, y, alpha, z, pi,
      theta, E, S, x, Q, omega, mL, newPart=False, dontSampleX=dontSampleX)
  
  return z, pi, theta, E, S, x, Q, omega, mL, moves[move], accept

def partPosteriors(o, y, x, z, theta, E):
  m = getattr(lie, o.lie)
  T, K = theta.shape[:2]
  colors = du.diffcolors(100, bgCols=[[1,1,1],[0,0,0]])

  # for each part, collect d_{theta_{tk}} for each time
  import matplotlib.pyplot as plt
  for k in range(K):
    d_theta = theta[:,k,:-1, -1]
    pts = du.stats.Gauss2DPoints( np.mean(d_theta,axis=0), np.cov(d_theta.T),
      deviations=2.0)

    # plt.plot(*pts, color=colors[k])
    # plt.scatter(d_theta[:,0], d_theta[:,1], s=5, color=colors[k])

    # draw part with mean transformation
    thetaMu = m.karcher(theta[:,k])
    T_obj_part = thetaMu
    zero = np.zeros(o.dy)
    yMu = TransformPointsNonHomog(T_obj_part, zero)
    R = T_obj_part[:-1,:-1]
    ySig = R.dot(E[k]).dot(R.T)
    plt.plot( *du.stats.Gauss2DPoints(yMu, ySig, deviations=1.25), c=colors[k],
      linestyle='--' )



  # invisibly plot observations to set scale
  t = 0
  yObj_t = TransformPointsNonHomog(m.inv(x[t]), y[t])
  plt.scatter(yObj_t[:,0], yObj_t[:,1], s=0)
  plt.gca().set_aspect('equal', 'box')
  plt.gca().set_xticks([])
  plt.gca().set_yticks([])
  plt.gca().invert_yaxis()
  plt.title('Part Offset Posterior')
  plt.savefig('posterior.png', dpi=300, bbox_iches='tight')
  # plt.show()

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

def mvnL_rv(o, mu, Sigma):
  m = getattr(lie, o.lie)
  zero = np.zeros(o.dxA)
  eps = m.expm(m.alg(mvn.rvs(zero, Sigma)))
  return mu @ eps
