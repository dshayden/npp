from . import SED, drawSED
from . import SED_tf
from .SED_tf import np2tf, ex, alg, generators_tf, obs_t_tf, Ei_tf, Si_tf, Qi_tf
import du, du.stats
import lie
from scipy.special import logsumexp
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import chi2
import scipy.optimize
import matplotlib.pyplot as plt
import functools, numdifftools
# import IPython as ip, sys
import tensorflow as tf
from tensorflow.linalg import matmul, matvec

def mahalanobis2_tf(y, SigmaI):
  """ Return squared mahalanobis distance of y[n] parameterized by SigmaI.
      
      dist2[n] = y[n].T @ SigmaI @ y[n]

  INPUT
    y (tensor, [N, dy]): points
    SigmaI (tensor, [dy, dy]): inverse covariance

  OUTPUT
    dist2 (tensor, [N,]): squared Mahalanobis distance
  """
  if tf.rank(y) == 1: y = tf.expand_dims(y, 0)
  return tf.einsum('nj,jk,nk->n', y, SigmaI, y)

def all2vec(o, x, omega, theta):
  # useful for seeding optimize_all
  m = getattr(lie, o.lie)
  mRot = getattr(lie, o.lieRot)

  T = x.shape[0]
  K = omega.shape[0]

  #      x         omega     theta     
  vDim = T*o.dxA + K*o.dxA + T*K*o.dxA
  v = np.zeros(vDim)

  x_v, parts_v = ( v[:T*o.dxA], v[T*o.dxA:] )
  omega_v, theta_v = ( parts_v[:K*o.dxA], parts_v[K*o.dxA:] )

  for k in range(K):
    w_k = m.algi(m.logm(m.inv(o.H_omega[1]) @ omega[k]))
    omega_v[ k*o.dxA : (k+1)*o.dxA ] = w_k

  xPrev = o.H_x[1]
  thetaPrev = np.tile(o.IdentityTransform, (K, 1, 1))
  for t in range(T):
    x_vt = m.algi(m.logm(m.inv(xPrev) @ x[t]))
    x_v[ t*o.dxA : (t+1)*o.dxA ] = x_vt

    theta_vt = theta_v[ t*K*o.dxA : (t+1)*K*o.dxA ]
    for k in range(K):
      R_thetaPrev_k, d_thetaPrev_k = m.Rt(thetaPrev[k])
      R_thetaCur_k, d_thetaCur_k = m.Rt(theta[t,k])

      s_tk = theta_vt[ k*o.dxA : (k+1)*o.dxA ]
      phi_tk = mRot.algi(mRot.logm(R_thetaPrev_k.T @ R_thetaCur_k))
      s_tk[:o.dy] = d_thetaCur_k
      s_tk[o.dy:] = phi_tk

  return v

def optimize_all(o, y, E, S, Q, **kwargs):
  m = getattr(lie, o.lie)
  T = len(y)
  K = E.shape[0]

  # cbNone = lambda v, cost, x, omega, theta: pass
  callbackInterval = kwargs.get('callbackInterval', None)
  callback = kwargs.get('callback', None)

  # observations
  y_ = [ np2tf(y[t]) for t in range(T) ]

  # generators
  G = generators_tf(o)
  GRot = G[o.dy:, :-1, :-1]

  # covariances
  Ei = [ np2tf(np.linalg.inv(E[k])) for k in range(K) ]
  Si = [ np2tf(np.linalg.inv(S[k])) for k in range(K) ]
  Qi = np2tf(np.linalg.inv(Q))
  Wi = np2tf(np.linalg.inv(o.H_omega[2]))

  # translation parameters
  A = np2tf(o.A)
  Bi = np2tf(o.Bi)

  # initial omega, x
  omega0 = np2tf(o.H_omega[1])
  x0 = np2tf(o.H_x[1])
  Sigma_x0i = np2tf(np.linalg.inv(o.H_x[2]))
  theta0 = np2tf(np.tile(o.IdentityTransform, (K,1,1)))

  # pre-allocate

  # parameters as group elements
  x = [ [] for t in range(T) ]
  omega = [ [] for k in range(K) ]
  theta = [ [] for t in range(T) ]
  for t in range(T): theta[t] = [ [] for k in range(K) ]
  
  # parameter dynamics costs
  xCost = [ [] for t in range(T) ]
  omegaCost = [ [] for k in range(K) ]
  thetaCost = [ [] for t in range(T) ]
  for t in range(T): thetaCost[t] = [ [] for k in range(K) ]
  thetaCost_t = [ [] for t in range(T) ]

  # observation costs
  negDists = [ [] for k in range(K) ]
  obsCost = [ [] for t in range(T) ]


  #      x         omega     theta     
  vDim = T*o.dxA + K*o.dxA + T*K*o.dxA
  v = tf.Variable(np2tf(kwargs.get('v', np.zeros(vDim))))
  def objective(v): 
    x_v, parts_v = ( v[:T*o.dxA], v[T*o.dxA:] )
    omega_v, theta_v = ( parts_v[:K*o.dxA], parts_v[K*o.dxA:] )

    # build omega and compute omega cost
    for k in range(K):
      omega_vk = omega_v[ k*o.dxA : (k+1)*o.dxA ]
      omega[k] = matmul(omega0, ex(alg(G, omega_vk)))
      omegaCost[k] = tf.reduce_sum(mahalanobis2_tf(
        tf.expand_dims(omega_vk,0), Wi))

    # iterate through each time
    xPrev, QPrev_i = (x0, Sigma_x0i)
    thetaPrev = theta0
    for t in range(T):
      # Make x_t
      x_vt = x_v[ t*o.dxA : (t+1)*o.dxA ]
      x[t] = matmul(xPrev, ex(alg(G, x_vt)))

      # x_t cost
      xCost[t] = tf.reduce_sum(mahalanobis2_tf(tf.expand_dims(x_vt,0), QPrev_i))

      theta_vt = theta_v[ t*K*o.dxA : (t+1)*K*o.dxA ]
      for k in range(K):
        # Make theta_tk
        s_tk = theta_vt[ k*o.dxA : (k+1)*o.dxA ]
        s_tkRot = s_tk[o.dy:]
        s_tkTrans = s_tk[:o.dy]
        R_thetaPrev_k, d_thetaPrev_k = SED_tf.Rt(np2tf(thetaPrev[k]))
        R_theta_tk = matmul(R_thetaPrev_k, ex(alg(GRot, s_tkRot)))
        theta[t][k] = SED_tf.MakeRd(o, R_theta_tk, s_tkTrans)

        # theta_tk cost
        m_tk = matvec(Bi, (s_tkTrans - matvec(A, d_thetaPrev_k)))
        val = tf.concat([m_tk, s_tkRot], axis=0)
        thetaCost[t][k] = tf.reduce_sum(mahalanobis2_tf(
          tf.expand_dims(val,0), Si[k]))

        lhs = SED_tf.inv(o,matmul(matmul(x[t], omega[k]), theta[t][k]))
        yPart = SED_tf.TransformPointsNonHomog(lhs, y_[t])
        negDists[k] = -mahalanobis2_tf(yPart, Ei[k])
      
      thetaCost_t[t] = tf.reduce_sum(thetaCost[t])
      negDistsStacked = tf.stack(negDists)
      smoothMins = -tf.math.reduce_logsumexp(negDistsStacked, axis=0)
      obsCost[t] = tf.reduce_sum(smoothMins)

      # Set prevs
      xPrev = x[t]
      thetaPrev = theta[t]
      QPrev_i = Qi

      ## end time t

    totalCost = tf.reduce_sum(xCost) + tf.reduce_sum(omegaCost) + \
      tf.reduce_sum(thetaCost_t) + tf.reduce_sum(obsCost)
    return totalCost

  # objective(v)

  def makeNumpy(s, v, cost, x, omega, theta):
    x_ = np.stack([ x[t].numpy() for t in range(T) ])
    theta_ = [ [] for t in range(T) ]
    for t in range(T):
      theta_[t] = np.stack([ theta[t][k].numpy() for k in range(K) ])
    theta__ = np.stack(theta_)
    omega_ = np.stack([ omega[k].numpy() for k in range(K) ])
    return s, v.numpy(), cost.numpy(), x_, omega_, theta__

  def grad(v):
    cost = tf.Variable(0.0)
    with tf.GradientTape() as tape: cost = objective(v)
    return cost, tape.gradient(cost, v)

  steps = kwargs.get('opt_steps', 10000)
  # steps = kwargs.get('opt_steps', 10)
  opt = tf.compat.v1.train.AdamOptimizer(learning_rate=0.1)
  prevCost = 1e6
  du.tic()
  for s in range(steps):
    cost, grads = grad(v)

    if callbackInterval is not None and s % callbackInterval == 0:
      callback(*makeNumpy(s, v, cost, x, omega, theta))

    opt.apply_gradients([(grads, v)])
    elapsedMin = du.toc() / 60.
    # print(f'{s:05}, cost: {cost.numpy():.2f}, elapsed (min): {elapsedMin:.2f}')
    if np.abs(cost.numpy() - prevCost) < 1e-6: break
    else: prevCost = cost.numpy()
  
  return makeNumpy(s, v, cost, x, omega, theta)


def multi_optimize_t(o, yt, omega, E, S, Q, xPrev, thetaPrev, xPrev2, thetaPrev2, **kwargs):
  m = getattr(lie, o.lie)
  t = len(yt)
  J = len(xPrev)
  K = E.shape[0]

  omega_ = [ np2tf(omega[k]) for k in range(K) ]
  omega_inv = [ np2tf(m.inv(omega[k])) for k in range(K) ]

  xPrev_ = np2tf(xPrev)
  thetaPrev_ = np2tf(thetaPrev)
  R_thetaPrev, d_thetaPrev = zip(*[SED_tf.Rt(np2tf(thetaPrev[k])) for k in range(K)])

  xPrev2_ = np2tf(xPrev2)
  thetaPrev2_ = np2tf(thetaPrev2)
  R_thetaPrev2, d_thetaPrev2 = zip(*[SED_tf.Rt(np2tf(thetaPrev2[k])) for k in range(K)])

  Ei = [ np2tf(np.linalg.inv(E[k])) for k in range(K) ]
  Si = [ np2tf(np.linalg.inv(S[k])) for k in range(K) ]
  Qi = np2tf(np.linalg.inv(Q))
  yt_ = np2tf(yt)

  Bi = np2tf(o.Bi)
  A = np2tf(o.A)

  G = generators_tf(o)
  GRot = G[o.dy:, :-1, :-1]

  # qs_t = tf.Variable(np2tf( kwargs.get('qs_t', np.zeros((K+1)*o.dxA)) ))
  qs_t = tf.Variable(np2tf( kwargs.get('qs_t', np.zeros(2*(K+1)*o.dxA)) ))

  def objective(qs_t):
    qs1 = qs_t[:(K+1)*o.dxA]
    q_t, s_t = ( qs1[:o.dxA], qs1[o.dxA:] )

    qs2 = qs_t[(K+1)*o.dxA:]
    q_t2, s_t2 = ( qs2[:o.dxA], qs2[o.dxA:] )

    # make x_t, theta_tk for each k
    x_t = matmul(xPrev_, ex(alg(G, q_t)))
    theta_t = [ [] for k in range(K) ]
    for k in range(K):
      s_tk = s_t[k*o.dxA : (k+1)*o.dxA]
      s_tkRot = s_tk[o.dy:]
      s_tkTrans = s_tk[:o.dy]
      R_theta_t = matmul(R_thetaPrev[k], ex(alg(GRot, s_tkRot)))
      theta_t[k] = SED_tf.MakeRd(o, R_theta_t, s_tkTrans)

    # same thing for x2
    x_t2 = matmul(xPrev2_, ex(alg(G, q_t2)))
    theta_t2 = [ [] for k in range(K) ]
    for k in range(K):
      s_tk2 = s_t2[k*o.dxA : (k+1)*o.dxA]
      s_tkRot2 = s_tk2[o.dy:]
      s_tkTrans2 = s_tk2[:o.dy]
      R_theta_t2 = matmul(R_thetaPrev2[k], ex(alg(GRot, s_tkRot2)))
      theta_t2[k] = SED_tf.MakeRd(o, R_theta_t2, s_tkTrans2)

    lhs = [ SED_tf.inv(o,matmul(matmul(x_t, omega_[k]), theta_t[k]))
      for k in range(K) ]
    yPart = [ SED_tf.TransformPointsNonHomog(lhs[k], yt_) for k in range(K) ]
    negDists = tf.stack([ -mahalanobis2_tf(yPart[k], Ei[k]) for k in range(K) ])

    lhs2 = [ SED_tf.inv(o,matmul(matmul(x_t2, omega_[k]), theta_t2[k]))
      for k in range(K) ]
    yPart2 = [ SED_tf.TransformPointsNonHomog(lhs2[k], yt_) for k in range(K) ]
    negDists2 = tf.stack([ -mahalanobis2_tf(yPart2[k], Ei[k]) for k in range(K) ])

    negDistsBoth = tf.concat([negDists, negDists2], axis=0)


    # smooth_mins = -tf.math.reduce_logsumexp(negDists, axis=0)
    smooth_mins = -tf.math.reduce_logsumexp(negDistsBoth, axis=0)
    cost = tf.reduce_sum(smooth_mins)

    # x dynamics
    cost_xDyn = tf.reduce_sum(mahalanobis2_tf(tf.expand_dims(q_t,0), Qi))

    # theta dynamics
    cost_thetaDyn = [ [] for k in range(K) ]
    for k in range(K):
      s_tk = s_t[k*o.dxA : (k+1)*o.dxA]
      s_tkRot = s_tk[o.dy:]
      s_tkTrans = s_tk[:o.dy]

      m_tk = matvec(Bi, (s_tkTrans - matvec(A, d_thetaPrev[k])))
      val = tf.concat([m_tk, s_tkRot], axis=0)
      cost_thetaDyn[k] = tf.reduce_sum(mahalanobis2_tf(
        tf.expand_dims(val,0), Si[k]))

    # x2 dynamics
    cost_xDyn2 = tf.reduce_sum(mahalanobis2_tf(tf.expand_dims(q_t2,0), Qi))

    # theta dynamics
    cost_thetaDyn2 = [ [] for k in range(K) ]
    for k in range(K):
      s_tk2 = s_t2[k*o.dxA : (k+1)*o.dxA]
      s_tkRot2 = s_tk2[o.dy:]
      s_tkTrans2 = s_tk2[:o.dy]

      m_tk2 = matvec(Bi, (s_tkTrans2 - matvec(A, d_thetaPrev2[k])))
      val2 = tf.concat([m_tk2, s_tkRot2], axis=0)
      cost_thetaDyn2[k] = tf.reduce_sum(mahalanobis2_tf(
        tf.expand_dims(val2,0), Si[k]))

    return cost + cost_xDyn + tf.reduce_sum(cost_thetaDyn) + cost_xDyn2 + tf.reduce_sum(cost_thetaDyn2)

  def grad(qs_t):
    cost = tf.Variable(0.0)
    with tf.GradientTape() as tape: cost = objective(qs_t)
    return cost, tape.gradient(cost, qs_t)

  steps = kwargs.get('opt_steps', 10000)
  opt = tf.compat.v1.train.AdamOptimizer(learning_rate=0.1)
  prevCost = 1e6
  for s in range(steps):
    cost, grads = grad(qs_t)
    opt.apply_gradients([(grads, qs_t)])
    # print(f'{s:05}, cost: {cost.numpy():.2f}, qs_t: {qs_t.numpy()}')
    if np.abs(cost.numpy() - prevCost) < 1e-6: break
    else: prevCost = cost.numpy()

  # omega_k = matmul(omega0, ex(alg(G, w)))
  # return omega_k.numpy()

  # q_t, s_t = ( qs_t[:o.dxA], qs_t[o.dxA:] )
  # x_t = matmul(xPrev_, ex(alg(G, q_t))).numpy()
  # theta_t = [ [] for k in range(K) ]
  # for k in range(K):
  #   s_tk = s_t[k*o.dxA : (k+1)*o.dxA]
  #   s_tkRot = s_tk[o.dy:]
  #   s_tkTrans = s_tk[:o.dy]
  #   R_theta_t = matmul(R_thetaPrev[k], ex(alg(GRot, s_tkRot)))
  #   theta_t[k] = SED_tf.MakeRd(o, R_theta_t, s_tkTrans).numpy()
  #
  # return x_t, np.stack(theta_t)

  qs1 = qs_t[:(K+1)*o.dxA]
  q_t, s_t = ( qs1[:o.dxA], qs1[o.dxA:] )

  qs2 = qs_t[(K+1)*o.dxA:]
  q_t2, s_t2 = ( qs2[:o.dxA], qs2[o.dxA:] )

  # make x_t, theta_tk for each k
  x_t = matmul(xPrev_, ex(alg(G, q_t)))
  theta_t = [ [] for k in range(K) ]
  for k in range(K):
    s_tk = s_t[k*o.dxA : (k+1)*o.dxA]
    s_tkRot = s_tk[o.dy:]
    s_tkTrans = s_tk[:o.dy]
    R_theta_t = matmul(R_thetaPrev[k], ex(alg(GRot, s_tkRot)))
    theta_t[k] = SED_tf.MakeRd(o, R_theta_t, s_tkTrans)

  # same thing for x2
  x_t2 = matmul(xPrev2_, ex(alg(G, q_t2)))
  theta_t2 = [ [] for k in range(K) ]
  for k in range(K):
    s_tk2 = s_t2[k*o.dxA : (k+1)*o.dxA]
    s_tkRot2 = s_tk2[o.dy:]
    s_tkTrans2 = s_tk2[:o.dy]
    R_theta_t2 = matmul(R_thetaPrev2[k], ex(alg(GRot, s_tkRot2)))
    theta_t2[k] = SED_tf.MakeRd(o, R_theta_t2, s_tkTrans2)

  return x_t, theta_t, x_t2, theta_t2



def optimize_t(o, yt, xPrev, omega, thetaPrev, E, S, Q, **kwargs):
  # jointly optimize x[t], theta[t,k]
  m = getattr(lie, o.lie)
  t = len(yt)
  K = E.shape[0]

  omega_ = [ np2tf(omega[k]) for k in range(K) ]
  omega_inv = [ np2tf(m.inv(omega[k])) for k in range(K) ]
  xPrev_ = np2tf(xPrev)
  thetaPrev_ = np2tf(thetaPrev)
  R_thetaPrev, d_thetaPrev = zip(*[SED_tf.Rt(np2tf(thetaPrev[k])) for k in range(K)])

  Ei = [ np2tf(np.linalg.inv(E[k])) for k in range(K) ]
  Si = [ np2tf(np.linalg.inv(S[k])) for k in range(K) ]
  Qi = np2tf(np.linalg.inv(Q))
  yt_ = np2tf(yt)

  Bi = np2tf(o.Bi)
  A = np2tf(o.A)

  G = generators_tf(o)
  GRot = G[o.dy:, :-1, :-1]

  qs_t = tf.Variable(np2tf( kwargs.get('qs_t', np.zeros((K+1)*o.dxA)) ))
  def objective(qs_t): 
    q_t, s_t = ( qs_t[:o.dxA], qs_t[o.dxA:] )

    # make x_t, theta_tk for each k
    x_t = matmul(xPrev_, ex(alg(G, q_t)))
    theta_t = [ [] for k in range(K) ]
    for k in range(K):
      s_tk = s_t[k*o.dxA : (k+1)*o.dxA]
      s_tkRot = s_tk[o.dy:]
      s_tkTrans = s_tk[:o.dy]
      R_theta_t = matmul(R_thetaPrev[k], ex(alg(GRot, s_tkRot)))
      theta_t[k] = SED_tf.MakeRd(o, R_theta_t, s_tkTrans)

    lhs = [ SED_tf.inv(o,matmul(matmul(x_t, omega_[k]), theta_t[k]))
      for k in range(K) ]
    yPart = [ SED_tf.TransformPointsNonHomog(lhs[k], yt_) for k in range(K) ]
    negDists = tf.stack([ -mahalanobis2_tf(yPart[k], Ei[k]) for k in range(K) ])
    smooth_mins = -tf.math.reduce_logsumexp(negDists, axis=0)
    cost = tf.reduce_sum(smooth_mins)

    # x dynamics
    cost_xDyn = tf.reduce_sum(mahalanobis2_tf(tf.expand_dims(q_t,0), Qi))

    # theta dynamics
    cost_thetaDyn = [ [] for k in range(K) ]
    for k in range(K):
      s_tk = s_t[k*o.dxA : (k+1)*o.dxA]
      s_tkRot = s_tk[o.dy:]
      s_tkTrans = s_tk[:o.dy]

      m_tk = matvec(Bi, (s_tkTrans - matvec(A, d_thetaPrev[k])))
      val = tf.concat([m_tk, s_tkRot], axis=0)
      cost_thetaDyn[k] = tf.reduce_sum(mahalanobis2_tf(
        tf.expand_dims(val,0), Si[k]))

    return cost + cost_xDyn + tf.reduce_sum(cost_thetaDyn)

  def grad(qs_t):
    cost = tf.Variable(0.0)
    with tf.GradientTape() as tape: cost = objective(qs_t)
    return cost, tape.gradient(cost, qs_t)

  steps = kwargs.get('opt_steps', 10000)
  learning_rate = kwargs.get('learning_rate', 0.1)

  # opt = tf.compat.v1.train.AdamOptimizer(learning_rate=0.1)

  opt = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
  prevCost = 1e6
  for s in range(steps):
    cost, grads = grad(qs_t)
    opt.apply_gradients([(grads, qs_t)])
    # print(f'{s:05}, cost: {cost.numpy():.2f}, qs_t: {qs_t.numpy()}')
    if np.abs(cost.numpy() - prevCost) < 1e-6: break
    else: prevCost = cost.numpy()

  # omega_k = matmul(omega0, ex(alg(G, w)))
  # return omega_k.numpy()

  q_t, s_t = ( qs_t[:o.dxA], qs_t[o.dxA:] )
  x_t = matmul(xPrev_, ex(alg(G, q_t))).numpy()
  theta_t = [ [] for k in range(K) ]
  for k in range(K):
    s_tk = s_t[k*o.dxA : (k+1)*o.dxA]
    s_tkRot = s_tk[o.dy:]
    s_tkTrans = s_tk[:o.dy]
    R_theta_t = matmul(R_thetaPrev[k], ex(alg(GRot, s_tkRot)))
    theta_t[k] = SED_tf.MakeRd(o, R_theta_t, s_tkTrans).numpy()
  
  return x_t, np.stack(theta_t), cost.numpy()


# optimize omega_k across all timesteps
def optimize_omega(o, yk, x, theta_k, E_k, **kwargs):
  m = getattr(lie, o.lie)
  T = len(yk)

  # y_tn = x_t omega_k theta_tk epsilon_tn
  # => (x_t omega_k theta_tk)^{-1} = epsilon_tn
  # => theta_tk^{-1} omega_k^{-1} x_t^{-1} y_tn = epsilon_tn
  # => theta_tk^{-1} omega_k^{-1} x_t^{-1} y_tn ~ N(0, E)
  yObj = [ np2tf(SED.TransformPointsNonHomog(m.inv(x[t]), yk[t]))
    for t in range(T) ]
  theta_ki = [ np2tf(m.inv(theta_k[t])) for t in range(T) ]

  omega0 = np2tf(o.H_omega[1])
  Wi = np2tf(np.linalg.inv(o.H_omega[2]))
  Ei = np2tf(np.linalg.inv(E_k))
  G = generators_tf(o)

  w = tf.Variable(np2tf( kwargs.get('w_t', np.zeros(o.dxA)) ))
  def objective(w): 
    omega = matmul(omega0, ex(alg(G, w)))
    omega_inv = SED_tf.inv(o, omega)

    lhs = [ matmul(theta_ki[t], omega_inv) for t in range(T) ]
    yPart = [ SED_tf.TransformPointsNonHomog(lhs[t], yObj[t])[:-1]
      for t in range(T) ]
    dists2 = [ tf.reduce_sum(mahalanobis2_tf(yPart[t], Ei))
      for t in range(T) ]

    costDyn = tf.reduce_sum(mahalanobis2_tf(tf.expand_dims(w,0), Wi))
    return tf.reduce_sum(dists2) + costDyn

  def grad(w):
    cost = tf.Variable(0.0)
    with tf.GradientTape() as tape: cost = objective(w)
    return cost, tape.gradient(cost, w)

  steps = kwargs.get('opt_steps', 10000)
  opt = tf.compat.v1.train.AdamOptimizer(learning_rate=0.1)
  prevCost = 1e6
  for s in range(steps):
    cost, grads = grad(w)
    opt.apply_gradients([(grads, w)])
    # print(f'{s:05}, cost: {cost.numpy():.2f}, w: {w.numpy()}')
    if np.abs(cost.numpy() - prevCost) < 1e-6: break
    else: prevCost = cost.numpy()

  omega_k = matmul(omega0, ex(alg(G, w)))
  return omega_k.numpy()

  # S_t = [ ex(alg(G, s_t[k*o.dxA : (k+1)*o.dxA])).numpy() for k in range(K) ]
  # return np.stack(S_t)


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
  opt = tf.compat.v1.train.AdamOptimizer(learning_rate=0.1)
  prevCost = 1e6
  for s in range(steps):
    cost, grads = grad(s_t)
    opt.apply_gradients([(grads, s_t)])
    # print(f'{s:05}, cost: {cost.numpy():.2f}, s: {s_t.numpy()}')
    if np.abs(cost.numpy() - prevCost) < 1e-6: break
    else: prevCost = cost.numpy()

  S_t = [ ex(alg(G, s_t[k*o.dxA : (k+1)*o.dxA])).numpy() for k in range(K) ]
  return np.stack(S_t)

def optimize_global2(o, y_t, xPrev, Q, theta_t, E, **kwargs):
  m = getattr(lie, o.lie)
  K = theta_t.shape[0]

  yObj = obs_t_tf(np2tf( SED.TransformPointsNonHomog(m.inv(xPrev), y_t) ))

  # theta = np2tf(theta_t)
  theta_inv = [ np2tf(m.inv(theta_t[k])) for k in range(K) ]

  Ei = Ei_tf(o, E)
  Qi = Qi_tf(o, Q)

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

    # dynamics cost   
    qQ = tf.linalg.matvec(Qi, q_t)
    qQq = tf.tensordot(qQ, q_t, axes=1)
    costDyn = qQq

    return cost + costDyn

  def grad(q_t):
    cost = tf.Variable(0.0)
    with tf.GradientTape() as tape: cost = objective(q_t)
    return cost, tape.gradient(cost, q_t)

  steps = kwargs.get('opt_steps', 500)
  opt = tf.compat.v1.train.AdamOptimizer(learning_rate=0.1)
  prevCost = 1e6
  for s in range(steps):
    cost, grads = grad(q_t)
    opt.apply_gradients([(grads, q_t)])
    # print(f'{s:05}, cost: {cost.numpy():.2f}, q: {q_t.numpy()}')
    if np.abs(cost.numpy() - prevCost) < 1e-4: break
    else: prevCost = cost.numpy()

  return ex(alg(G,q_t)).numpy()

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
  opt = tf.compat.v1.train.AdamOptimizer(learning_rate=0.1)
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
