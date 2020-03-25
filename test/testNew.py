import unittest
import numpy as np
from npp import SED, evalSED, drawSED as draw, icp, SED_tf
from npp.SED_tf import np2tf

from scipy.stats import multivariate_normal as mvn
from scipy.stats import invwishart as iw
from scipy.spatial.distance import mahalanobis
import scipy.optimize as so
import numdifftools as nd
import lie
import du, du.stats
import matplotlib.pyplot as plt
import IPython as ip, sys
import scipy.linalg as sla
import functools
np.set_printoptions(suppress=True, precision=4)

def testConditional():
  D1 = 2
  D2 = 3

  C = iw.rvs(10, np.eye(D1))
  Ci = np.linalg.inv(C)

  u = mvn.rvs(np.zeros(D1))

  mu = mvn.rvs(np.zeros(D1+D2))
  mu1 = mu[:D1]
  mu2 = mu[D1:]

  Sigma = iw.rvs(20, np.eye(D1 + D2))
  Sigma11 = Sigma[:D1,:D1]
  Sigma12 = Sigma[:D1,D1:]
  Sigma21 = Sigma[D1:,:D1]
  Sigma22 = Sigma[D1:,D1:]

  # x = mvn.rvs(mu, Sigma)
  # x1 = x[:D1]
  # x2 = x[D1:]
  
  # x1 | x2
  Ci_Sigma11_CiT = Ci @ Sigma11 @ Ci.T
  Ci_Sigma12 = Ci @ Sigma12
  Ci_mu1_u = Ci @ (mu1 - u)
  modifiedSigma = np.block([ [Ci_Sigma11_CiT, Ci_Sigma12], [Ci_Sigma12.T, Sigma22]])
  modifiedMu = np.concatenate((Ci_mu1_u, mu2))

  nSamples = 10000
  # draw from (x1, x2) joint, get x1 marginal
  x1x2 = mvn.rvs(modifiedMu, modifiedSigma, size=nSamples)
  x1 = x1x2[:,:D1]

  # draw from (Cx1 + u, x2) joint, get x1 marginal
  Cx1_u_x2 = mvn.rvs(mu, Sigma, size=nSamples)
  y = Cx1_u_x2[:,:D1]
  x1_ = np.stack([ Ci @ (y[n] - u) for n in range(nSamples) ])

  print(np.mean(x1,axis=0))
  print(np.cov(x1.T))

  print(np.mean(x1_,axis=0))
  print(np.cov(x1_.T))

  plt.scatter(*x1.T, s=1, color='b', alpha=0.5)
  plt.scatter(*x1_.T, s=1, color='g', alpha=0.5)
  plt.show()


  # test: is the joint distribution of (x1, x2) with above modifications same as
  # joint distribution of (C x1 + u)?
  #   y = C x1 + u
  #   x1 = Ci @ (y - u)


def show(o, y, x, omega, theta, E, z, t):
  o = SED.opts(lie='se2')
  m = getattr(lie, o.lie)
  mRot = getattr(lie, o.lieRot)
  K = E.shape[0]
  colors = du.diffcolors(K, bgCols=[[0,0,0],[1,1,1]])

  # plot x
  T_world_object = x[t]
  m.plot(T_world_object, np.tile([0,0,0], [2,1]), l=10.0)

  # plot x[t] @ omega[k] @ theta[t,k] in world coordinates
  for k in range(K):
    T_world_part = x[t] @ omega[k] @ theta[t,k]
    m.plot(T_world_part, np.tile(colors[k], [2,1]), l=10.0)

    yMu = SED.TransformPointsNonHomog(T_world_part, o.zeroObs)
    R = T_world_part[:-1,:-1]
    ySig = R.dot(E[k]).dot(R.T)
    plt.plot( *du.stats.Gauss2DPoints(yMu, ySig, deviations=2.0), color=colors[k])

  plt.scatter(*y[t].T, s=1, color='k')
  plt.gca().set_aspect('equal', 'box')

  plt.xlim(-50, 50)
  plt.ylim(-50, 50)
  plt.title(f'{t:04}')


def generateSyntheticK4(o, T, Nk=100):
  # general parameters
  o = SED.opts(lie='se2')
  m = getattr(lie, o.lie)
  mRot = getattr(lie, o.lieRot)
  K = 4

  # Q, x
  Q = np.diag((1.0, 2.0, .05))
  x = np.zeros((T,) + o.dxGm)
  x[0] = np.eye(o.dxA)
  q = mvn.rvs(np.zeros(o.dxA), Q, size=T-1)
  for t in range(1, T): x[t] = x[t-1] @ m.expm(m.alg(q[t-1]))

  # S
  S = iw.rvs(10, np.diag((50, 50, 0.5)), size=K)
  Strans = S[:,:o.dy,:o.dy]
  Srot = S[:,o.dy:,o.dy:]

  # E
  E = iw.rvs(10, np.diag((50, 5)), size=K)

  # omega
  w = np.array([[0, 12, 0], [0, -12, 0], [12, 0, 0], [-12, 0, 0]])
  omega = np.zeros((K,) + o.dxGm)
  for k in range(K): omega[k] = m.expm(m.alg(w[k]))

  theta = np.zeros((T, K) + o.dxGm)
  for k in range(K): theta[0,k] = np.eye(o.dy+1)

  # zeroObs = np.zeros(o.dy)
  zeroRot = np.zeros(o.dxA - o.dy)

  y = [ [] for t in range(T) ]   
  z = [ [] for t in range(T) ]   
  for t in range(T):
    for k in range(K):
      if t == 0: R_prev, d_prev = (o.IdentityRotation, o.zeroObs)
      else: R_prev, d_prev = m.Rt(theta[t-1,k])

      # random translation in lie group, serves the role of n[t]
      m_tk = mvn.rvs(o.zeroObs, Strans[k]) 

      # random rotation in lie algebra
      phi_tk = mvn.rvs(zeroRot, Srot[k]) 

      # rotation in lie group
      # R_tk = mRot.expm(mRot.alg(phi_tk))
      R_tk = R_prev @ mRot.expm(mRot.alg(phi_tk))

      # updated translation
      d_new = o.A @ d_prev + o.B @ m_tk

      # updated transformation
      S_tk = SED.MakeRd(R_tk, d_new)
      theta[t,k] = S_tk

      e_tk = mvn.rvs(o.zeroObs, E[k], size=Nk)
      y[t].append(SED.TransformPointsNonHomog(
        x[t] @ omega[k] @ theta[t,k], e_tk))
      z[t].append(k*np.ones(Nk, dtype=np.int))

    y[t] = np.vstack(y[t])
    z[t] = np.concatenate(z[t])

  return y, z, x, omega, theta, Q, S, E

def test_tf_mahal():
  dy = 2
  N = 50
  y = mvn.rvs( np.zeros(dy), np.eye(dy), size=N )
  mu = np.zeros(dy)

  SigmaI = np.linalg.inv(iw.rvs(10, np.eye(dy)))
  
  mahal_tf = icp.mahalanobis2_tf( np2tf(y), np2tf(SigmaI) ).numpy()
  
  mahal_np = np.zeros(N)
  for n in range(N):
    mahal_np[n] = mahalanobis( y[n], mu, SigmaI ) ** 2

  assert np.allclose(mahal_tf, mahal_np)

def test_optimize_all2vec():
  sample = 'omega/synthetic/se2_randomwalk3/002/init.gz'
  o, alpha, z, pi, theta, E, S, x, Q, omega, mL, ll, subsetIdx, datasetPath = \
    SED.loadSample(sample)
  v = icp.all2vec(o, x, omega, theta)


def test_optimize_all_synthetic():
  sample = 'omega/synthetic/se2_randomwalk3/002/init.gz'
  o, alpha, z, pi, theta, E, S, x, Q, omega, mL, ll, subsetIdx, datasetPath = \
    SED.loadSample(sample) 

  data = du.load(f'{datasetPath}/data')
  yAll = data['y']
  T = len(z)
  K = theta.shape[1]
  if subsetIdx is not None:
    y = [yt[subsetIdx[t]] for t, yt in enumerate(yAll)]

  basePath = 'optimize/synthetic/se2_randomwalk3/002'
  def callback(s, v, cost, x, omega, theta):
    z = [ SED.inferZ(o, y[t], pi, theta[t], E, x[t], omega, mL[t])
      for t in range(T) ]
    z, pi_, theta_, omega_, E_, S_ = SED.consolidatePartsAndResamplePi(o, z, pi,
      alpha, theta, omega, E, S)
    ll = SED.logJoint(o, y, z, x, theta_, E_, S_, Q, alpha, pi_, omega_, mL)

    extra = dict(s=s, v=v, cost=cost)
    filename = f'{basePath}/optimize-{s:05}'
    SED.saveSample(filename, o, alpha, z, pi_, theta_, E_, S_, x, Q, omega_, mL,
      ll, subsetIdx, datasetPath, extra=extra)

  s, v, cost, x, omega, theta = icp.optimize_all(o, y, E, S, Q,
    callback=callback, callbackInterval=1)

  ip.embed()

def test_optimize_all():
  sample = 'omega/se2_waving_hand/001/init.gz'
  o, alpha, z, pi, theta, E, S, x, Q, omega, mL, ll, subsetIdx, datasetPath = \
    SED.loadSample(sample) 

  data = du.load(f'{datasetPath}/data')
  yAll = data['y']
  T = len(z)
  K = theta.shape[1]
  if subsetIdx is not None:
    y = [yt[subsetIdx[t]] for t, yt in enumerate(yAll)]

  basePath = 'optimize/se2_waving_hand/001'
  def callback(s, v, cost, x, omega, theta):
    z = [ SED.inferZ(o, y[t], pi, theta[t], E, x[t], omega, mL[t])
      for t in range(T) ]
    z, pi_, theta_, omega_, E_, S_ = SED.consolidatePartsAndResamplePi(o, z, pi,
      alpha, theta, omega, E, S)
    ll = SED.logJoint(o, y, z, x, theta_, E_, S_, Q, alpha, pi_, omega_, mL)

    extra = dict(s=s, v=v, cost=cost)
    filename = f'{basePath}/optimize-{s:05}'
    SED.saveSample(filename, o, alpha, z, pi_, theta_, E_, S_, x, Q, omega_, mL,
      ll, subsetIdx, datasetPath, extra=extra)

  s, v, cost, x, omega, theta = icp.optimize_all(o, y, E, S, Q,
    callback=callback, callbackInterval=1)

  ip.embed()

  # x_t, theta_t = icp.optimize_t(o, y[t], x[t-1], omega, theta[t-1], E, S, Q)
  
def test_optimize_omega():
  sample = 'omega/se2_waving_hand/001/init.gz'
  o, alpha, z, pi, theta, E, S, x, Q, omega, mL, ll, subsetIdx, datasetPath = \
    SED.loadSample(sample) 

  data = du.load(f'{datasetPath}/data')
  yAll = data['y']
  T = len(z)
  K = theta.shape[1]
  if subsetIdx is not None:
    y = [yt[subsetIdx[t]] for t, yt in enumerate(yAll)]

  # omegaNew = np.zeros_like(omega)
  # # for k in range(K):
  # for k in range(0,1):
  #   yk = [ y[t][z[t]==k] for t in range(T) ]
  #   theta_k = theta[:,k]
  #   omegaNew[k] = icp.optimize_omega(o, yk, x, theta_k, E[k])

  t = 2
  x_t, theta_t = icp.optimize_t(o, y[t], x[t-1], omega, theta[t-1], E, S, Q)

  omegaTheta_t = theta_t.copy()
  for k in range(K): omegaTheta_t[k] = omega[k] @ omegaTheta_t[k]
  draw.draw_t(o, y=y[t], x=x_t, theta=omegaTheta_t, E=E)

  ip.embed()




def test_tf_Rt():
  o = SED.opts(lie='se2')
  m = getattr(lie, o.lie)
  x = m.rvs()
  R_lie, d_lie = m.Rt(x)
  R_tf, d_tf = SED_tf.Rt(np2tf(x))
  assert np.allclose(R_tf.numpy(), R_lie)
  assert np.allclose(d_tf.numpy().squeeze(), d_lie)

  o = SED.opts(lie='se3')
  m = getattr(lie, o.lie)
  x = m.rvs()
  R_lie, d_lie = m.Rt(x)
  R_tf, d_tf = SED_tf.Rt(np2tf(x))
  assert np.allclose(R_tf.numpy(), R_lie)
  assert np.allclose(d_tf.numpy().squeeze(), d_lie)

  xi_tf = SED_tf.inv(o,np2tf(x)).numpy()
  xi_lie = m.inv(x)
  assert np.allclose(xi_tf, xi_lie)

  y = mvn.rvs(np.zeros(o.dy), size=100)
  xy_tf = SED_tf.TransformPointsNonHomog(np2tf(x), np2tf(y)).numpy()
  xy_lie = SED.TransformPointsNonHomog(x, y)
  assert np.abs(np.sum(xy_tf - xy_lie)) < 1e-4




def testTranslationWithObs():
  o = SED.opts(lie='se2')
  m = getattr(lie, o.lie)
  mRot = getattr(lie, o.lieRot)

  T = 10
  y, z, x, omega, theta, Q, S, E = generateSyntheticK4(o, T)

  t, k = (5, 2)

  # pi = np.array([.25, .25, .25, .24, .01])
  # mL_t = -14 * np.ones(y[t].shape[0])
  # z_t = SED.inferZ(o, y[t], pi, theta[t], E, x[t], omega, mL_t)
  # ip.embed()

  R_cur, d_cur = m.Rt(theta[t,k])
  mu, Sigma = SED.thetaTranslationDynamicsPosterior(o, R_cur, theta[t-1,k],
    S[k], nextU=theta[t+1,k])
  y_tk = y[t][z[t]==k]
  lhs = x[t] @ omega[k]
  rhs = o.IdentityTransform
  mu, Sigma = SED.translationObservationPosterior(o, y_tk, R_cur,
    lhs, rhs, E[k], mu, Sigma)
  print(d_cur)
  print(mu)


def testSimpleInit():
  o = SED.opts(lie='se2')
  m = getattr(lie, o.lie)
  mRot = getattr(lie, o.lieRot)

  T = 5
  y, z_, x_, omega_, theta_, Q_, S_, E_ = generateSyntheticK4(o, T)

  mL = [ -14.0 * np.ones(y[t].shape[0]) for t in range(T) ]
  SED.initPriorsDataDependent(o, y)

  x = SED.initXDataMeans(o, y)
  # x = x_

  Q = SED.inferQ(o, x)
  tInit = np.random.choice(range(T))
  # tInit = 0

  alpha = 0.1
  theta, omega, E, S, z, pi = SED.initPartsAndAssoc(o, y, x, alpha, mL,
    tInit=tInit)
  K = len(pi) - 1

  # show_ = functools.partial(show, o, y, x_, omega, theta, E, z)
  # show_ = functools.partial(show, o, y, x_, omega_, theta_, E_, z_)
  # du.ViewPlots(range(T), show_)
  # plt.show()
  # show_ = functools.partial(show, o, y, x_, omega, theta, E, z)
  
  omegaTheta = np.zeros_like(theta)
  for t in range(T):
    for k in range(K):
      omegaTheta[t,k] = omega[k] @ theta[t,k]
  draw.draw(o, y=y, z=z, x=x, theta=omegaTheta, E=E)
  plt.show()


def testNewData():
  o = SED.opts(lie='se2')
  m = getattr(lie, o.lie)
  mRot = getattr(lie, o.lieRot)

  T = 10
  y, z, x, omega, theta, Q, S, E = generateSyntheticK4(o, T)
  K = S.shape[0]
  # ip.embed()

def testControlledWalkCov():
  o = SED.opts(lie='se2')
  m = getattr(lie, o.lie)
  mRot = getattr(lie, o.lieRot)
  T = 10000
  y, z, x, omega, theta, Q, S, E = generateSyntheticK4(o, T, Nk=2)
  Strans = S[:,:o.dy,:o.dy]
  Srot = S[:,o.dy:,o.dy:]
  K = S.shape[0]

  plt.figure(figsize=(12,8))
  colors = du.diffcolors(K, bgCols=[[0,0,0],[1,1,1]])
  for k in range(K):
    plt.subplot(2,2,k+1)
    d_k = theta[:,k,:-1,-1]
    omega_trans = omega[k,:-1,-1]
    plt.scatter(*d_k.T, color=colors[k], s=1)
    plt.plot(*du.stats.Gauss2DPoints(o.zeroObs, Strans[k]), color=colors[k])
    plt.title(f'Part {k+1}')
    plt.gca().set_aspect('equal', 'box')
    # plt.plot(*du.stats.Gauss2DPoints(omega_trans, Strans), color=colors[k])

  path = 'reparam/controlled'
  plt.savefig(f'{path}/cov.png', dpi=300, bbox_inches='tight')
  plt.show()

def testVisualizeGenerative():
  o = SED.opts(lie='se2')
  m = getattr(lie, o.lie)
  mRot = getattr(lie, o.lieRot)
  T = 500
  y, z, x, omega, theta, Q, S, E = generateSyntheticK4(o, T)
  K = S.shape[0]

  show_ = functools.partial(show, o, y, x, omega, theta, E, z)
  du.ViewPlots(range(T), show_)
  plt.show()


def testTranslationBasic():
  # general parameters
  o = SED.opts(lie='se2')
  m = getattr(lie, o.lie)
  mRot = getattr(lie, o.lieRot)

  T = 10
  y, z, x, omega, theta, Q, S, E = generateSyntheticK4(o, T)
  K = S.shape[0]

  Strans = S[:,:o.dy,:o.dy]
  Srot = S[:,o.dy:,o.dy:]

  # t, k = (5, 2)
  for t in range(1,T-1):
    for k in range(K):
      R_cur, d_cur = m.Rt(theta[t,k])
      zeroAlgebra = np.zeros(o.dxA)

      # test inference here
      R_prev, d_prev = m.Rt(theta[t-1,k])
      R_next, d_next = m.Rt(theta[t+1,k])
      def thetat_fwdBack(v):
        ll = 0.0

        # v is a translation; we'll already have a rotation so make the full
        V = SED.MakeRd(R_cur, v)

        # past
        phi = mRot.algi(mRot.logm(R_prev.T @ R_cur))
        phi = np.atleast_1d(phi)

        m = o.Bi @ (v - o.A @ d_prev)
        val = np.concatenate((m, phi))
        ll = mvn.logpdf(val, zeroAlgebra, S[k])

        # future
        phi_ = mRot.algi(mRot.logm(R_cur.T @ R_next))
        phi_ = np.atleast_1d(phi_)

        m_ = o.Bi @ (d_next - o.A @ v)
        val_ = np.concatenate((m_, phi_))
        ll += mvn.logpdf(val_, zeroAlgebra, S[k])
        
        return -ll
      
      mu, Sigma = SED.thetaTranslationDynamicsPosterior(o, R_cur, theta[t-1,k],
        S[k], nextU=theta[t+1,k])
      mu_noFuture, Sigma_noFuture = SED.thetaTranslationDynamicsPosterior(o, R_cur, theta[t-1,k],
        S[k])
      # print(f'd_cur:\n {d_cur}\n mu:\n {mu}\n Sigma:\n {Sigma}\n S:\n{S[k]}')

      g = nd.Gradient(thetat_fwdBack)
      map_est = so.minimize(thetat_fwdBack, np.zeros(o.dy), method='BFGS',
        jac=g)

      norm = np.linalg.norm(map_est.x - mu)
      print(f'norm: {norm:.6f}')
      assert norm < 1e-2, \
        f'SED.test5 bad, norm {norm:.6f}'

  plt.scatter(*d_cur, color='g')
  plt.plot(*du.stats.Gauss2DPoints(mu, Sigma), color='b')
  plt.plot(*du.stats.Gauss2DPoints(mu_noFuture, Sigma_noFuture), color='r')
  plt.scatter(*mu, color='b')
  plt.scatter(*mu_noFuture, color='r')
  plt.show()

  sys.exit()

  
  colors = du.diffcolors(K, bgCols=[[0,0,0],[1,1,1]])
  def show(t):
    # plot x
    T_world_object = x[t]
    m.plot(T_world_object, np.tile([0,0,0], [2,1]), l=10.0)

    # plot theta[t,k]
    for k in range(K):
      T_world_part = x[t] @ omega[k] @ theta[t,k]
      m.plot(T_world_part, np.tile(colors[k], [2,1]), l=10.0)

    plt.xlim(-100, 100)
    plt.ylim(-100, 100)

  path = 'reparam/controlled'
  # for t in range(T):
  #   show(t)
  #   plt.savefig(f'{path}/img-{t:05}.png', dpi=300, bbox_inches='tight')
  #   plt.close()

  # du.ViewPlots(range(T), show)
  # plt.show()

def testRotationBasic():
  # general parameters
  o = SED.opts(lie='se2')
  m = getattr(lie, o.lie)
  mRot = getattr(lie, o.lieRot)

  # T, K = ( 10000, 4 )
  T, K = ( 10, 4 )

  T = 10
  y, z, x, omega, theta, Q, S, E = generateSyntheticK4(o, T)
  K = S.shape[0]

  Strans = S[:,:o.dy,:o.dy]
  Srot = S[:,o.dy:,o.dy:]
  
  thetaR = theta.copy()
  for t in range(1,T-1):
    for k in range(K):
      R_cur, d_cur = m.Rt(theta[t,k])
      zeroAlgebra = np.zeros(o.dxA)

      # test inference here
      R_prev, d_prev = m.Rt(theta[t-1,k])
      R_next, d_next = m.Rt(theta[t+1,k])

      lhs = x[t] @ omega[k]
      rhs = np.eye(o.dy+1)
      y_tk = y[t][z[t]==k]
      R_U = SED.sampleRotation(o, y_tk, d_cur, lhs, rhs, E[k], theta[t-1,k],
        S[k], nextU=theta[t+1,k], controlled=True)
      thetaR[t,k,:o.dy,:o.dy] = R_U

  colorsA = du.diffcolors(K, bgCols=[[0,0,0],[1,1,1]], alpha=0.5)
  colors = du.diffcolors(K, bgCols=[[0,0,0],[1,1,1]])
  def show(t):
    # plot x
    T_world_object = x[t]
    m.plot(T_world_object, np.tile([0,0,0], [2,1]), l=10.0)

    # plot theta[t,k]
    for k in range(K):
      T_world_part = x[t] @ omega[k] @ theta[t,k]
      m.plot(T_world_part, np.tile(colorsA[k], [2,1]), l=10.0)
      plt.scatter(*y[t][k].T, color=colors[k], s=1, alpha=0.25)

    # plot thetaR[t,k]
    for k in range(K):
      T_world_part = x[t] @ omega[k] @ thetaR[t,k]
      m.plot(T_world_part, np.tile(colors[k], [2,1]), l=10.0)

    plt.xlim(-100, 100)
    plt.ylim(-100, 100)

  du.ViewPlots(range(1,T-1), show)
  plt.show()
