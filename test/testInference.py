import unittest
import numpy as np
from npp import SED, evalSED, drawSED as draw, icp
from scipy.stats import multivariate_normal as mvn
from scipy.stats import invwishart as iw
import scipy.optimize as so
import numdifftools as nd
import lie
import du, du.stats
import matplotlib.pyplot as plt
import IPython as ip, sys
import scipy.linalg as sla
np.set_printoptions(suppress=True, precision=4)

class test_counting_sorting(unittest.TestCase):
  def setUp(s):
    None

  def test_inferPi(s):
    o = SED.opts()
    alpha =  0.1
    Nk = np.array([1,2,3,4], dtype=np.int)
    pi = SED.inferPi(o, Nk, alpha)
    assert len(pi) == len(Nk)
    assert np.isclose(np.sum(pi), 1.0)
    logPi = np.log(pi) # calculate to ensure no errors

    alpha =  1e-6
    Nk = np.array([1,2,3,0], dtype=np.int)
    pi = SED.inferPi(o, Nk, alpha)
    assert len(pi) == len(Nk)
    assert np.isclose(np.sum(pi), 1.0)
    logPi = np.log(pi) # calculate to ensure no errors

    alpha =  1e-6
    Nk = np.array([10000,20000,30000,0], dtype=np.int)
    pi = SED.inferPi(o, Nk, alpha)
    assert len(pi) == len(Nk)
    assert np.isclose(np.sum(pi), 1.0)
    logPi = np.log(pi) # calculate to ensure no errors

    alpha =  1e-6
    Nk = np.array([1], dtype=np.int)
    pi = SED.inferPi(o, Nk, alpha)
    assert len(pi) == len(Nk)
    assert np.isclose(np.sum(pi), 1.0)
    logPi = np.log(pi) # calculate to ensure no errors

  def test_consolidateExtantParts_relabel(s):
    o = SED.opts()
    T, K = (5, 3)
    Nk = np.array([2, 0, 2, 5])
    z = [ np.array([-1, 0, -1, 0, 2, -1, 2, -1, -1], dtype=np.int)
      for t in range(T) ]
    pi = 0.25*np.ones(K+1)

    theta = np.random.rand(T, K, *o.dxGm)
    E = np.random.rand(K, o.dy, o.dy)
    S = np.random.rand(K, o.dxA, o.dxA)

    z_, theta_, E_, S_ = SED.consolidateExtantParts(o, z, pi, theta, E, S)
    for t in range(T):
      assert np.sum(z_[t]==0) == Nk[0]
      assert np.sum(z_[t]==1) == Nk[2]
      assert np.sum(z_[t]==2) == 0
      assert np.sum(z_[t]==-1) == Nk[-1]

      assert np.allclose(theta[t][0], theta_[t][0])
      assert np.allclose(theta[t][2], theta_[t][1])
    
    assert np.allclose(E[0], E_[0])
    assert np.allclose(E[2], E_[1])

    assert np.allclose(S[0], S_[0])
    assert np.allclose(S[2], S_[1])

  def test_consolidateExtantParts_no_relabel(s):
    o = SED.opts()
    T, K = (5, 3)
    Nk = np.array([2, 1, 2, 4])
    z = [ np.array([-1, 0, 1, 0, 2, -1, 2, -1, -1], dtype=np.int)
      for t in range(T) ]
    pi = 0.25*np.ones(K+1)

    theta = np.random.rand(T, K, *o.dxGm)
    E = np.random.rand(K, o.dy, o.dy)
    S = np.random.rand(K, o.dxA, o.dxA)

    z_, theta_, E_, S_ = SED.consolidateExtantParts(o, z, pi, theta, E, S)
    for t in range(T):
      for k in range(K):
        assert np.sum(z_[t]==k) == Nk[k]
        assert np.allclose(theta[t][k], theta_[t][k])

      assert np.sum(z_[t]==-1) == Nk[-1]

    for k in range(K):
      assert np.allclose(E[k], E_[k])
      assert np.allclose(S[k], S_[k])

  def test_getComponentCounts(s):
    o = SED.opts()
    z = np.array([0, 1, 0, 2, 3, -1, -1], dtype=np.int)
    pi = 0.2 * np.ones(len(np.unique(z)))
    Nk = SED.getComponentCounts(o, z, pi)
    assert Nk[0] == 2
    assert Nk[1] == 1
    assert Nk[2] == 1
    assert Nk[3] == 1
    assert Nk[4] == 2

class test_se2_randomwalk10(unittest.TestCase):
  def setUp(s):
    data = du.load('data/synthetic/se2_randomwalk10/data')
    s.o = SED.opts(lie='se2')
    s.T = len(data['x'])
    s.y = data['y']

    # Ground-truth
    s.KTrue = data['theta'].shape[1]
    s.xTrue = du.asShape(data['x'], (s.T,) + s.o.dxGm)
    s.thetaTrue = data['theta']
    s.ETrue = data['E']
    s.STrue = data['S']
    s.QTrue = data['Q']
    s.zTrue = data['z']
    s.piTrue = data['pi']
    s.data = data

  def test_logMarginalPartLikelihoodMonteCarlo(s):
    o = s.o
    o.H_S = s.data['H_S']
    o.H_E = s.data['H_E']
    o.H_theta = s.data['H_theta']
    T, K = (len(s.y), 100)

    try:
      theta, E, S = SED.sampleKPartsFromPrior(s.o, T, K)
      mL = SED.logMarginalPartLikelihoodMonteCarlo(o, s.y, s.xTrue, theta, E, S)
    except:
      assert False, 'problem with logMarginalPartLikelihoodMonteCarlo'
    
  def test_sampleKPartsFromPrior(s):
    o = s.o
    o.H_S = s.data['H_S']
    o.H_E = s.data['H_E']
    o.H_theta = s.data['H_theta']
    T, K = (5, 10)

    theta, E, S = SED.sampleKPartsFromPrior(s.o, T, K)
    assert theta.shape[0] == T
    assert theta.shape[1] == K
    assert E.shape[0] == K
    assert S.shape[0] == K
    for t in range(T):
      for k in range(K):
        assert np.isclose(np.linalg.det(theta[t,k,:-1,:-1]), 1.0)
        for i in range(o.dy):
          assert theta[t,k,-1,i] == 0
        assert theta[t,k,-1,-1] == 1.0

    T, K = (5, 1)
    theta, E, S = SED.sampleKPartsFromPrior(s.o, T, K)
    assert theta.shape[0] == T
    assert theta.shape[1] == K
    assert E.shape[0] == K
    assert S.shape[0] == K
    for t in range(T):
      for k in range(K):
        assert np.isclose(np.linalg.det(theta[t,k,:-1,:-1]), 1.0)
        for i in range(o.dy):
          assert theta[t,k,-1,i] == 0
        assert theta[t,k,-1,-1] == 1.0

    T, K = (5, 0)
    theta, E, S = SED.sampleKPartsFromPrior(s.o, T, K)
    assert theta.shape[0] == T
    assert theta.shape[1] == K
    assert E.shape[0] == K
    assert S.shape[0] == K

  def test_logJoint(s):
    alpha = 0.1
    o = s.o
    m = getattr(lie, o.lie)

    o.H_Q = s.data['H_Q']
    o.H_E = s.data['H_E']
    o.H_S = s.data['H_S']
    o.H_x = s.data['H_x']
    o.H_theta = s.data['H_theta']

    pi = np.concatenate((s.piTrue, np.array([alpha,])))

    # todo: handle mL
    ll = SED.logJoint(o, s.y, s.zTrue, s.xTrue, s.thetaTrue, s.ETrue, s.STrue,
      s.QTrue, alpha, pi, mL=None)
    assert not np.isnan(ll)
    assert not np.isinf(ll)

  def testInferQ(s):
    m = getattr(lie, s.o.lie)
    o = s.o

    # Test with true prior
    o.H_Q = s.data['H_Q']
    Q_ = SED.inferQ(s.o, s.xTrue)
    norm = np.linalg.norm(Q_ - s.QTrue)
    assert norm <= 2.0, f'bad inferQ, norm: {norm:.6f}'

    # Test with default data-dependent prior
    SED.initPriorsDataDependent(o, s.y)
    Q_ = SED.inferQ(s.o, s.xTrue)
    norm = np.linalg.norm(Q_ - s.QTrue)
    assert norm <= 15.0, f'bad inferQ, norm: {norm:.6f}'

  def testInferSk(s):
    m = getattr(lie, s.o.lie)
    o = s.o

    # Test with true prior
    o.H_S = s.data['H_S']
    for k in range(s.KTrue):
      Sk_ = SED.inferSk(s.o, s.thetaTrue[:,k])
      norm = np.linalg.norm(Sk_ - s.STrue[k])
      assert norm <= 0.5, f'bad inferSk, norm: {norm:.6f}'

    # Test with default data-dependent prior
    SED.initPriorsDataDependent(o, s.y)
    for k in range(s.KTrue):
      Sk_ = SED.inferSk(s.o, s.thetaTrue[:,k])
      norm = np.linalg.norm(Sk_ - s.STrue[k])
      assert norm <= 15.0, f'bad inferSk, norm: {norm:.6f}'


  def testInferE(s):
    m = getattr(lie, s.o.lie)
    o = s.o

    # Test with true prior
    o.H_E = s.data['H_E']
    E_ = SED.inferE(s.o, s.xTrue, s.thetaTrue, s.y, s.zTrue)
    for k in range(s.KTrue):
      norm = np.linalg.norm(E_[k] - s.ETrue[k])
      assert norm <= 1.0, f'bad inferE, norm: {norm:.6f}'

    # Test with default data-dependent prior
    SED.initPriorsDataDependent(o, s.y)
    E__ = SED.inferE(s.o, s.xTrue, s.thetaTrue, s.y, s.zTrue)
    for k in range(s.KTrue):
      norm = np.linalg.norm(E__[k] - s.ETrue[k])
      assert norm <= 4.0, f'bad inferE, norm: {norm:.6f}'

  def testTranslationX_fwd_noObs(s):
    m = getattr(lie, s.o.lie)
    for t in range(1,s.T-1):
      y_t = np.zeros((0, s.y[t].shape[1]))
      z_t = np.zeros(0, dtype=np.int)

      # infer d_x_t
      # copy true R_x_t, but have incorrect d_x_t
      x_t = s.xTrue[t-1].copy()
      x_t[:-1,:-1] = s.xTrue[t,:-1,:-1]
      mu, Sigma = SED.sampleTranslationX(s.o, y_t, z_t, x_t,
        s.thetaTrue[t], s.ETrue, s.QTrue, s.xTrue[t-1], returnMuSigma=True)

      # optimize same objective and compare
      Qi = np.linalg.inv(s.QTrue)
      def xt_fwd(v):
        # Set x_t candidate
        V = x_t.copy()
        V[:-1,-1] = v

        # log map and evaluate
        V_algebra = m.algi(m.logm(m.inv(s.xTrue[t-1]).dot(V)))
        nll = 0.5 * V_algebra.dot(Qi).dot(V_algebra)
        return nll
      
      g = nd.Gradient(xt_fwd)
      map_est = so.minimize(xt_fwd, np.zeros(s.o.dy), method='BFGS', jac=g)

      norm = np.linalg.norm(map_est.x - mu)
      # print(f'norm: {norm:.6f}')
      assert norm < 1e-2, f'SED.sampleTranslationX bad, norm {norm:.6f}'

  def testTranslationX_fwdBack_noObs(s):
    m = getattr(lie, s.o.lie)
    for t in range(1,s.T-1):
      y_t = np.zeros((0, s.y[t].shape[1]))
      z_t = np.zeros(0, dtype=np.int)

      # infer d_x_t
      # copy true R_x_t, but have incorrect d_x_t
      x_t = s.xTrue[t-1].copy()
      x_t[:-1,:-1] = s.xTrue[t,:-1,:-1]
      mu, Sigma = SED.sampleTranslationX(s.o, y_t, z_t, x_t,
        s.thetaTrue[t], s.ETrue, s.QTrue, s.xTrue[t-1], x_tplus1=s.xTrue[t+1],
        returnMuSigma=True)

      # optimize same objective and compare
      Qi = np.linalg.inv(s.QTrue)
      def xt_fwdBack(v):
        # Set x_t candidate
        V1 = x_t.copy()
        V1[:-1,-1] = v

        # log map and evaluate
        V1_algebra = m.algi(m.logm(m.inv(s.xTrue[t-1]).dot(V1)))
        nll = 0.5 * V1_algebra.dot(Qi).dot(V1_algebra)

        V2_algebra = m.algi(m.logm(m.inv(V1).dot(s.xTrue[t+1])))
        nll += 0.5 * V2_algebra.dot(Qi).dot(V2_algebra)

        return nll
      
      g = nd.Gradient(xt_fwdBack)
      map_est = so.minimize(xt_fwdBack, np.zeros(s.o.dy), method='BFGS', jac=g)

      norm = np.linalg.norm(map_est.x - mu)
      # print(f'norm: {norm:.6f}')
      assert norm < 1e-2, f'SED.sampleTranslationX bad, norm {norm:.6f}'

  def testTranslationX_fwd_obs(s):
    m = getattr(lie, s.o.lie)
    for t in range(1,s.T-1):
      y_t = s.y[t]

      # infer d_x_t
      # copy true R_x_t, but have incorrect d_x_t
      x_t = s.xTrue[t-1].copy()
      x_t[:-1,:-1] = s.xTrue[t,:-1,:-1]

      mu, Sigma = SED.sampleTranslationX(s.o, y_t, s.zTrue[t], x_t,
        s.thetaTrue[t], s.ETrue, s.QTrue, s.xTrue[t-1], returnMuSigma=True)

      # optimize same objective and compare
      Qi = np.linalg.inv(s.QTrue)
      def xt_fwd(v):
        # Set x_t candidate
        V = x_t.copy()
        V[:-1,-1] = v

        # log map and evaluate
        V_algebra = m.algi(m.logm(m.inv(s.xTrue[t-1]).dot(V)))
        nll = 0.5 * V_algebra.dot(Qi).dot(V_algebra)

        for k in range(s.KTrue):
          y_tk_world = y_t[s.zTrue[t]==k]
          T_part_world = m.inv(V.dot(s.thetaTrue[t,k]))
          y_tk_part = SED.TransformPointsNonHomog(T_part_world, y_tk_world)
          nll += np.sum( -mvn.logpdf(y_tk_part, np.zeros(s.o.dy), s.ETrue[k]) )

        return nll
      
      g = nd.Gradient(xt_fwd)
      map_est = so.minimize(xt_fwd, np.zeros(s.o.dy), method='BFGS', jac=g)

      norm = np.linalg.norm(map_est.x - mu)
      # print(f'norm: {norm:.6f}')

      assert norm < 1e-2, \
        f'SED.sampleTranslationX bad, norm {norm:.6f}, time {t}'

  def testTranslationX_fwdBack_obs(s):
    m = getattr(lie, s.o.lie)
    for t in range(1,s.T-1):
      y_t = s.y[t]

      # infer d_x_t
      # copy true R_x_t, but have incorrect d_x_t
      x_t = s.xTrue[t-1].copy()
      x_t[:-1,:-1] = s.xTrue[t,:-1,:-1]

      mu, Sigma = SED.sampleTranslationX(s.o, y_t, s.zTrue[t], x_t,
        s.thetaTrue[t], s.ETrue, s.QTrue, s.xTrue[t-1], x_tplus1=s.xTrue[t+1],
        returnMuSigma=True)

      # optimize same objective and compare
      Qi = np.linalg.inv(s.QTrue)
      def xt_fwdBack(v):
        # Set x_t candidate
        V1 = x_t.copy()
        V1[:-1,-1] = v

        # log map and evaluate
        V1_algebra = m.algi(m.logm(m.inv(s.xTrue[t-1]).dot(V1)))
        nll = 0.5 * V1_algebra.dot(Qi).dot(V1_algebra)

        V2_algebra = m.algi(m.logm(m.inv(V1).dot(s.xTrue[t+1])))
        nll += 0.5 * V2_algebra.dot(Qi).dot(V2_algebra)

        for k in range(s.KTrue):
          y_tk_world = y_t[s.zTrue[t]==k]
          T_part_world = m.inv(V1.dot(s.thetaTrue[t,k]))
          y_tk_part = SED.TransformPointsNonHomog(T_part_world, y_tk_world)
          nll += np.sum( -mvn.logpdf(y_tk_part, np.zeros(s.o.dy), s.ETrue[k]) )

        return nll
      
      g = nd.Gradient(xt_fwdBack)
      map_est = so.minimize(xt_fwdBack, np.zeros(s.o.dy), method='BFGS', jac=g)

      norm = np.linalg.norm(map_est.x - mu)
      # print(f'norm: {norm:.6f}')

      assert norm < 1e-2, \
        f'SED.sampleTranslationX bad, norm {norm:.6f}, time {t}'

  def testTranslationTheta_fwd_noObs(s):
    m = getattr(lie, s.o.lie)
    for t in range(1,s.T-1):
      # y_t = np.zeros((0, s.y[t].shape[1]))
      # z_t = np.zeros(0, dtype=np.int)

      # infer d_x_t
      # copy true R_x_t, but have incorrect d_x_t
      for k in range(s.KTrue):
        y_tk = np.zeros((0, s.y[t].shape[1]))

        theta_tk = s.thetaTrue[t-1,k].copy()
        theta_tk[:-1,:-1] = s.thetaTrue[t,k,:-1,:-1]
        mu, Sigma = SED.sampleTranslationTheta(s.o, y_tk, theta_tk,
          s.xTrue[t], s.STrue[k], s.ETrue[k], s.thetaTrue[t-1,k],
          returnMuSigma=True)

        # optimize same objective and compare
        Si = np.linalg.inv(s.STrue[k])
        def thetat_fwd(v):
          # Set theta_tk candidate
          V = theta_tk.copy()
          V[:-1,-1] = v

          # log map and evaluate
          V_algebra = m.algi(m.logm(m.inv(s.thetaTrue[t-1,k]).dot(V)))
          nll = 0.5 * V_algebra.dot(Si).dot(V_algebra)
          return nll
        
        g = nd.Gradient(thetat_fwd)
        map_est = so.minimize(thetat_fwd, np.zeros(s.o.dy), method='BFGS',
          jac=g)

        norm = np.linalg.norm(map_est.x - mu)
        assert norm < 1e-2, \
          f'SED.sampleTranslationTheta bad, norm {norm:.6f}'

  def testTranslationTheta_fwdBack_noObs(s):
    m = getattr(lie, s.o.lie)
    for t in range(1,s.T-1):

      # infer d_theta_tk
      # copy true R_theta_tk, but have incorrect d_theta_tk
      for k in range(s.KTrue):
        y_tk = np.zeros((0, s.y[t].shape[1]))

        theta_tk = s.thetaTrue[t-1,k].copy()
        theta_tk[:-1,:-1] = s.thetaTrue[t,k,:-1,:-1]
        mu, Sigma = SED.sampleTranslationTheta(s.o, y_tk, theta_tk,
          s.xTrue[t], s.STrue[k], s.ETrue[k], s.thetaTrue[t-1,k],
          theta_tplus1_k=s.thetaTrue[t+1,k], returnMuSigma=True)

        # optimize same objective and compare
        Si = np.linalg.inv(s.STrue[k])
        def thetat_fwd(v):
          # Set theta_tk candidate
          V1 = theta_tk.copy()
          V1[:-1,-1] = v

          # log map and evaluate
          V1_algebra = m.algi(m.logm(m.inv(s.thetaTrue[t-1,k]).dot(V1)))
          nll = 0.5 * V1_algebra.dot(Si).dot(V1_algebra)

          V2_algebra = m.algi(m.logm(m.inv(V1).dot(s.thetaTrue[t+1,k])))
          nll += 0.5 * V2_algebra.dot(Si).dot(V2_algebra)

          return nll

        g = nd.Gradient(thetat_fwd)
        map_est = so.minimize(thetat_fwd, np.zeros(s.o.dy), method='BFGS',
          jac=g)

        norm = np.linalg.norm(map_est.x - mu)
        assert norm < 1e-2, \
          f'SED.sampleTranslationTheta bad, norm {norm:.6f}'

  def testTranslationTheta_fwd_obs(s):
    m = getattr(lie, s.o.lie)
    for t in range(1,s.T-1):
      for k in range(s.KTrue):
        y_tk = s.y[t][s.zTrue[t]==k]

        theta_tk = s.thetaTrue[t-1,k].copy()
        theta_tk[:-1,:-1] = s.thetaTrue[t,k,:-1,:-1]
        mu, Sigma = SED.sampleTranslationTheta(s.o, y_tk, theta_tk,
          s.xTrue[t], s.STrue[k], s.ETrue[k], s.thetaTrue[t-1,k],
          returnMuSigma=True)

        # optimize same objective and compare
        Si = np.linalg.inv(s.STrue[k])
        def thetat_fwd(v):
          # Set theta_tk candidate
          V = theta_tk.copy()
          V[:-1,-1] = v

          # log map and evaluate
          V_algebra = m.algi(m.logm(m.inv(s.thetaTrue[t-1,k]).dot(V)))
          nll = 0.5 * V_algebra.dot(Si).dot(V_algebra)

          y_tk_world = y_tk
          T_part_world = m.inv(s.xTrue[t].dot(V))
          y_tk_part = SED.TransformPointsNonHomog(T_part_world, y_tk_world)
          nll += np.sum( -mvn.logpdf(y_tk_part, np.zeros(s.o.dy), s.ETrue[k]) )

          return nll
        
        g = nd.Gradient(thetat_fwd)
        map_est = so.minimize(thetat_fwd, np.zeros(s.o.dy), method='BFGS',
          jac=g)

        norm = np.linalg.norm(map_est.x - mu)
        assert norm < 1e-2, \
          f'SED.sampleTranslationTheta bad, norm {norm:.6f}'

  def testTranslationTheta_fwdBack_obs(s):
    m = getattr(lie, s.o.lie)
    for t in range(1,s.T-1):
      for k in range(s.KTrue):
        y_tk = s.y[t][s.zTrue[t]==k]

        theta_tk = s.thetaTrue[t-1,k].copy()
        theta_tk[:-1,:-1] = s.thetaTrue[t,k,:-1,:-1]
        mu, Sigma = SED.sampleTranslationTheta(s.o, y_tk, theta_tk,
          s.xTrue[t], s.STrue[k], s.ETrue[k], s.thetaTrue[t-1,k],
          theta_tplus1_k=s.thetaTrue[t+1,k], returnMuSigma=True)

        # optimize same objective and compare
        Si = np.linalg.inv(s.STrue[k])
        def thetat_fwd(v):
          # Set theta_tk candidate
          V1 = theta_tk.copy()
          V1[:-1,-1] = v

          # log map and evaluate
          V1_algebra = m.algi(m.logm(m.inv(s.thetaTrue[t-1,k]).dot(V1)))
          nll = 0.5 * V1_algebra.dot(Si).dot(V1_algebra)
          
          # forward term
          V2_algebra = m.algi(m.logm(m.inv(V1).dot(s.thetaTrue[t+1,k])))
          nll += 0.5 * V2_algebra.dot(Si).dot(V2_algebra)

          # obs
          y_tk_world = y_tk
          T_part_world = m.inv(s.xTrue[t].dot(V1))
          y_tk_part = SED.TransformPointsNonHomog(T_part_world, y_tk_world)
          nll += np.sum( -mvn.logpdf(y_tk_part, np.zeros(s.o.dy), s.ETrue[k]) )

          return nll
        
        g = nd.Gradient(thetat_fwd)
        map_est = so.minimize(thetat_fwd, np.zeros(s.o.dy), method='BFGS',
          jac=g)

        norm = np.linalg.norm(map_est.x - mu)
        assert norm < 1e-2, \
          f'SED.sampleTranslationTheta bad, norm {norm:.6f}'

  ## Todo: figure out better way to automatically test rotation sampler
  ## either
  ##   1. take multiple samples and look at Riemannian distance to true
  ##   2. take multiple samples and look at Riemannian distance to optimal
  
  # def testRotationTheta_fwd(s):
  #   import matplotlib.pyplot as plt
  #   m = getattr(lie, s.o.lie)
  #
  #   for t in range(1,s.T):
  #     plt.figure()
  #     plt.title(f'{t}')
  #     for k in range(s.KTrue):
  #       theta_tk = s.thetaTrue[t-1,k].copy()
  #       theta_tk[:-1,-1] = s.thetaTrue[t,k,:-1,-1]
  #       y_tk = s.y[t][s.zTrue[t]==k]
  #
  #       R_theta_tk = SED.sampleRotationTheta(s.o, y_tk, theta_tk, s.xTrue[t],
  #         s.STrue[k], s.ETrue[k], s.thetaTrue[t-1,k])
  #       theta_tk[:-1,:-1] = R_theta_tk
  #
  #       m.plot(theta_tk, colors=np.array([[1., 0, 0], [1., 0, 0]]))
  #       m.plot(s.thetaTrue[t,k], colors=np.array([[0., 0, 0], [0., 0, 0]]))
  #   plt.show()
  #
  # def testRotationTheta_fwdBack(s):
  #   import matplotlib.pyplot as plt
  #   m = getattr(lie, s.o.lie)
  #
  #   for t in range(1,s.T-1):
  #     plt.figure()
  #     plt.title(f'{t}')
  #     for k in range(s.KTrue):
  #       theta_tk = s.thetaTrue[t-1,k].copy()
  #       theta_tk[:-1,-1] = s.thetaTrue[t,k,:-1,-1]
  #       y_tk = s.y[t][s.zTrue[t]==k]
  #
  #       R_theta_tk = SED.sampleRotationTheta(s.o, y_tk, theta_tk, s.xTrue[t],
  #         s.STrue[k], s.ETrue[k], s.thetaTrue[t-1,k],
  #         theta_tplus1_k=s.thetaTrue[t+1,k])
  #       theta_tk[:-1,:-1] = R_theta_tk
  #
  #       m.plot(theta_tk, colors=np.array([[1., 0, 0], [1., 0, 0]]))
  #       m.plot(s.thetaTrue[t,k], colors=np.array([[0., 0, 0], [0., 0, 0]]))
  #   plt.show()
  #
  # def testRotationX_fwdBack(s):
  #   m = getattr(lie, s.o.lie)
  #
  #   for t in range(1,s.T-1):
  #     x_t = s.xTrue[t-1].copy()
  #     x_t[:-1,-1] = s.xTrue[t,:-1,-1]
  #     R_x_t = SED.sampleRotationX(s.o, s.y[t], s.zTrue[t], x_t, s.thetaTrue[t],
  #       s.ETrue, s.QTrue, s.xTrue[t-1], x_tplus1=s.xTrue[t+1])
  #     x_t[:-1,:-1] = R_x_t
  #
  #     import matplotlib.pyplot as plt
  #     plt.figure()
  #     m.plot(x_t, colors=np.array([[1., 0, 0], [1., 0, 0]]))
  #     m.plot(s.xTrue[t], colors=np.array([[0., 0, 0], [0., 0, 0]]))
  #     plt.title(f'{t}')
  #   plt.show()
  #
  # def testRotationX_fwd(s):
  #   m = getattr(lie, s.o.lie)
  #
  #   for t in range(1,s.T):
  #     x_t = s.xTrue[t-1].copy()
  #     x_t[:-1,-1] = s.xTrue[t,:-1,-1]
  #     R_x_t = SED.sampleRotationX(s.o, s.y[t], s.zTrue[t], x_t, s.thetaTrue[t],
  #       s.ETrue, s.QTrue, s.xTrue[t-1])
  #     x_t[:-1,:-1] = R_x_t
  #
  #     import matplotlib.pyplot as plt
  #     plt.figure()
  #     m.plot(x_t, colors=np.array([[1., 0, 0], [1., 0, 0]]))
  #     m.plot(s.xTrue[t], colors=np.array([[0., 0, 0], [0., 0, 0]]))
  #     plt.title(f'{t}')
  #   plt.show()


    # print(m.dist2(s.xTrue[0], s.xTrue[1], d=0))
    # for t in range(1,s.T):
    #   # copy true d_x_t, but have incorrect d_x_t
    #   x_t = s.xTrue[t-1].copy()
    #   x_t[:-1,-1] = s.xTrue[t,:-1,-1]
    #   R_x_t = SED.sampleRotationX(s.o, s.y[t], s.zTrue[t], x_t, s.thetaTrue[t],
    #     s.ETrue, s.QTrue, s.xTrue[t-1])
    #   x_t[:-1,:-1] = R_x_t
    #
    #   dst_sample_true = m.dist2(x_t, s.xTrue[t], d=0)
    #   dst_true_true = m.dist2(s.xTrue[t-1], s.xTrue[t], d=0)
    #   print(dst_sample_true)
    #   print(dst_true_true)
    #   assert dst_sample_true < 1e-2
    #   assert dst_sample_true <= dst_true_true
    #
    # # print(x_t)
    # # print(s.xTrue[t])
    #

  def testNewTranslationX_fwd_noObs(s):
    m = getattr(lie, s.o.lie)
    for t in range(1,s.T-1):
      # infer d_x_t
      R_x_t = m.Rt(s.xTrue[t])[0]

      mu, Sigma = SED.translationDynamicsPosterior(
        s.o, R_x_t, s.xTrue[t-1], s.QTrue)

      # optimize same objective and compare
      Qi = np.linalg.inv(s.QTrue)

      ## copy true R_x_t, but have incorrect d_x_t
      x_t = s.xTrue[t-1].copy()
      x_t[:-1,:-1] = R_x_t
      def xt_fwd(v):
        # Set x_t candidate
        V = x_t.copy()
        V[:-1,-1] = v

        # log map and evaluate
        V_algebra = m.algi(m.logm(m.inv(s.xTrue[t-1]).dot(V)))
        nll = 0.5 * V_algebra.dot(Qi).dot(V_algebra)
        return nll
      
      g = nd.Gradient(xt_fwd)
      map_est = so.minimize(xt_fwd, np.zeros(s.o.dy), method='BFGS', jac=g)

      norm = np.linalg.norm(map_est.x - mu)
      # print(f'norm: {norm:.6f}')
      assert norm < 1e-2, f'SED.translationDynamicsPosterior bad, norm {norm:.6f}'

  def testNewTranslationX_fwdBack_noObs(s):
    m = getattr(lie, s.o.lie)
    for t in range(1,s.T-1):

      R_x_t = m.Rt(s.xTrue[t])[0]
      mu, Sigma = SED.translationDynamicsPosterior(
        s.o, R_x_t, s.xTrue[t-1], s.QTrue, nextU=s.xTrue[t+1])

      # optimize same objective and compare
      Qi = np.linalg.inv(s.QTrue)
      x_t = s.xTrue[t-1].copy()
      x_t[:-1,:-1] = s.xTrue[t,:-1,:-1]
      def xt_fwdBack(v):
        # Set x_t candidate
        V1 = x_t.copy()
        V1[:-1,-1] = v

        # log map and evaluate
        V1_algebra = m.algi(m.logm(m.inv(s.xTrue[t-1]).dot(V1)))
        nll = 0.5 * V1_algebra.dot(Qi).dot(V1_algebra)

        V2_algebra = m.algi(m.logm(m.inv(V1).dot(s.xTrue[t+1])))
        nll += 0.5 * V2_algebra.dot(Qi).dot(V2_algebra)

        return nll
      
      g = nd.Gradient(xt_fwdBack)
      map_est = so.minimize(xt_fwdBack, np.zeros(s.o.dy), method='BFGS', jac=g)

      norm = np.linalg.norm(map_est.x - mu)
      # print(f'norm: {norm:.6f}')
      assert norm < 1e-2, f'SED.translationDynamicsPosterior bad, norm {norm:.6f}'

  def testNewTranslationX_fwdBack_obs(s):
    m = getattr(lie, s.o.lie)
    o = s.o

    K = s.ETrue.shape[0]
    for t in range(1,s.T-1):
      y_t = s.y[t]

      R_x_t = m.Rt(s.xTrue[t])[0]
      mu, Sigma = SED.translationDynamicsPosterior(
        s.o, R_x_t, s.xTrue[t-1], s.QTrue, nextU=s.xTrue[t+1])
      for k in range(K):
        ys = y_t[s.zTrue[t]==k]
        lhs = np.eye(o.dy+1)
        rhs = s.thetaTrue[t,k]
        mu, Sigma = SED.translationObservationPosterior(o, ys, R_x_t, lhs, rhs,
          s.ETrue[k], mu, Sigma)

      # optimize same objective and compare
      Qi = np.linalg.inv(s.QTrue)
      x_t = s.xTrue[t-1].copy()
      x_t[:-1,:-1] = s.xTrue[t,:-1,:-1]
      def xt_fwdBack(v):
        # Set x_t candidate
        V1 = x_t.copy()
        V1[:-1,-1] = v

        # log map and evaluate
        V1_algebra = m.algi(m.logm(m.inv(s.xTrue[t-1]).dot(V1)))
        nll = 0.5 * V1_algebra.dot(Qi).dot(V1_algebra)

        V2_algebra = m.algi(m.logm(m.inv(V1).dot(s.xTrue[t+1])))
        nll += 0.5 * V2_algebra.dot(Qi).dot(V2_algebra)

        for k in range(s.KTrue):
          y_tk_world = y_t[s.zTrue[t]==k]
          T_part_world = m.inv(V1.dot(s.thetaTrue[t,k]))
          y_tk_part = SED.TransformPointsNonHomog(T_part_world, y_tk_world)
          nll += np.sum( -mvn.logpdf(y_tk_part, np.zeros(s.o.dy), s.ETrue[k]) )

        return nll
      
      g = nd.Gradient(xt_fwdBack)
      map_est = so.minimize(xt_fwdBack, np.zeros(s.o.dy), method='BFGS', jac=g)

      norm = np.linalg.norm(map_est.x - mu)
      assert norm < 1e-2, \
        f'SED.translationDynamicsPosterior bad, norm {norm:.6f}, time {t}'

  def testNewTranslationTheta_fwd_noObs(s):
    m = getattr(lie, s.o.lie)
    o = s.o
    K = s.ETrue.shape[0]

    for t in range(1,s.T-1):
      for k in range(s.KTrue):
        R_theta_tk = m.Rt(s.thetaTrue[t,k])[0]
        
        mu, Sigma = SED.translationDynamicsPosterior(
          s.o, R_theta_tk, s.thetaTrue[t-1,k], s.STrue[k])

        # optimize same objective and compare
        Si = np.linalg.inv(s.STrue[k])
        theta_tk = s.thetaTrue[t-1,k].copy()
        theta_tk[:-1,:-1] = s.thetaTrue[t,k,:-1,:-1]
        def thetat_fwd(v):
          # Set theta_tk candidate
          V = theta_tk.copy()
          V[:-1,-1] = v

          # log map and evaluate
          V_algebra = m.algi(m.logm(m.inv(s.thetaTrue[t-1,k]).dot(V)))
          nll = 0.5 * V_algebra.dot(Si).dot(V_algebra)
          return nll
        
        g = nd.Gradient(thetat_fwd)
        map_est = so.minimize(thetat_fwd, np.zeros(s.o.dy), method='BFGS',
          jac=g)

        norm = np.linalg.norm(map_est.x - mu)
        assert norm < 1e-2, \
          f'SED.translationDynamicsPosterior bad, norm {norm:.6f}'

  def testNewTranslationTheta_fwdBack_noObs(s):
    m = getattr(lie, s.o.lie)
    o = s.o
    K = s.ETrue.shape[0]

    for t in range(1,s.T-1):
      for k in range(s.KTrue):
        R_theta_tk = m.Rt(s.thetaTrue[t,k])[0]
        mu, Sigma = SED.translationDynamicsPosterior(
          s.o, R_theta_tk, s.thetaTrue[t-1,k], s.STrue[k],
          nextU=s.thetaTrue[t+1,k])

        # optimize same objective and compare
        Si = np.linalg.inv(s.STrue[k])
        theta_tk = s.thetaTrue[t-1,k].copy()
        theta_tk[:-1,:-1] = s.thetaTrue[t,k,:-1,:-1]
        def thetat_fwd(v):
          # Set theta_tk candidate
          V1 = theta_tk.copy()
          V1[:-1,-1] = v

          # log map and evaluate
          V1_algebra = m.algi(m.logm(m.inv(s.thetaTrue[t-1,k]).dot(V1)))
          nll = 0.5 * V1_algebra.dot(Si).dot(V1_algebra)

          V2_algebra = m.algi(m.logm(m.inv(V1).dot(s.thetaTrue[t+1,k])))
          nll += 0.5 * V2_algebra.dot(Si).dot(V2_algebra)

          return nll

        g = nd.Gradient(thetat_fwd)
        map_est = so.minimize(thetat_fwd, np.zeros(s.o.dy), method='BFGS',
          jac=g)

        norm = np.linalg.norm(map_est.x - mu)
        assert norm < 1e-2, \
          f'SED.sampleTranslationTheta bad, norm {norm:.6f}'

  def testNewTranslationTheta_fwdBack_obs(s):
    m = getattr(lie, s.o.lie)
    o = s.o

    for t in range(1,s.T-1):
      for k in range(s.KTrue):
        y_tk = s.y[t][s.zTrue[t]==k]

        R_theta_tk = m.Rt(s.thetaTrue[t,k])[0]
        mu, Sigma = SED.translationDynamicsPosterior(
          s.o, R_theta_tk, s.thetaTrue[t-1,k], s.STrue[k],
          nextU=s.thetaTrue[t+1,k])

        ys = y_tk
        lhs = s.xTrue[t]
        rhs = np.eye(o.dy+1)
        mu, Sigma = SED.translationObservationPosterior(o, ys, R_theta_tk, lhs, rhs,
          s.ETrue[k], mu, Sigma)

        # optimize same objective and compare
        Si = np.linalg.inv(s.STrue[k])
        theta_tk = s.thetaTrue[t-1,k].copy()
        theta_tk[:-1,:-1] = s.thetaTrue[t,k,:-1,:-1]
        def thetat_fwd(v):
          # Set theta_tk candidate
          V1 = theta_tk.copy()
          V1[:-1,-1] = v

          # log map and evaluate
          V1_algebra = m.algi(m.logm(m.inv(s.thetaTrue[t-1,k]).dot(V1)))
          nll = 0.5 * V1_algebra.dot(Si).dot(V1_algebra)
          
          # forward term
          V2_algebra = m.algi(m.logm(m.inv(V1).dot(s.thetaTrue[t+1,k])))
          nll += 0.5 * V2_algebra.dot(Si).dot(V2_algebra)

          # obs
          y_tk_world = y_tk
          T_part_world = m.inv(s.xTrue[t].dot(V1))
          y_tk_part = SED.TransformPointsNonHomog(T_part_world, y_tk_world)
          nll += np.sum( -mvn.logpdf(y_tk_part, np.zeros(s.o.dy), s.ETrue[k]) )

          return nll
        
        g = nd.Gradient(thetat_fwd)
        map_est = so.minimize(thetat_fwd, np.zeros(s.o.dy), method='BFGS',
          jac=g)

        norm = np.linalg.norm(map_est.x - mu)
        assert norm < 1e-2, \
          f'SED.translationObservationPosterior bad, norm {norm:.6f}'


class test_se2_randomwalk3(unittest.TestCase):
  def setUp(s):
    data = du.load('data/synthetic/se2_randomwalk3/data')
    s.o = SED.opts(lie='se2')
    s.T = len(data['x'])
    s.y = data['y']

    # Ground-truth
    s.KTrue = data['theta'].shape[1]
    s.xTrue = du.asShape(data['x'], (s.T,) + s.o.dxGm)
    s.thetaTrue = data['theta']
    s.ETrue = data['E']
    s.STrue = data['S']
    s.QTrue = data['Q']
    s.zTrue = data['z']
    s.piTrue = data['pi']
    s.data = data

  def testInitRJMCMC_spider(s):
    alpha = 0.1
    mL_const = -14.0

    o = SED.opts(lie='se2')
    y = du.load('../npp-data/se2_spider/data')['y']
    T = len(y)
    maxObs = 3000
    if maxObs > 0:
      y = [ yt[np.random.choice(range(len(yt)), min(maxObs, len(yt)), replace=False)] for yt in y ]

    SED.initPriorsDataDependent(o, y)
    x = SED.initXDataMeans(o, y)
    Q = SED.inferQ(o, x)

    mL = [ mL_const * np.ones(y[t].shape[0]) for t in range(T) ]
    theta, E, S, z, pi = SED.initPartsAndAssoc(o, y, x, alpha, mL)

    # theta = np.zeros((T, 0,) + o.dxGm)
    # E = np.zeros((0, o.dy, o.dy))
    # S = np.zeros((0, o.dxA, o.dxA))
    # Q = SED.inferQ(o, x)
    # mL = [ mL_const * np.ones(y[t].shape[0]) for t in range(T) ]
    # pi = np.array([1.0,])
    # z = [ -1 * np.ones(y[t].shape[0], dtype=np.int) for t in range(T) ]

    llInit = SED.logJoint(o, y, z, x, theta, E, S, Q, alpha, pi, mL)
    print(f'llInit: {llInit:.2f}')

    pBirth, pDeath = (0.1, 0.3)
    nSamples = 10000
    ll = np.zeros(nSamples+1)
    ll[0] = llInit
    nBirthProp, nBirthAccept, nDeathProp, nDeathAccept = (0, 0, 0, 0)
    for nS in range(nSamples):
      dontSampleX = True if nS < 0 else False
      z, pi, theta, E, S, x, Q, mL, move, accept = SED.sampleRJMCMC( o, y,
        alpha, z, pi, theta, E, S, x, Q, mL, pBirth, pDeath,
        dontSampleX=dontSampleX)
      ll[nS+1] = SED.logJoint(o, y, z, x, theta, E, S, Q, alpha, pi, mL)

      if move == 'birth':
        nBirthProp += 1
        if accept: nBirthAccept += 1
      elif move == 'death':
        nDeathProp += 1
        if accept: nDeathAccept += 1

      a = '+' if accept == True else '-'
      print(f'Iter {nS:05}, LL: {ll[nS+1]:.2f}, K: {len(pi)-1}, Move: {move[0]}{a}')

      # if nS % 10 == 0:
      filename = f'tmp/rjmcmc_spider/sample-{nS:05}'
      SED.saveSample(filename, o, alpha, z, pi, theta, E, S, x, Q, ll=ll[nS+1])

    plt.figure()
    plt.plot(ll, c='b', label='Sample LL')
    plt.legend()
    plt.title(f'nBirthProp: {nBirthProp}, nBirthAccept: {nBirthAccept}, nDeathProp: {nDeathProp}, nDeathAccept: {nDeathAccept}')
    plt.show()

    from npp import drawSED
    drawSED.draw(o, y=y, x=x, z=z, theta=theta, E=E)
    plt.show()

  def testSkeletonize(s):
    o = SED.opts(lie='se2')
    # y = du.load('../npp-data/se2_waving_hand/data')['y']
    y = du.load('../npp-data/se2_spider/data')['y']
    T = len(y)
    # maxObs = 3000
    # if maxObs > 0:
    #   y = [ yt[np.random.choice(range(len(yt)), min(maxObs, len(yt)), replace=False)] for yt in y ]
    from skimage.morphology import medial_axis

    skels = [ ]
    for t in range(T):
      # make binary image out of points
      minI = np.min(y[t], axis=0).astype(np.int)
      maxI = np.max(y[t], axis=0).astype(np.int)
      im = np.zeros((maxI[1]+1 - minI[1], maxI[0]+1 - minI[0]), dtype=np.bool)
      for ytn in y[t]:
        i = ytn[1].astype(np.int) - minI[1]
        j = ytn[0].astype(np.int) - minI[0]
        im[i, j] = True
      skel = medial_axis(im)
      skels.append(skel)

    du.ViewImgs(skels)
    plt.show()

    # plt.figure()
    # plt.imshow(im)
    # plt.figure()
    # plt.imshow(skel)
    # plt.show()

    # skel, dist = medial_axis(im)

    ip.embed()

  def test_interp(s):
    o = SED.opts(lie='se2')
    m = getattr(lie, o.lie)


    def fL(a, b, lam):
      inner = lam * m.logm(m.inv(a) @ b)
      return a @ m.expm(inner)

    a = m.rvs()
    b = m.rvs()

    T = 10
    theta = np.zeros((T,) + o.dxGm)
    for t, lam in enumerate(np.linspace(0, 1, 10)):
      theta[t] = fL(a,b,lam)
      if t==0 or t==T-1: l = 1.0
      else: l = 0.3
      m.plot(theta[t], l=l)

    plt.gca().set_aspect('equal', 'box')
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.gca().axis('off')
    plt.savefig('interp.png', dpi=300)
    plt.show()

  def test_likelihood_transform(s):
    sample = 'tmp/initialization_output/se2_waving_hand/se2_waving_hand'
    o, alpha, z, pi, theta, E, S, x, Q, mL, ll, subsetIdx, dataset = \
      SED.loadSample(sample)
    T, K = theta.shape[:2]
    data = du.load(f'{dataset}/data')
    y = data['y']
    y = [ y[t][subsetIdx[t]] for t in range(T) ]
    m = getattr(lie, o.lie)

    ll = SED.logJoint(o, y, z, x, theta, E, S, Q, alpha, pi, mL)

    fnames = [ f'tmp/draw/likelihood_reparam/original/img-{t:05}.png' for t in range(T) ]
    titles = [ f'{ll:.2f}' for t in range(T) ]
    draw.draw(o, y=y, x=x, theta=theta, z=z, E=E, filename=fnames, title=titles)

    # w = mvn.rvs(size=o.dxA)
    w = np.array([100.0, 0, 0])
    W = m.expm(m.alg(w))
    Wi = m.inv(W)

    # adjust x to x * w
    # adjust theta to w^{-1} * theta
    for t in range(T):
      x[t] = x[t] @ W
      for k in range(K):
        theta[t,k] = Wi @ theta[t,k]

    llNew = SED.logJoint(o, y, z, x, theta, E, S, Q, alpha, pi, mL)

    fnames = [ f'tmp/draw/likelihood_reparam/modified/img-{t:05}.png' for t in range(T) ]
    titles = [ f'{llNew:.2f}' for t in range(T) ]
    draw.draw(o, y=y, x=x, theta=theta, z=z, E=E, filename=fnames, title=titles)

    print(ll)
    print(llNew)

    # plt.show()
    # ip.embed()
    #
    #

  def test_partOffsetDynamics2(s):
    sample = 'tmp/initialization_output/se2_waving_hand/se2_waving_hand'
    # sample = 'tmp/initialization_output/se2_spider/se2_spider'
    # sample = 'tmp/initialization_output/se2_marmoset/tInit/k3/se2_marmoset_k3.gz'
    o, alpha, z, pi, theta, E, S, x, Q, mL, ll, subsetIdx, dataset = \
      SED.loadSample(sample)
    data = du.load(f'{dataset}/data')
    y = data['y']
    m = getattr(lie, o.lie)
    T, K = theta.shape[:2]

    # karcher mean of each part and MLE covariance
    theta0 = np.stack([ m.karcher(theta[:,k]) for k in range(K) ])
    S0 = np.zeros((K, o.dxA, o.dxA))
    for k in range(K):
      sk = np.zeros((T, o.dxA))
      for t in range(1,T):
        sk[t] = m.algi(m.logm(m.inv(theta[t-1,k]).dot(theta[t,k])))
      sk[0] = m.algi(m.logm(m.inv(o.H_theta[1]).dot(theta[0,k])))
      S0[k] = np.cov(sk.T) / 4.
      # S0[k] = np.cov(sk.T) / 4.
      # S0[k] = np.cov(sk.T) / 2.

    zero = np.zeros(o.dy)
    colors = du.diffcolors(100, bgCols=[[1,1,1],[0,0,0]])

    # generate hand dynamics with
    #   theta_t = f(theta_{t-1}, theta_0) e^{s_t}
    # where
    #   f_L(a, b, lambda) = a e^{lambda log(a^{-1} b)}
    #

    lam = 1.0
    # lam = 0.0

    T_ = T
    # T_ = 2

    zeroA = np.zeros(o.dxA)
    theta_ = np.zeros((T_, K) + o.dxGm)
    for t in range(T):
      for k in range(K):
        thetaPrev = theta0[k] if t==0 else theta_[t-1,k]


        theta_[t,k] = SED.mvnL_rv(o, f_, S[k])

        # # Stk = SED.mvnL_rv(o, theta0[k], S0[k])
        # Stk = SED.mvnL_rv(o, zeroA, S0[k])
        # theta_[t,k] = fL(thetaPrev, theta0[k], lam) @ Stk
        # # theta_[t,k] = fR(thetaPrev, theta0[k], lam) @ Stk

    fnames = [ f'tmp/draw/offsetDynamics/se2_waving_hand_lam_1_0/img-{t:03}.png' for t in range(T) ]
    draw.draw(o, x=x, theta=theta_, E=E, reverseY=False, filename=fnames,
      xlim=data['xlim'], ylim=data['ylim'])
    # plt.show()

  ####

  def test_partOffsetDynamics(s):
    sample = 'tmp/initialization_output/se2_waving_hand/se2_waving_hand'
    # sample = 'tmp/initialization_output/se2_spider/se2_spider'
    # sample = 'tmp/initialization_output/se2_marmoset/tInit/k3/se2_marmoset_k3.gz'
    o, alpha, z, pi, theta, E, S, x, Q, mL, ll, subsetIdx, dataset = \
      SED.loadSample(sample)
    data = du.load(f'{dataset}/data')
    y = data['y']
    m = getattr(lie, o.lie)
    T, K = theta.shape[:2]

    # karcher mean of each part and MLE covariance
    theta0 = np.stack([ m.karcher(theta[:,k]) for k in range(K) ])
    S0 = np.zeros((K, o.dxA, o.dxA))
    for k in range(K):
      sk = np.zeros((T, o.dxA))
      for t in range(1,T):
        sk[t] = m.algi(m.logm(m.inv(theta[t-1,k]).dot(theta[t,k])))
      sk[0] = m.algi(m.logm(m.inv(o.H_theta[1]).dot(theta[0,k])))
      S0[k] = np.cov(sk.T) / 4.
      # S0[k] = np.cov(sk.T) / 4.
      # S0[k] = np.cov(sk.T) / 2.

    zero = np.zeros(o.dy)
    colors = du.diffcolors(100, bgCols=[[1,1,1],[0,0,0]])

    # generate hand dynamics with
    #   theta_t = f(theta_{t-1}, theta_0) e^{s_t}
    # where
    #   f_L(a, b, lambda) = a e^{lambda log(a^{-1} b)}
    #

    lam = 1.0
    # lam = 0.0

    T_ = T
    # T_ = 2

    def fL(a, b, lam):
      inner = lam * m.logm(m.inv(a) @ b)
      return a @ m.expm(inner)
    def fR(a, b, lam):
      inner = lam * m.logm(b @ m.inv(a))
      return m.expm(inner) @ a
    # f = fR
    f = fL

    def g(a, b, lam): return lam*a + (1-lam)*b
    
    zeroA = np.zeros(o.dxA)
    theta_ = np.zeros((T_, K) + o.dxGm)
    for t in range(T):
      for k in range(K):
        thetaPrev = theta0[k] if t==0 else theta_[t-1,k]

        f_ = f(thetaPrev, theta0[k], lam)
        # f_ = f(theta0[k], thetaPrev, lam)

        # if t == 1:
        #   ip.embed()
        #   sys.exit()

        # assert np.allclose(f_, theta0[k])

        # f_ = thetaPrev
        Sigma = g(S[k], S0[k], lam)
        theta_[t,k] = SED.mvnL_rv(o, f_, Sigma)

        # # Stk = SED.mvnL_rv(o, theta0[k], S0[k])
        # Stk = SED.mvnL_rv(o, zeroA, S0[k])
        # theta_[t,k] = fL(thetaPrev, theta0[k], lam) @ Stk
        # # theta_[t,k] = fR(thetaPrev, theta0[k], lam) @ Stk

    fnames = [ f'tmp/draw/offsetDynamics/se2_waving_hand_lam_1_0/img-{t:03}.png' for t in range(T) ]
    draw.draw(o, x=x, theta=theta_, E=E, reverseY=False, filename=fnames,
      xlim=data['xlim'], ylim=data['ylim'])
    # plt.show()

  def test_partOffsetGenerative(s):
    sample = 'tmp/initialization_output/se2_waving_hand/se2_waving_hand'
    # sample = 'tmp/initialization_output/se2_spider/se2_spider'
    # sample = 'tmp/initialization_output/se2_marmoset/tInit/k3/se2_marmoset_k3.gz'
    o, alpha, z, pi, theta, E, S, x, Q, mL, ll, subsetIdx, dataset = \
      SED.loadSample(sample)
    y = du.load(f'{dataset}/data')['y']
    m = getattr(lie, o.lie)
    T, K = theta.shape[:2]

    # karcher mean of each part and MLE covariance
    theta0 = np.stack([ m.karcher(theta[:,k]) for k in range(K) ])
    S0 = np.zeros((K, o.dxA, o.dxA))
    for k in range(K):
      sk = np.zeros((T, o.dxA))
      for t in range(1,T):
        sk[t] = m.algi(m.logm(m.inv(theta[t-1,k]).dot(theta[t,k])))
      sk[0] = m.algi(m.logm(m.inv(o.H_theta[1]).dot(theta[0,k])))
      # S0[k] = np.cov(sk.T) / 4.
      S0[k] = np.cov(sk.T) / 2.

    zero = np.zeros(o.dy)
    colors = du.diffcolors(100, bgCols=[[1,1,1],[0,0,0]])

    # # draw part with mean transformation
    # for k in range(K):
    #   thetaMu = theta0[k]
    #   T_obj_part = thetaMu
    #   yMu = SED.TransformPointsNonHomog(T_obj_part, zero)
    #   R = T_obj_part[:-1,:-1]
    #   ySig = R.dot(E[k]).dot(R.T)
    #   plt.plot( *du.stats.Gauss2DPoints(yMu, ySig, deviations=1.25), c=colors[k],
    #     linestyle='-' )

    # sample part realizations
    nSamples = 20
    colorsA = du.diffcolors(100, bgCols=[[1,1,1],[0,0,0]], alpha=1.0)
    for nS in range(nSamples):
      plt.figure()
      for k in range(K):
        thetaMu = SED.mvnL_rv(o, theta0[k], S0[k])
        T_obj_part = thetaMu
        yMu = SED.TransformPointsNonHomog(T_obj_part, zero)
        R = T_obj_part[:-1,:-1]
        ySig = R.dot(E[k]).dot(R.T)
        plt.plot( *du.stats.Gauss2DPoints(yMu, ySig, deviations=1.25), c=colorsA[k],
          linestyle='--' )

      # invisibly plot observations to set scale
      t = 0
      yObj_t = SED.TransformPointsNonHomog(m.inv(x[t]), y[t])
      plt.scatter(yObj_t[:,0], yObj_t[:,1], s=0)
      plt.gca().set_aspect('equal', 'box')
      plt.gca().set_xticks([])
      plt.gca().set_yticks([])
      plt.gca().invert_yaxis()
      # plt.title('Part Sample')
      plt.gca().axis('off')
      plt.savefig(f'part_sample_{nS:03}.png', dpi=300, bbox_inches='tight')
      plt.close()

    # plt.show()

    
    # ip.embed()

  # come back to me 
  def testControlledInference(s):
    sample = 'tmp/initialization_output_omega/se2_waving_hand/se2_waving_hand'
    # sample = 'tmp/initialization_output_omega/se2_spider/se2_spider'
    # sample = 'tmp/initialization_output_omega/se2_waving_hand/se2_waving_hand'
    o, alpha, z, pi, theta, E, S, x, Q, omegaOld, mL, ll, subsetIdx, dataset = \
      SED.loadSample(sample)

    yAll = du.load(f'{dataset}/data')['y']
    if subsetIdx is not None:
      y = [yt[subsetIdx[t]] for t, yt in enumerate(yAll)]
    else:
      y = yAll

    T = len(y)
    K = E.shape[0]
    m = getattr(lie, o.lie)

    t, k = (5, 4)
    R_cur, d_cur = m.Rt(theta[t,k])

    print(d_cur)
    a = 0.95
    A = np.sqrt(a * np.eye(o.dy))
    beta = 0.05
    B = np.sqrt(beta * np.eye(o.dy))

    mu, Sigma = SED.thetaTranslationDynamicsPosterior(o, R_cur, theta[t-1,k],
      S[k], A, B, thetaNext=theta[t+1,k])
    # ip.embed()

    # for t in range(1,T-1):
    #   for k in range(K):
    #     R_cur = m.Rt(theta[t,k])[0]
    #     mu, Sigma = SED.sampleThetaTranslation(o, R_cur, theta[t-1,k], S[k], omega[k], lam,
    #       thetaNext=theta[t+1,k])




  def testFeedbackMCMC2(s):
    sample = 'tmp/initialization_output_omega/se2_waving_hand/se2_waving_hand'
    # sample = 'tmp/initialization_output_omega/se2_spider/se2_spider'
    # sample = 'tmp/initialization_output_omega/se2_waving_hand/se2_waving_hand'
    o, alpha, z, pi, theta, E, S, x, Q, omegaOld, mL, ll, subsetIdx, dataset = \
      SED.loadSample(sample)

    yAll = du.load(f'{dataset}/data')['y']
    if subsetIdx is not None:
      y = [yt[subsetIdx[t]] for t, yt in enumerate(yAll)]
    else:
      y = yAll

    T = len(y)
    K = E.shape[0]
    m = getattr(lie, o.lie)
    lam = 0.1

    # get back to old parts
    for t in range(T):
      for k in range(K):
        theta[t,k] = omegaOld[k] @ theta[t,k]

    # sample omega
    omegaKarcher = np.stack([ m.karcher(theta[:,k]) for k in range(K) ])
    omega = np.zeros((K,) + o.dxGm)

    # for k in range(K):
    #   w = SED.sampleOmega(o, theta[:,k], S[k], nSamples=100, lam=lam)
    #   omega[k] = m.expm(m.alg(w))
    #
    # thetaOmega = np.tile(omega, (T, 1, 1, 1))
    #
    # # draw.draw(o, y=y, x=x, theta=thetaOmega, E=E)
    # draw.draw(o, x=x, theta=thetaOmega, E=E)
    # plt.show()

    for k in range(K):
      R_omega_k = omegaKarcher[k,:o.dy,:o.dy]
      mu, Sigma = SED.sampleOmegaTranslation(o, theta[:,k], S[k], R_omega_k, lam)
      d_omega_k = mvn.rvs(mu, Sigma)
      omega[k] = SED.MakeRd(R_omega_k, d_omega_k)
      # omega[k] = SED.MakeRd(R_omega_k, mu)

    thetaOmega = np.tile(omega, (T, 1, 1, 1))
    # thetaOmega = np.tile(omegaKarcher, (T, 1, 1, 1))

    # draw.draw(o, y=y, x=x, theta=thetaOmega, E=E)
    # plt.show()

    for t in range(1,T-1):
      for k in range(K):
        R_cur = m.Rt(theta[t,k])[0]
        mu, Sigma = SED.sampleThetaTranslation(o, R_cur, theta[t-1,k], S[k], omega[k], lam,
          thetaNext=theta[t+1,k])


    # t, k = (5, 4)
    # R_cur = m.Rt(theta[t,k])[0]
    # mu, Sigma = SED.sampleThetaTranslation(o, R_cur, theta[t-1,k], S[k], omega[k], lam,
    #   thetaNext=theta[t+1,k])

    ip.embed()


  def testFeedbackMCMC(s):
    # sample = 'tmp/initialization_output/se2_waving_hand/se2_waving_hand'
    # sample = 'tmp/initialization_output/se2_spider/se2_spider'
    sample = 'tmp/initialization_output_omega/se2_waving_hand/se2_waving_hand'
    o, alpha, z, pi, theta, E, S, x, Q, omegaOld, mL, ll, subsetIdx, dataset = \
      SED.loadSample(sample)

    yAll = du.load(f'{dataset}/data')['y']
    if subsetIdx is not None:
      y = [yt[subsetIdx[t]] for t, yt in enumerate(yAll)]
    else:
      y = yAll

    T = len(y)
    K = E.shape[0]
    m = getattr(lie, o.lie)
    lam = 0.3

    # get back to old parts
    for t in range(T):
      for k in range(K):
        theta[t,k] = omegaOld[k] @ theta[t,k]

    W_k = iw.rvs(10, np.diag([10000., 10000., 5.0]))
    zero = np.zeros(o.dxA)

    def interp(a, b, lam):
      inner = lam * m.logm(m.inv(a) @ b)
      return a @ m.expm(inner)

    def objective(k, w):
      ll = mvn.logpdf(w, zero, W_k)
      omega = m.expm(m.alg(w))
      for t in range(1,T):
        mu = interp(theta[t-1,k], omega, lam)
        ll += SED.mvnL_logpdf(o, theta[t,k], mu, S[k])
      return ll

    proposalVariance = np.diag([1,1,0.01])
    def omega_mcmc(nSamples, k):
      samples = np.zeros((nSamples, o.dxA))
      ll = np.zeros(nSamples)
      ll[0] = objective(k, samples[0])
      accepts = np.zeros(nSamples, dtype=np.bool)
      accepts[0] = 1
      
      for nS in range(1, nSamples):
        gamma = mvn.rvs(zero, proposalVariance)
        omegaProp = samples[nS-1] + gamma
        llProp = objective(k, omegaProp)
        logA = llProp - ll[nS-1]
        if logA >= 0 or np.random.rand() < np.exp(logA):
          samples[nS] = omegaProp
          ll[nS] = llProp
          accepts[nS] = True
        else:
          samples[nS] = samples[nS-1]
          ll[nS] = ll[nS-1]

      return m.expm(m.alg(samples[-1])), samples, ll, accepts

    omega = np.zeros((K,) + o.dxGm)

    omegaKarcher = np.stack([ m.karcher(theta[:,k]) for k in range(K) ])
    llKarcher = np.array([ objective(k, m.algi(m.logm(omegaKarcher[k]))) for k in range(K) ])
  
    nSamples = 1000
    # for k in range(4,5):
    for k in range(K):
      omega[k], samples, ll, accepts = omega_mcmc(nSamples, k)

      plt.plot(ll, color='b', label='mcmc')
      plt.axhline(llKarcher[k], color='r', label='karcher')
      plt.title(f'nAccept: {(np.sum(accepts) / nSamples):.2f}')
      plt.show()
    
    # ip.embed()

    t = 5
    draw.draw_t(o, y=y[t], x=x[t], theta=omega, z=z[t], E=E, reverseY=True)
    plt.show()

    # # do mcmc on each part, init with karcher mean
    # omega = np.stack([ m.karcher(theta[:,k]) for k in range(K) ])
    #
    # proposalVariance = np.diag([1,1,0.1])
    # nSamples = 100
    # for k in range(K):
    # # for k in range(1):
    #   omegaPrev = m.algi(m.logm(omega[k]))
    #   llPrev = objective(k, omegaPrev)
    #   accept = 0
    #   for nS in range(nSamples):
    #     gamma = mvn.rvs(zero, proposalVariance)
    #     omegaProp = omegaPrev + gamma
    #     llProp = objective(k, omegaProp)
    #     logA = llProp - llPrev
    #     if logA >= 0 or np.random.rand() < np.exp(logA):
    #       # accept
    #       omegaPrev, llPrev = ( omegaProp, llProp )
    #       accept += 1
    #   print(k, accept)
    #   omega[k] = m.expm(m.alg(omegaPrev))
    #
    #
    # t = 5
    # draw.draw_t(o, y=y[t], x=x[t], theta=omega, z=z[t], E=E, reverseY=True)
    # plt.show()
    #
    # # colors = du.diffcolors(100, bgCols=[[1,1,1],[0,0,0]])
    # # plt.scatter(, s=20, color=colors[:K])
    # ip.embed()


  def testTranslationRotationConditionals(s):
    sample = 'tmp/initialization_output/se2_waving_hand/se2_waving_hand'
    # sample = 'tmp/initialization_output/se2_spider/se2_spider'
    o, alpha, z, pi, theta, E, S, x, Q, mL, ll, subsetIdx, dataset = \
      SED.loadSample(sample)
    yAll = du.load(f'{dataset}/data')['y']
    if subsetIdx is not None:
      y = [yt[subsetIdx[t]] for t, yt in enumerate(yAll)]
    else:
      y = yAll

    T = len(y)
    K = E.shape[0]
    m = getattr(lie, o.lie)

    # sample R_theta[t,k] under old model
    for t in range(T):
      for k in range(K):
        # sample theta_tk
        if t==0: thetaPrev, SPrev = ( o.H_theta[1], o.H_theta[2] )
        else: thetaPrev, SPrev = ( theta[t-1,k], S[k] )
        if t==T-1: thetaNext, SNext = ( None, None )
        else: thetaNext, SNext = ( theta[t+1,k], S[k] )
        thetaPredict = theta[t,k]
        y_tk = y[t][z[t]==k]

        # sample rotation
        d_theta_tk = m.Rt(theta[t,k])[1]
        ys = y_tk
        d_U = d_theta_tk
        lhs = x[t]
        rhs = np.eye(o.dy+1)
        prevU = thetaPrev
        prevS = SPrev
        nextU = thetaNext
        nextS = SNext

        R_U = SED.sampleRotation(o, ys, d_U, lhs, rhs, E[k], prevU, prevS,
          nextU=nextU, nextS=nextS)

        # sample translation
        mu, Sigma = SED.translationDynamicsPosterior(o, R_U, prevU, prevS,
          nextU=nextU, nextS=nextS)
        mu, Sigma = SED.translationObservationPosterior(o, ys, R_U, lhs, rhs,
          E[k], mu, Sigma)
        d_U = mvn.rvs(mu, Sigma)

        theta[t,k] = SED.MakeRd(R_U, d_U)




    draw.draw(o, y=y, x=x, theta=theta, E=E)
    plt.show()


    
  def testTranslationOmega(s):
    sample = 'tmp/initialization_output/se2_waving_hand/se2_waving_hand'
    # sample = 'tmp/initialization_output/se2_spider/se2_spider'
    o, alpha, z, pi, theta, E, S, x, Q, mL, ll, subsetIdx, dataset = \
      SED.loadSample(sample)
    yAll = du.load(f'{dataset}/data')['y']
    if subsetIdx is not None:
      y = [yt[subsetIdx[t]] for t, yt in enumerate(yAll)]
    else:
      y = yAll

    T = len(y)
    K = E.shape[0]
    m = getattr(lie, o.lie)

    # omega_k = np.eye(3)
    # W_k = iw.rvs(10, np.diag([10000., 10000., 5.0]))
    # d_omega = np.zeros((K, o.dy))
    # for k in range(K):
    #   y_k = [ y[t][z[t]==k] for t in range(T) ]
    #   d_omega[k] = SED.sampleTranslationOmega(o, y_k, x, omega_k, theta[:,k], E[k], W_k)
    #
    # t = 0
    # yObj_t = SED.TransformPointsNonHomog(m.inv(x[t]), y[t])
    # draw.draw_t(o, y=yObj_t, x=np.eye(3), theta=theta[t], z=z[t], reverseY=True)
    # colors = du.diffcolors(100, bgCols=[[1,1,1],[0,0,0]])
    # plt.scatter(d_omega[:,0], d_omega[:,1], s=20, color=colors[:K])
    # plt.show()
    # ip.embed()

    omega = np.stack([ m.karcher(theta[:,k]) for k in range(K) ])
    for t in range(T):
      for k in range(K):
        theta[t,k] = m.inv(omega[k]) @ theta[t,k]

    omega_k = np.eye(3)
    W_k = iw.rvs(10, np.diag([10000., 10000., 5.0]))
    d_omega = np.zeros((K, o.dy))
    for k in range(K):
      y_k = [ y[t][z[t]==k] for t in range(T) ]
      d_omega[k] = SED.sampleTranslationOmega(o, y_k, x, omega_k, theta[:,k], E[k], W_k)
      omega[k,:-1, -1] = d_omega[k]

    # resample theta translation
    thetaOld = theta.copy()

    for t in range(T):
      for k in range(K):
        # sample theta_tk
        if t==0: thetaPrev, SPrev = ( o.H_theta[1], o.H_theta[2] )
        else: thetaPrev, SPrev = ( theta[t-1,k], S[k] )
        if t==T-1: thetaNext, SNext = ( None, None )
        else: thetaNext, SNext = ( theta[t+1,k], S[k] )
        thetaPredict = theta[t,k]
        y_tk = y[t][z[t]==k]

        thetaPredict[:-1,-1] = SED.sampleTranslationTheta_with_omega(o, y_tk,
          thetaPredict, x[t], SPrev, E[k], thetaPrev, omega[k],
          theta_tplus1_k=thetaNext, S_tplus1_k=SNext)
        theta[t,k] = thetaPredict
    # done resampling theta

    parts = np.zeros_like(theta)
    for t in range(T):
      for k in range(K): 
        parts[t,k] = omega[k] @ theta[t,k]
    
    t = 0
    # yObj_t = SED.TransformPointsNonHomog(m.inv(x[t]), y[t])
    # draw.draw_t(o, y=yObj_t, x=x[t], theta=parts[t], reverseY=True)
    # draw.draw_t(o, y=y[t], x=x[t], theta=parts[t], E=E, reverseY=True)
    draw.draw(o, y=y, x=x, theta=parts, E=E)
    colors = du.diffcolors(100, bgCols=[[1,1,1],[0,0,0]])
    # plt.scatter(d_omega[:,0], d_omega[:,1], s=20, color=colors[:K])
    plt.show()


  # def testPartsPosterior2(s):
  #   sample = 'tmp/initialization_output/se2_waving_hand/se2_waving_hand'
  #   # sample = 'tmp/initialization_output/se2_spider/se2_spider'
  #   # sample = 'tmp/initialization_output/se2_marmoset/tInit/k3/se2_marmoset_k3.gz'
  #   o, alpha, z, pi, theta, E, S, x, Q, mL, ll, subsetIdx, dataset = \
  #     SED.loadSample(sample)
  #   y = du.load(f'{dataset}/data')['y']

  def test_partsPosterior(s):
    # sample = 'tmp/initialization_output/se2_waving_hand/se2_waving_hand'
    sample = 'tmp/initialization_output/se2_spider/se2_spider'
    # sample = 'tmp/initialization_output/se2_marmoset/tInit/k3/se2_marmoset_k3.gz'
    o, alpha, z, pi, theta, E, S, x, Q, mL, ll, subsetIdx, dataset = \
      SED.loadSample(sample)
    yAll = du.load(f'{dataset}/data')['y']
    if subsetIdx is not None:
      y = [yt[subsetIdx[t]] for t, yt in enumerate(yAll)]
    else:
      y = yAll
    m = getattr(lie, o.lie)
    T, K = theta.shape[:2]

    # initialize omega with karcher mean
    omega = np.stack([ m.karcher(theta[:,k]) for k in range(K) ])
    for t in range(T):
      for k in range(K):
        theta[t,k] = m.inv(omega[k]) @ theta[t,k]

    # R_omega_prior = np.tile(np.eye(o.dy), (K, 1, 1))
    # omega = SED.sampleOmega(o, y, z, x, theta, E, R_omega_prior)

    I = np.eye(o.dy+1)

    nSamples = 3
    for nS in range(nSamples):
      print(nS)

      # sample theta
      for t in range(T):
        for k in range(K):
          # sample theta_tk
          if t==0: thetaPrev, SPrev = ( o.H_theta[1], o.H_theta[2] )
          else: thetaPrev, SPrev = ( theta[t-1,k], S[k] )
          if t==T-1: thetaNext, SNext = ( None, None )
          else: thetaNext, SNext = ( theta[t+1,k], S[k] )

          y_tk = y[t][z[t]==k]
          R_theta_tk = m.Rt(theta[t,k])[0]
          lhs, rhs = ( x[t] @ omega[k], I )

          # sample translation | rotation
          mu, Sigma = SED.translationDynamicsPosterior(
            o, R_theta_tk, thetaPrev, SPrev, nextU=thetaNext, nextS=SNext)
          mu, Sigma = SED.translationObservationPosterior(o, y_tk, R_theta_tk,
            lhs, rhs, E[k], mu, Sigma)
          d_theta_tk = mvn.rvs(mu, Sigma)

          # sample rotation | translation
          R_theta_tk = SED.sampleRotation(o, y_tk, d_theta_tk, lhs, rhs, E[k],
            thetaPrev, SPrev, nextU=thetaNext, nextS=SNext)
          
          # set new sample
          theta[t,k] = SED.MakeRd(R_theta_tk, d_theta_tk)

      # sample omega
      R_omega_prior = omega[:,:o.dy,:o.dy]     
      omega = SED.sampleOmega(o, y, z, x, theta, E, R_omega_prior)

      print(theta)


    drawTheta = np.zeros_like(theta)
    for t in range(T):
      for k in range(K):
        drawTheta[t,k] = omega[k] @ theta[t,k]

    draw.draw(o, y=y, x=x, theta=drawTheta, z=z, E=E)
    plt.show()

    ip.embed()

    # t = 5
    # draw.draw_t(o, y=y[t], x=x[t], theta=omega, z=z[t], E=E, reverseY=True)
    # draw.draw_t(o, y=y[t], x=x[t], theta=drawTheta_t, z=z[t], E=E, reverseY=True)

    # colors = du.diffcolors(100, bgCols=[[1,1,1],[0,0,0]])
    # plt.scatter(, s=20, color=colors[:K])
    # ip.embed()


    # omega = np.tile(np.eye(o.dy+1), (K, 1, 1))
    # # iterate between sampling parts and sampling omega
    # nSamples = 10
    # for nS in range(nSamples):
    # SED.partPosteriors(o, y, x, z, theta, E)
    
  def test_parts_icp(s):

    useHand = True
    if useHand:
      alpha = 0.1
      mL_const = -14.0

      y = du.load('../npp-data/se2_waving_hand/data')['y']
      # y = du.load('../npp-data/se2_spider/data')['y']
      # y = y[:10]
      y = y[:4]
      T = len(y)
      maxObs = 3000
      if maxObs > 0:
        y = [ yt[np.random.choice(range(len(yt)), min(maxObs, len(yt)), replace=False)] for yt in y ]

      mL = [ mL_const * np.ones(y[t].shape[0]) for t in range(T) ]
      o = SED.opts(lie='se2')
      SED.initPriorsDataDependent(o, y, scaleE=1.0, dfE=10, rotQ=25.0)
      x = SED.initXDataMeans(o, y)
      Q = SED.inferQ(o, x)
      theta, E, S, z, pi = SED.initPartsAndAssoc(o, y[:1], x, alpha, mL,
        maxIter=500, nInit=5)
      # S[:,:o.dy,:o.dy] *= 3
      # E /= 2.0

    else:
      o = s.o
      y = s.y
      theta = s.thetaTrue
      E = s.ETrue
      S = s.STrue

    m = getattr(lie, o.lie)
    t1, t2 = (0, 3)
    K = E.shape[0]
    yPrev = y[t1]
    yNext = y[t2]
    xPrev = x[t1]
    thetaPrev = theta[t1]

    muDiff = np.mean(yNext, axis=0) - np.mean(yPrev, axis=0)
    Q_t0 = SED.MakeRd(np.eye(o.dy), muDiff)
    q_t0 = m.algi(m.logm(Q_t0))

    Q_t = icp.optimize_global(o, yNext, xPrev, thetaPrev, E, q_t=q_t0)
    xNext = xPrev @ Q_t

    S_t = icp.optimize_local(o, yNext, xNext, thetaPrev, E, S)
    thetaNext = np.stack([ thetaPrev[k] @ S_t[k] for k in range(K) ])
    
    # first try method
    # Q_t = icp.register(o, yPrev, yNext, xPrev, thetaPrev, E, plot=False)

    # original global, local estimate
    xNext0 = xPrev @ Q_t0
    draw.draw_t(o, y=yNext, x=xNext0, theta=thetaPrev, E=E, reverseY=True)
    plt.title('Initial')

    # new global estimate
    plt.figure()
    xNext = xPrev @ Q_t
    draw.draw_t(o, y=yNext, x=xNext, theta=thetaPrev, E=E, reverseY=True)
    plt.title('Updated Global')

    # new global and local estimate
    plt.figure()
    draw.draw_t(o, y=yNext, x=xNext, theta=thetaNext, E=E, reverseY=True)
    plt.title('Updated Global and Local')

    plt.show()



  def testInitRJMCMC_hand(s):
    alpha = 0.1
    mL_const = -14.0

    o = SED.opts(lie='se2')
    y = du.load('../npp-data/se2_waving_hand/data')['y']
    y = y[:4]
    T = len(y)
    maxObs = 3000
    if maxObs > 0:
      y = [ yt[np.random.choice(range(len(yt)), min(maxObs, len(yt)), replace=False)] for yt in y ]

    # SED.initPriorsDataDependent(o, y, scaleE=0.01, dfE=200)
    SED.initPriorsDataDependent(o, y, scaleE=1.0, dfE=10, rotQ=25.0)
    x = SED.initXDataMeans(o, y)
    Q = SED.inferQ(o, x)

    mL = [ mL_const * np.ones(y[t].shape[0]) for t in range(T) ]
    theta, E, S, z, pi = SED.initPartsAndAssoc(o, y, x, alpha, mL)
    E /= 10.

    # theta = np.zeros((T, 0,) + o.dxGm)
    # E = np.zeros((0, o.dy, o.dy))
    # S = np.zeros((0, o.dxA, o.dxA))
    # Q = SED.inferQ(o, x)
    # mL = [ mL_const * np.ones(y[t].shape[0]) for t in range(T) ]
    # pi = np.array([1.0,])
    # z = [ -1 * np.ones(y[t].shape[0], dtype=np.int) for t in range(T) ]

    llInit = SED.logJoint(o, y, z, x, theta, E, S, Q, alpha, pi, mL)
    print(f'llInit: {llInit:.2f}')

    # pBirth, pDeath = (0.1, 0.3)
    pBirth, pDeath, pSwitch = (0.0, 0.0, 0.9)

    nSamples = 10000
    ll = np.zeros(nSamples+1)
    ll[0] = llInit
    nBirthProp, nBirthAccept, nDeathProp, nDeathAccept = (0, 0, 0, 0)
    nSwitchProp, nSwitchAccept = (0, 0)

    for nS in range(nSamples):
      dontSampleX = True if nS < 0 else False
      if nS % 2 == 0: pSwitch = 0.99
      else: pSwitch = 0.0

      z, pi, theta, E, S, x, Q, mL, move, accept = SED.sampleRJMCMC( o, y,
        alpha, z, pi, theta, E, S, x, Q, mL, pBirth, pDeath, pSwitch,
        dontSampleX=dontSampleX)
      ll[nS+1] = SED.logJoint(o, y, z, x, theta, E, S, Q, alpha, pi, mL)

      if move == 'birth':
        nBirthProp += 1
        if accept: nBirthAccept += 1
      elif move == 'death':
        nDeathProp += 1
        if accept: nDeathAccept += 1
      elif move == 'switch':
        nSwitchProp += 1
        if accept: nSwitchAccept += 1

      a = '+' if accept == True else '-'
      print(f'Iter {nS:05}, LL: {ll[nS+1]:.2f}, K: {len(pi)-1}, Move: {move[0]}{a}')

      # if nS % 10 == 0:
      filename = f'tmp/rjmcmc_hand_switch/sample-{nS:05}'
      SED.saveSample(filename, o, alpha, z, pi, theta, E, S, x, Q, ll=ll[nS+1])

    plt.figure()
    plt.plot(ll, c='b', label='Sample LL')
    plt.legend()
    plt.title(f'nBirthProp: {nBirthProp}, nBirthAccept: {nBirthAccept}, nDeathProp: {nDeathProp}, nDeathAccept: {nDeathAccept}')
    plt.show()

    from npp import drawSED
    drawSED.draw(o, y=y, x=x, z=z, theta=theta, E=E)
    plt.show()


  def testInitRJMCMC(s):
    # run with 
    # nosetests -s --nologcapture test/testInference.py:test_se2_randomwalk3.testInitRJMCMC
    alpha = 0.1
    # mL_const = -7.0
    mL_const = -10.0

    o = s.o
    SED.initPriorsDataDependent(o, s.y)

    ## 0-part init
    # x = SED.initXDataMeans(o, s.y)
    # theta = np.zeros((s.T, 0,) + o.dxGm)
    # E = np.zeros((0, o.dy, o.dy))
    # S = np.zeros((0, o.dxA, o.dxA))
    # Q = SED.inferQ(o, x)
    # mL = [ mL_const * np.ones(s.y[t].shape[0]) for t in range(s.T) ]
    # pi = np.array([1.0,])
    # z = [ -1 * np.ones(s.y[t].shape[0], dtype=np.int) for t in range(s.T) ]
    ## end 0-part init

    # nonparametric init
    y, T = (s.y, s.T)

    SED.initPriorsDataDependent(o, y)
    x = SED.initXDataMeans(o, y)
    Q = SED.inferQ(o, x)

    mL = [ mL_const * np.ones(y[t].shape[0]) for t in range(T) ]
    theta, E, S, z, pi = SED.initPartsAndAssoc(o, y, x, alpha, mL)
    # end init

    llTrue = SED.logJoint(o, s.y, s.zTrue, s.xTrue, s.thetaTrue, s.ETrue,
      s.STrue, s.QTrue, alpha, s.piTrue, mL)
    llInit = SED.logJoint(o, s.y, z, x, theta, E, S, Q, alpha, pi, mL)
    print(f'llTrue: {llTrue:.2f}, llInit: {llInit:.2f}')

    # z, pi, theta, E, S, x, Q, ll = SED.sampleStepFC(o, s.y, alpha, z, pi,
    #   theta, E, S, x, Q, mL, newPart=False)

    pBirth, pDeath, pSwitch = (0.3, 0.1, 0.3)
    nSamples = 10000
    ll = np.zeros(nSamples+1)
    ll[0] = llInit
    nBirthProp, nBirthAccept, nDeathProp, nDeathAccept = (0, 0, 0, 0)
    nSwitchProp, nSwitchAccept = (0, 0)

    for nS in range(nSamples):
      dontSampleX = True if nS < 100 else False
      z, pi, theta, E, S, x, Q, mL, move, accept = SED.sampleRJMCMC( o, s.y,
        alpha, z, pi, theta, E, S, x, Q, mL, pBirth, pDeath, pSwitch,
        dontSampleX=dontSampleX)
      ll[nS+1] = SED.logJoint(o, s.y, z, x, theta, E, S, Q, alpha, pi, mL)

      if move == 'birth':
        nBirthProp += 1
        if accept: nBirthAccept += 1
      elif move == 'death':
        nDeathProp += 1
        if accept: nDeathAccept += 1
      elif move == 'switch':
        nSwitchProp += 1
        if accept: nSwitchAccept += 1

      a = '+' if accept == True else '-'
      print(f'Iter {nS:05}, LL: {ll[nS+1]:.2f}, K: {len(pi)-1}, Move: {move[0]}{a}')

      if nS % 100 == 0:
        filename = f'tmp/rjmcmc_switch/sample-{nS:05}'
        SED.saveSample(filename, o, alpha, z, pi, theta, E, S, x, Q, ll=ll[nS+1])

    plt.figure()
    plt.plot(ll, c='b', label='Sample LL')
    plt.axhline(llTrue, c='r', label='True LL')
    plt.legend()
    plt.title(f'nBirthProp: {nBirthProp}, nBirthAccept: {nBirthAccept}, nDeathProp: {nDeathProp}, nDeathAccept: {nDeathAccept}')
    plt.show()

    from npp import drawSED
    drawSED.draw(o, y=s.y, x=x, z=z, theta=theta, E=E)
    plt.show()

    # # # artifically add a part
    # theta = np.concatenate((theta, np.zeros((s.T,1,) + o.dxGm)), axis=1)
    # E = np.concatenate((E, np.zeros((o.dy, o.dy))[np.newaxis]), axis=0)
    # S = np.concatenate((S, np.zeros((1, o.dxA, o.dxA))), axis=0)
    # pi = np.array([0.5, 0.5])
    # z, pi, theta, E, S = SED.consolidatePartsAndResamplePi(o, z, pi, alpha, theta, E, S)
    # assert theta.shape[1] == 0 and E.shape[0] == 0 and S.shape[0] == 0
    #

    
    # for i in range(100):
    #   accept, z_, x, theta_, E_, S_, Q, alpha, pi_, mL = SED.try_birth(o, s.y,
    #     z, x, theta, E, S, Q, alpha, pi, mL, pBirth, pDeath)
    #   if accept:
    #     print(f'Accept at step {i}')
    #     break


    # import matplotlib.pyplot as plt
    # from npp import drawSED
    # drawSED.draw(o, y=s.y, x=x, z=z)
    # plt.show()
     


    # # nSamples = 50
    # nSamples = 5
    # ll = np.zeros(nSamples)
    # for nS in range(nSamples):
    #   z, pi, theta, E, S, x, Q, ll[nS] = SED.sampleStepFC(o, s.y, alpha, z, pi,
    #     theta, E, S, x, Q, mL, newPart=False)
    #   Nk = np.sum([SED.getComponentCounts(o, z[t], pi)
    #     for t in range(s.T)], axis=0)
    #   print(nS, Nk[-1])
    #
    #
    # import matplotlib.pyplot as plt
    # from npp import drawSED
    # plt.plot(ll, label='Samples', c='b');
    # piTrue = np.concatenate((s.piTrue, np.array([alpha,])))
    # llTrue = SED.logJoint(o, s.y, s.zTrue, s.xTrue, s.thetaTrue, s.ETrue,
    #   s.STrue, s.QTrue, alpha, s.piTrue, mL)
    # plt.axhline(llTrue, label='True', c='g')
    # plt.legend()
    # plt.show()
    # drawSED.draw(o, y=s.y, x=x, z=z, theta=theta, E=E)
    # plt.show()

    ## quick test of save and load
    # filename = 'tmp-123'
    # SED.saveSample(filename, o, alpha, z, pi, theta, E, S, x, Q, ll=np.nan)
    # o, alpha, z, pi, theta, E, S, x, Q, ll = SED.loadSample(filename)
    #
    # ip.embed()

  def testInit(s):
    alpha = 0.1
    nParticles = 100

    o = s.o
    SED.initPriorsDataDependent(o, s.y)
    x = SED.initXDataMeans(o, s.y)
    theta_, E_, S_ = SED.sampleKPartsFromPrior(o, s.T, nParticles)
    mL = SED.logMarginalPartLikelihoodMonteCarlo(o, s.y, x, theta_, E_, S_)
    theta, E, S, z, pi = SED.initPartsAndAssoc(o, s.y, x, alpha, mL)
    # theta, E, S, z, pi = SED.initPartsAndAssoc(o, s.y, x, alpha, mL,
    #   fixedK=True, maxBreaks=2)
    Q = SED.inferQ(o, x)

    # nSamples = 50
    nSamples = 5
    ll = np.zeros(nSamples)
    for nS in range(nSamples):
      z, pi, theta, E, S, x, Q, ll[nS] = SED.sampleStepFC(o, s.y, alpha, z, pi,
        theta, E, S, x, Q, mL)
      Nk = np.sum([SED.getComponentCounts(o, z[t], pi)
        for t in range(s.T)], axis=0)
      print(nS, Nk[-1])


    import matplotlib.pyplot as plt
    from npp import drawSED
    plt.plot(ll, label='Samples', c='b');
    piTrue = np.concatenate((s.piTrue, np.array([alpha,])))
    llTrue = SED.logJoint(o, s.y, s.zTrue, s.xTrue, s.thetaTrue, s.ETrue,
      s.STrue, s.QTrue, alpha, s.piTrue, mL)
    plt.axhline(llTrue, label='True', c='g')
    plt.legend()
    plt.show()
    drawSED.draw(o, y=s.y, x=x, z=z, theta=theta, E=E)
    plt.show()

    ## quick test of save and load
    # filename = 'tmp-123'
    # SED.saveSample(filename, o, alpha, z, pi, theta, E, S, x, Q, ll=np.nan)
    # o, alpha, z, pi, theta, E, S, x, Q, ll = SED.loadSample(filename)
    #
    # ip.embed()


  def testNoParts(s):
    alpha = 0.1
    nParticles = 100

    o = s.o
    SED.initPriorsDataDependent(o, s.y)
    x = SED.initXDataMeans(o, s.y)
    theta_, E_, S_ = SED.sampleKPartsFromPrior(o, s.T, nParticles)

    # force everything to be associated to base measure
    mL = [ 1000 * np.ones(s.y[t].shape[0]) for t in range(s.T) ]

    theta, E, S, z, pi = SED.initPartsAndAssoc(o, s.y, x, alpha, mL)
    Q = SED.inferQ(o, x)

    nSamples = 1
    ll = np.zeros(nSamples)
    for nS in range(nSamples):
      z, pi, theta, E, S, x, Q, ll[nS] = SED.sampleStepFC(o, s.y, alpha, z, pi,
        theta, E, S, x, Q, mL, newPart=False)
      assert len(pi) == 1
      assert theta.shape[1] == 0
      assert E.shape[0] == 0
      assert S.shape[0] == 0

    assert np.all(~np.isinf(ll))
    assert np.all(~np.isnan(ll))
    
    # force everything to be associated to new part
    mL = [ -1000 * np.ones(s.y[t].shape[0]) for t in range(s.T) ]

    for nS in range(nSamples):
      z, pi, theta, E, S, x, Q, ll[nS] = SED.sampleStepFC(o, s.y, alpha, z, pi,
        theta, E, S, x, Q, mL, newPart=True)
      assert len(pi) == 2
      assert theta.shape[1] == 1
      assert E.shape[0] == 1
      assert S.shape[0] == 1

    assert np.all(~np.isinf(ll))
    assert np.all(~np.isnan(ll))

class test_drawing(unittest.TestCase):
  def setUp(s):
    s.o = SED.opts(lie='se2')

  def test_se2(s):
    import matplotlib.pyplot as plt

    dataset = 'se2_waving_hand'
    alpha = 0.1
    nParticles = 1

    o = s.o
    y = du.load(f'../npp-data/{dataset}/data')['y']
    maxObs = 3000
    y = [ yt[np.random.choice(range(len(yt)), maxObs, replace=False)] for yt in y ]
    T = len(y)

    SED.initPriorsDataDependent(o, y)
    x = SED.initXDataMeans(o, y)
    theta_, E_, S_ = SED.sampleKPartsFromPrior(o, T, nParticles)

    # force everything to not be associated to base measure
    # mL = [ 1000 * np.ones(y[t].shape[0]) for t in range(T) ]
    mL = [ -1000 * np.ones(y[t].shape[0]) for t in range(T) ]

    theta, E, S, z, pi = SED.initPartsAndAssoc(o, y, x, alpha, mL)
    Q = SED.inferQ(o, x)

    img = du.For(du.imread, du.GetImgPaths(f'../npp-data/{dataset}/imgs'))
    # img = None
    draw.draw(o, y=y, z=z, img=img, x=x, theta=theta, E=E)
    plt.show()

    # t = 0
    # img = du.imread(imgs[t])
    # draw.draw_t_SE2(o, y=y[t], z=z[t], img=img, x=x[t], theta=theta[t], E=E)
    # plt.show()

class test_eval(unittest.TestCase):
  def setUp(s):
    s.o = SED.opts(lie='se2')

  def test_se2(s):
    import matplotlib.pyplot as plt

    dataset = 'se2_waving_hand'
    alpha = 0.1
    nParticles = 1

    o = s.o
    y = du.load(f'../npp-data/{dataset}/data')['y']
    gt = f'../npp-data/{dataset}/gtLabels'

    # maxObs = 3000
    # y = [ yt[np.random.choice(range(len(yt)), maxObs, replace=False)] for yt in y ]
    T = len(y)

    SED.initPriorsDataDependent(o, y)
    x = SED.initXDataMeans(o, y)
    theta_, E_, S_ = SED.sampleKPartsFromPrior(o, T, nParticles)

    # force everything to not be associated to base measure
    # mL = [ 1000 * np.ones(y[t].shape[0]) for t in range(T) ]
    mL = [ -1000 * np.ones(y[t].shape[0]) for t in range(T) ]

    theta, E, S, z, pi = SED.initPartsAndAssoc(o, y, x, alpha, mL)
    Q = SED.inferQ(o, x)

    img = du.For(du.imread, du.GetImgPaths(f'../npp-data/{dataset}/imgs'))
    labels = evalSED.MakeLabelImgs(y, z, img)

    tp, fp, fn, ids, tilde_tp, motsa, motsp, s_motsa = evalSED.mots(labels, gt)
    print(s_motsa)

class test_reparam(unittest.TestCase):
  def test1(s):
    # general parameters
    o = SED.opts(lie='se2')
    m = getattr(lie, o.lie)
    T, K = ( 500, 4 )
    
    # body frame
    Q = np.diag((1.0, 2.0, .05))
    x = np.zeros((T,) + o.dxGm)
    x[0] = np.eye(o.dxA)
    q = mvn.rvs(np.zeros(o.dxA), Q, size=T-1)
    for t in range(1, T):
      # qt = mvn.rvs(np.zeros(o.dxA), Q)
      x[t] = x[t-1] @ m.expm(m.alg(q[t-1]))

    # parts
    S = np.diag((0.01, 0.02, .01))
    theta = np.zeros((T, K) + o.dxGm)
    s1 = np.array([[0, 12, 0], [0, -12, 0], [12, 0, 0], [-12, 0, 0]])
    for k in range(K): theta[0,k] = m.expm(m.alg(s1[k]))
    for t in range(1,T):
      for k in range(K):
        stk = mvn.rvs(np.zeros(o.dxA), S)
        # stk = np.zeros(o.dxA)
        # theta[t,k] = m.inv(x[t]) @ x[t-1] @ theta[t-1,k] @ m.expm(m.alg(stk))

        theta[t,k] = theta[t-1,k] @ m.expm(m.alg(stk))
        # theta[t,k] = m.expm(-m.alg(q[t-1])) @ theta[t-1,k] @ m.expm(m.alg(stk))
    
    colors = du.diffcolors(K, bgCols=[[0,0,0],[1,1,1]])
    def show(t):
      # plot x
      T_world_object = x[t]
      m.plot(T_world_object, np.tile([0,0,0], [2,1]), l=10.0)

      # plot theta[t,k]
      for k in range(K):
        T_world_part = x[t] @ theta[t,k]
        m.plot(T_world_part, np.tile(colors[k], [2,1]), l=10.0)

      plt.xlim(-50, 50)
      plt.ylim(-50, 50)

    path = 'reparam/original'
    for t in range(T):
      show(t)
      plt.savefig(f'{path}/img-{t:05}.png', dpi=300, bbox_inches='tight')
      plt.close()

    # du.ViewPlots(range(T), show)
    # plt.show()

  def test2(s):
    # general parameters
    o = SED.opts(lie='se2')
    m = getattr(lie, o.lie)
    T, K = ( 500, 4 )
    
    # body frame
    Q = np.diag((1.0, 2.0, .05))
    x = np.zeros((T,) + o.dxGm)
    x[0] = np.eye(o.dxA)
    q = mvn.rvs(np.zeros(o.dxA), Q, size=T-1)
    for t in range(1, T):
      # qt = mvn.rvs(np.zeros(o.dxA), Q)
      x[t] = x[t-1] @ m.expm(m.alg(q[t-1]))

    # parts
    S = np.diag((0.01, 0.02, .01))
    theta = np.zeros((T, K) + o.dxGm)

    s1 = np.array([[0, 12, 0], [0, -12, 0], [12, 0, 0], [-12, 0, 0]])
    for k in range(K): theta[0,k] = m.expm(m.alg(s1[k]))
    omega = theta[0].copy()
    AdjOmega = [ m.Adj(omega[k]) for k in range(K) ]
    print(AdjOmega)

    for t in range(1,T):
      for k in range(K):
        stk = mvn.rvs(np.zeros(o.dxA), S)
        Stk = m.expm( m.alg( AdjOmega[k] @ stk))

        theta[t,k] = theta[t-1,k] @ Stk
    
    colors = du.diffcolors(K, bgCols=[[0,0,0],[1,1,1]])
    def show(t):
      # plot x
      T_world_object = x[t]
      m.plot(T_world_object, np.tile([0,0,0], [2,1]), l=10.0)

      # plot theta[t,k]
      for k in range(K):
        T_world_part = x[t] @ theta[t,k]
        m.plot(T_world_part, np.tile(colors[k], [2,1]), l=10.0)

      plt.xlim(-50, 50)
      plt.ylim(-50, 50)

    # ip.embed()
    path = 'reparam/adjoint'
    for t in range(T):
      show(t)
      plt.savefig(f'{path}/img-{t:05}.png', dpi=300, bbox_inches='tight')
      plt.close()
  
    # du.ViewPlots(range(T), show)
    # plt.show()

  def test3(s):
    # general parameters
    o = SED.opts(lie='se2')
    m = getattr(lie, o.lie)
    T, K = ( 500, 4 )
    
    # body frame
    Q = np.diag((1.0, 2.0, .05))
    x = np.zeros((T,) + o.dxGm)
    x[0] = np.eye(o.dxA)
    q = mvn.rvs(np.zeros(o.dxA), Q, size=T-1)
    for t in range(1, T):
      # qt = mvn.rvs(np.zeros(o.dxA), Q)
      x[t] = x[t-1] @ m.expm(m.alg(q[t-1]))

    # parts
    S = np.diag((0.01, 0.02, .01))
    theta = np.zeros((T, K) + o.dxGm)

    s1 = np.array([[0, 12, 0], [0, -12, 0], [12, 0, 0], [-12, 0, 0]])
    for k in range(K): theta[0,k] = m.expm(m.alg(s1[k]))
    w = s1.copy()
    lam = 0.3

    for t in range(1,T):
      for k in range(K):
        stk = mvn.rvs(np.zeros(o.dxA), S)
        Stk = m.expm( m.alg( lam * stk + (1-lam)*w[k] ))

        theta[t,k] = theta[t-1,k] @ Stk
    
    colors = du.diffcolors(K, bgCols=[[0,0,0],[1,1,1]])
    def show(t):
      # plot x
      T_world_object = x[t]
      m.plot(T_world_object, np.tile([0,0,0], [2,1]), l=10.0)

      # plot theta[t,k]
      for k in range(K):
        T_world_part = x[t] @ theta[t,k]
        m.plot(T_world_part, np.tile(colors[k], [2,1]), l=10.0)

      plt.xlim(-50, 50)
      plt.ylim(-50, 50)

    # ip.embed()
    path = 'reparam/convex'
    for t in range(T):
      show(t)
      plt.savefig(f'{path}/img-{t:05}.png', dpi=300, bbox_inches='tight')
      plt.close()
  
    # du.ViewPlots(range(T), show)
    # plt.show()

  def test4(s):
    # general parameters
    o = SED.opts(lie='se2')
    m = getattr(lie, o.lie)
    T, K = ( 500, 4 )
    
    # body frame
    Q = np.diag((1.0, 2.0, .05))
    x = np.zeros((T,) + o.dxGm)
    x[0] = np.eye(o.dxA)
    q = mvn.rvs(np.zeros(o.dxA), Q, size=T-1)
    for t in range(1, T):
      # qt = mvn.rvs(np.zeros(o.dxA), Q)
      x[t] = x[t-1] @ m.expm(m.alg(q[t-1]))

    # parts
    S = np.diag((0.01, 0.02, .01))
    theta = np.zeros((T, K) + o.dxGm)

    s1 = np.array([[0, 12, 0], [0, -12, 0], [12, 0, 0], [-12, 0, 0]])
    for k in range(K): theta[0,k] = m.expm(m.alg(s1[k]))
    omega = theta[0].copy()
    lam = 0.3

    def interp(a, b, lam):
      inner = lam * m.logm(m.inv(a) @ b)
      return a @ m.expm(inner)

    for t in range(1,T):
      for k in range(K):
        stk = mvn.rvs(np.zeros(o.dxA), S)
        Stk = m.expm(m.alg(stk))
        
        theta[t,k] = interp(theta[t-1,k], omega[k], lam) @ Stk
    
    colors = du.diffcolors(K, bgCols=[[0,0,0],[1,1,1]])
    def show(t):
      # plot x
      T_world_object = x[t]
      m.plot(T_world_object, np.tile([0,0,0], [2,1]), l=10.0)

      # plot theta[t,k]
      for k in range(K):
        T_world_part = x[t] @ theta[t,k]
        m.plot(T_world_part, np.tile(colors[k], [2,1]), l=10.0)

      plt.xlim(-50, 50)
      plt.ylim(-50, 50)

    # ip.embed()
    path = 'reparam/interp'
    for t in range(T):
      show(t)
      plt.savefig(f'{path}/img-{t:05}.png', dpi=300, bbox_inches='tight')
      plt.close()
  
    # du.ViewPlots(range(T), show)
    # plt.show()

  def test5(s):
    # general parameters
    o = SED.opts(lie='se2')
    m = getattr(lie, o.lie)
    if o.lie == 'se2': mRot = lie.so2
    else: mRot = lie.so3

    # T, K = ( 10000, 4 )
    T, K = ( 10, 4 )
    
    # body frame
    Q = np.diag((1.0, 2.0, .05))
    x = np.zeros((T,) + o.dxGm)
    x[0] = np.eye(o.dxA)
    q = mvn.rvs(np.zeros(o.dxA), Q, size=T-1)
    for t in range(1, T):
      # qt = mvn.rvs(np.zeros(o.dxA), Q)
      x[t] = x[t-1] @ m.expm(m.alg(q[t-1]))

    # parts
    # Strans = [ iw.rvs(10, 50*np.eye(o.dy)) for k in range(K) ]
    # Srot = np.diag([0.05])

    S = iw.rvs(10, np.diag((50, 50, 0.5)), size=K)
    Strans = S[:,:o.dy,:o.dy]
    Srot = S[:,o.dy:,o.dy:]

    theta = np.zeros((T, K) + o.dxGm)

    w = np.array([[0, 12, 0], [0, -12, 0], [12, 0, 0], [-12, 0, 0]])
    omega = np.zeros((K,) + o.dxGm)
    for k in range(K): omega[k] = m.expm(m.alg(w[k]))
    for k in range(K): theta[0,k] = np.eye(o.dy+1)

    # theta[0] = omega.copy()
    # theta[0] = omega.copy()

    zeroObs = np.zeros(o.dy)
    zeroRot = np.zeros(o.dxA - o.dy)
    IdentityRot = np.eye(o.dy)

    alpha, beta = (0.95, 0.05)
    A = np.sqrt(np.diag(alpha * np.ones(o.dy)))
    B = np.sqrt(np.diag(beta * np.ones(o.dy)))
    Ai = np.linalg.inv(A)
    Bi = np.linalg.inv(B)


    for t in range(1,T):
      for k in range(K):
        d_prev = m.Rt(theta[t-1,k])[1]

        # random translation in lie group, serves the role of n[t]
        m_tk = mvn.rvs(zeroObs, Strans[k]) 

        # random rotation in lie algebra
        phi_tk = mvn.rvs(zeroRot, Srot[k]) 

        # rotation in lie group
        R_tk = mRot.expm(mRot.alg(phi_tk))

        # updated translation
        d_new = A @ d_prev + B @ m_tk

        # updated transformation
        S_tk = SED.MakeRd(R_tk, d_new)
        theta[t,k] = S_tk
 
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

          m = Bi @ (v - A @ d_prev)
          val = np.concatenate((m, phi))
          ll = mvn.logpdf(val, zeroAlgebra, S[k])

          # future
          phi_ = mRot.algi(mRot.logm(R_cur.T @ R_next))
          phi_ = np.atleast_1d(phi_)

          m_ = Bi @ (d_next - A @ v)
          val_ = np.concatenate((m_, phi_))
          ll += mvn.logpdf(val_, zeroAlgebra, S[k])
          
          return -ll
        
        mu, Sigma = SED.thetaTranslationDynamicsPosterior(o, R_cur, theta[t-1,k],
          S[k], A, B, nextU=theta[t+1,k])
        mu_noFuture, Sigma_noFuture = SED.thetaTranslationDynamicsPosterior(o, R_cur, theta[t-1,k],
          S[k], A, B)
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


    plt.figure(figsize=(12,8))
    for k in range(K):
      plt.subplot(2,2,k+1)
      d_k = theta[:,k,:-1,-1]
      omega_trans = omega[k,:-1,-1]
      plt.scatter(*d_k.T, color=colors[k], s=1)
      plt.plot(*du.stats.Gauss2DPoints(zeroObs, Strans[k]), color=colors[k])
      plt.title(f'Part {k+1}')
      plt.gca().set_aspect('equal', 'box')
      # plt.plot(*du.stats.Gauss2DPoints(omega_trans, Strans), color=colors[k])
    plt.savefig(f'{path}/cov.png', dpi=300, bbox_inches='tight')
    plt.show()

  def test6(s):
    # general parameters
    o = SED.opts(lie='se2')
    m = getattr(lie, o.lie)
    if o.lie == 'se2': mRot = lie.so2
    else: mRot = lie.so3

    # T, K = ( 10000, 4 )
    T, K = ( 10, 4 )
    
    # body frame
    Q = np.diag((1.0, 2.0, .05))
    x = np.zeros((T,) + o.dxGm)
    x[0] = np.eye(o.dxA)
    q = mvn.rvs(np.zeros(o.dxA), Q, size=T-1)
    for t in range(1, T):
      # qt = mvn.rvs(np.zeros(o.dxA), Q)
      x[t] = x[t-1] @ m.expm(m.alg(q[t-1]))

    # parts
    # Strans = [ iw.rvs(10, 50*np.eye(o.dy)) for k in range(K) ]
    # Srot = np.diag([0.05])

    S = iw.rvs(10, np.diag((50, 50, 0.5)), size=K)
    Strans = S[:,:o.dy,:o.dy]
    Srot = S[:,o.dy:,o.dy:]

    E = iw.rvs(10, np.diag((50, 5)), size=K)

    theta = np.zeros((T, K) + o.dxGm)

    w = np.array([[0, 12, 0], [0, -12, 0], [12, 0, 0], [-12, 0, 0]])
    omega = np.zeros((K,) + o.dxGm)
    for k in range(K): omega[k] = m.expm(m.alg(w[k]))
    for k in range(K): theta[0,k] = np.eye(o.dy+1)

    # theta[0] = omega.copy()
    # theta[0] = omega.copy()

    zeroObs = np.zeros(o.dy)
    zeroRot = np.zeros(o.dxA - o.dy)
    IdentityRot = np.eye(o.dy)

    alpha, beta = (0.95, 0.05)
    A = np.sqrt(np.diag(alpha * np.ones(o.dy)))
    B = np.sqrt(np.diag(beta * np.ones(o.dy)))
    Ai = np.linalg.inv(A)
    Bi = np.linalg.inv(B)


    y = [ [] for t in range(T) ]   
    for t in range(1,T):
      for k in range(K):
        d_prev = m.Rt(theta[t-1,k])[1]

        # random translation in lie group, serves the role of n[t]
        m_tk = mvn.rvs(zeroObs, Strans[k]) 

        # random rotation in lie algebra
        phi_tk = mvn.rvs(zeroRot, Srot[k]) 

        # rotation in lie group
        R_tk = mRot.expm(mRot.alg(phi_tk))

        # updated translation
        d_new = A @ d_prev + B @ m_tk

        # updated transformation
        S_tk = SED.MakeRd(R_tk, d_new)
        theta[t,k] = S_tk

        e_tk = mvn.rvs(zeroObs, E[k], size=100)
        y[t].append(SED.TransformPointsNonHomog(
          x[t] @ omega[k] @ theta[t,k], e_tk))

    thetaR = theta.copy()

    for t in range(1,T-1):
      for k in range(K):
    # for t in range(4,5):
    #   for k in range(2,3):
        R_cur, d_cur = m.Rt(theta[t,k])
        zeroAlgebra = np.zeros(o.dxA)

        # test inference here
        R_prev, d_prev = m.Rt(theta[t-1,k])
        R_next, d_next = m.Rt(theta[t+1,k])

        lhs = x[t] @ omega[k]
        rhs = np.eye(o.dy+1)
        R_U = SED.sampleRotation(o, y[t][k], d_cur, lhs, rhs, E[k], theta[t-1,k],
          S[k], nextU=theta[t+1,k], A=A, B=B)
        thetaR[t,k,:o.dy,:o.dy] = R_U

        # R_theta_tk = sampleRotation(o, y_tk, d_theta_tk, lhs, rhs, E[k],
        #   thetaPrev, SPrev, nextU=thetaNext, nextS=SNext)



        # def thetat_fwdBack(v):
        #   ll = 0.0
        #
        #   # v is a translation; we'll already have a rotation so make the full
        #   V = SED.MakeRd(R_cur, v)
        #
        #   # past
        #   phi = mRot.algi(mRot.logm(R_prev.T @ R_cur))
        #   phi = np.atleast_1d(phi)
        #
        #   m = Bi @ (v - A @ d_prev)
        #   val = np.concatenate((m, phi))
        #   ll = mvn.logpdf(val, zeroAlgebra, S[k])
        #
        #   # future
        #   phi_ = mRot.algi(mRot.logm(R_cur.T @ R_next))
        #   phi_ = np.atleast_1d(phi_)
        #
        #   m_ = Bi @ (d_next - A @ v)
        #   val_ = np.concatenate((m_, phi_))
        #   ll += mvn.logpdf(val_, zeroAlgebra, S[k])
        #
        #   return -ll
        # print(f'd_cur:\n {d_cur}\n mu:\n {mu}\n Sigma:\n {Sigma}\n S:\n{S[k]}')
        # g = nd.Gradient(thetat_fwdBack)
        # map_est = so.minimize(thetat_fwdBack, np.zeros(o.dy), method='BFGS',
        #   jac=g)
        # norm = np.linalg.norm(map_est.x - mu)
        # print(f'norm: {norm:.6f}')
        # assert norm < 1e-2, \
        #   f'SED.test5 bad, norm {norm:.6f}'
        #

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






    # plt.scatter(*d_cur, color='g')
    # plt.plot(*du.stats.Gauss2DPoints(mu, Sigma), color='b')
    # plt.plot(*du.stats.Gauss2DPoints(mu_noFuture, Sigma_noFuture), color='r')
    # plt.scatter(*mu, color='b')
    # plt.scatter(*mu_noFuture, color='r')
    # plt.show()
