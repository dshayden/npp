import unittest
import numpy as np
from npp import SED, evalSED, drawSED as draw
from scipy.stats import multivariate_normal as mvn
import scipy.optimize as so
import numdifftools as nd
import lie
import du
import IPython as ip
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

  def testInitRJMCMC(s):
    # run with 
    # nosetests -s --nologcapture test/testInference.py:test_se2_randomwalk3.testInitRJMCMC
    alpha = 0.1
    mL_const = -7.0

    o = s.o
    SED.initPriorsDataDependent(o, s.y)
    x = SED.initXDataMeans(o, s.y)
    theta = np.zeros((s.T, 0,) + o.dxGm)
    E = np.zeros((0, o.dy, o.dy))
    S = np.zeros((0, o.dxA, o.dxA))
    Q = SED.inferQ(o, x)
    mL = [ mL_const * np.ones(s.y[t].shape[0]) for t in range(s.T) ]
    pi = np.array([1.0,])
    z = [ -1 * np.ones(s.y[t].shape[0], dtype=np.int) for t in range(s.T) ]

    llTrue = SED.logJoint(o, s.y, s.zTrue, s.xTrue, s.thetaTrue, s.ETrue,
      s.STrue, s.QTrue, alpha, s.piTrue, mL)
    llInit = SED.logJoint(o, s.y, z, x, theta, E, S, Q, alpha, pi, mL)
    print(f'llTrue: {llTrue:.2f}, llInit: {llInit:.2f}')

    z, pi, theta, E, S, x, Q, ll = SED.sampleStepFC(o, s.y, alpha, z, pi,
      theta, E, S, x, Q, mL, newPart=False)

    # # artifically add a part
    theta = np.concatenate((theta, np.zeros((s.T,1,) + o.dxGm)), axis=1)
    E = np.concatenate((E, np.zeros((o.dy, o.dy))[np.newaxis]), axis=0)
    S = np.concatenate((S, np.zeros((1, o.dxA, o.dxA))), axis=0)
    pi = np.array([0.5, 0.5])
    z, pi, theta, E, S = SED.consolidatePartsAndResamplePi(o, z, pi, alpha, theta, E, S)
    assert theta.shape[1] == 0 and E.shape[0] == 0 and S.shape[0] == 0



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
