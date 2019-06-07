import unittest
import numpy as np
from npp import SED, evalSED
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
    ## this is probably wrong, probably need shape[1]
    s.KTrue = data['theta'].shape[1]
    s.xTrue = du.asShape(data['x'], (s.T,) + s.o.dxGm)
    s.thetaTrue = data['theta']
    s.ETrue = data['E']
    s.STrue = data['S']
    s.QTrue = data['Q']
    s.zTrue = data['z']
    s.piTrue = data['pi']
    s.data = data

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
    assert norm <= 10.0, f'bad inferQ, norm: {norm:.6f}'

  def testInferS(s):
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
      assert norm <= 5.0, f'bad inferSk, norm: {norm:.6f}'


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
