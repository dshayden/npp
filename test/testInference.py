import unittest
import numpy as np
from npp import SED, evalSED
from scipy.stats import multivariate_normal as mvn
import scipy.optimize as so
import numdifftools as nd
import lie
import du
import IPython as ip

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
    # s.data = data

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
