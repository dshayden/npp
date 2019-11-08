import unittest
import numpy as np
from npp import SED, evalSED, drawSED as draw, icp
from scipy.stats import multivariate_normal as mvn
from scipy.stats import invwishart as iw
import scipy.optimize as so
import numdifftools as nd
import lie
import du
import matplotlib.pyplot as plt
import IPython as ip
np.set_printoptions(suppress=True, precision=4)

class testMisc(unittest.TestCase):
  def setUp(s): None
  def testPseudo(s):
    group = 'se2'
    m = getattr(lie, group)
    mRot = getattr(lie, 'so2')
    dy = m.n - 1

    x = m.rvs()
    R, d = m.Rt(x)

    a = m.logmP(x)
    assert np.allclose( a[:dy,:dy], mRot.logm(R) )
    assert np.allclose( a[:-1,-1], d )
    
    A = m.expmP(a)
    assert np.allclose(A[:dy,:dy], R)
    assert np.allclose(A[:-1,-1], d)

  def testVi(s):
    group = 'se2'
    m = getattr(lie, group)
    mRot = getattr(lie, 'so2')
    dy = m.n - 1

    x = m.rvs()
    R_x, d_x = m.Rt(x)

    mu = m.rvs()
    R_mu, d_mu = m.Rt(mu)

    mu_inv_x = m.inv(mu) @ x
    R_mu_inv_x = m.Rt(mu_inv_x)[0]

    print(mu_inv_x.shape)
    Vi = m.getVi(mu_inv_x)
    log_R_mu_inv_x = mRot.logm(R_mu_inv_x)

    print(Vi)
    print(log_R_mu_inv_x)

  def testLinInterp(s):
    group = 'se2'
    m = getattr(lie, group)
    mRot = getattr(lie, 'so2')
    dy = m.n - 1
    
    theta = m.rvs()
    R_theta, d_theta = m.Rt(theta)

    w = m.rvs()
    R_w, d_w = m.Rt(w)

    lam = 0.3
    
    true = m.expm( -lam * m.logm( m.inv(theta) @ w ) )
    R_true, d_true = m.Rt(true)
    
    Vi_theta_inv_w = m.getVi( R_theta.T @ R_w )
    
    R_test = mRot.expm( -lam * mRot.logm( R_theta.T @ R_w ) )
    
    V_e = np.linalg.inv(m.getVi(R_test))
    d_test = -lam * V_e @ Vi_theta_inv_w @ R_theta.T @ (d_w - d_theta)

    assert np.isclose(np.linalg.norm( R_true - R_test ), 0.0)
    assert np.isclose(np.linalg.norm( d_true - d_test ), 0.0)

  def testConditional(s):
    D = 2
    zeroD = np.zeros(D)
    zeroD2 = np.zeros(2*D)

    H = iw.rvs(10, np.eye(D)) # force it to be invertible
    b = mvn.rvs(zeroD)
    Sigma = iw.rvs(10, 10*np.eye(2*D))

    x = mvn.rvs(zeroD2, Sigma)
    x1, x2 = ( x[:D], x[D:] )

    # want distribution for x1 | x2
    # but have access to H x1 + b | x2
    trueMu, trueSig = SED.inferNormalConditional(x2, np.eye(D), zeroD, zeroD2, Sigma)

    # modify mu, Sigma
    Hi = np.linalg.inv(H)
    muMod = zeroD2
    muMod[:D] = Hi @ (muMod[:D] - b)
    SigmaMod = Sigma.copy()
    SigmaMod[:D,:D] = Hi @ SigmaMod[:D,:D] @ Hi.T
    SigmaMod[:D,D:] = Hi @ SigmaMod[:D,D:]
    SigmaMod[D:,:D] = SigmaMod[:D,D:].T
    testMu, testSig = SED.inferNormalConditional(x2, np.eye(D), zeroD, muMod, SigmaMod)
    # adjust testMu 
    testMu = H @ testMu + b
    testSig = H @ testSig @ H.T

    print('True Mu\n', trueMu, '\ntestMu\n', testMu)
    print('True Sigma\n', trueSig, '\ntestSigma\n', testSig)

    plt.scatter(*x1, color='g', label='True x1')
    plt.plot(*du.stats.Gauss2DPoints(trueMu, trueSig), c='g', linestyle='--', label='True Conditional')
    plt.plot(*du.stats.Gauss2DPoints(testMu, testSig), c='b', linestyle=':', label='Test Conditional')
    plt.legend()
    plt.show()
