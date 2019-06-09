import unittest
import numpy as np
from npp import SED, evalSED, util
import du, lie
from scipy.stats import invwishart as iw, multivariate_normal as mvn
import IPython as ip, sys
import numdifftools as nd
import scipy.optimize as so
import matplotlib.pyplot as plt
import testUtil as tu

class test_random(unittest.TestCase):
  def setUp(s):
    None

  def test_rotation_symmetries_se2(s):
    T, K, N, grp = (3, 1, 10, 'se2')
    o, x, theta, E, S, Q, y, z = tu.GenerateRandomDataset(T, K, N, grp)
    m = getattr(lie, o.lie)

    # show that logpdf_data_t is same for rotation symmetries of theta
    t,k = (1,0)
    symmetries = util.rotation_symmetries(theta[t,k])
    lls = [ SED.logpdf_data_t(o, y[t], z[t], x[t], sym[np.newaxis], E)
      for sym in symmetries ]
    assert np.all( [np.isclose(ll_, lls[0]) for ll_ in lls] )

  def test_rotation_symmetries_se3(s):
    T, K, N, grp = (3, 1, 1000, 'se3')
    o, x, theta, E, S, Q, y, z = tu.GenerateRandomDataset(T, K, N, grp)
    m = getattr(lie, o.lie)

    # show that logpdf_data_t is same for rotation symmetries of theta
    t,k = (1,0)
    symmetries = util.rotation_symmetries(theta[t,k])

    lls = [ SED.logpdf_data_t(o, y[t], z[t], x[t], sym[np.newaxis], E)
      for sym in symmetries ]

    assert np.all( [ np.isclose(ll_, lls[0]) for ll_ in lls ] )
    
  def test_decomp(s):
    T, K, N, grp = (3, 1, 0, 'se2')
    o, x, theta, E, S, Q, y, z = tu.GenerateRandomDataset(T, K, N, grp)
    m = getattr(lie, o.lie)

    # infer d_x_t in forward direction
    t = 1   

    R_x_tminus1, d_x_tminus1 = m.Rt(x[t-1])
    R_x_tplus1, d_x_tplus1 = m.Rt(x[t+1])
    R_x_t, _ = m.Rt(x[t])

    V_x_tminus1_x_t_inv = m.getVi( m.inv(x[t-1]).dot(x[t]) )
    _, d_x_tminus1_x_t = m.Rt(m.inv(x[t-1]).dot(x[t]))
    u_x_t = m.algi(m.logm(m.inv(x[t-1]).dot(x[t])))[:-1]
    norm = np.linalg.norm(u_x_t - V_x_tminus1_x_t_inv.dot(d_x_tminus1_x_t))
    assert norm < 1e-8, 'bad norm'
    
    V_x_t_x_tplus1_inv = m.getVi( m.inv(x[t]).dot(x[t+1]) )
    _, d_x_t_x_tplus1 = m.Rt(m.inv(x[t]).dot(x[t+1]))
    u_x_tplus1 = m.algi(m.logm(m.inv(x[t]).dot(x[t+1])))[:-1]
    norm = np.linalg.norm(u_x_tplus1 - V_x_t_x_tplus1_inv.dot(d_x_t_x_tplus1))
    assert norm < 1e-8, 'bad norm'

    x_t = util.make_rigid_transform(R_x_t, d_x_tminus1)
    V_x_tminus1_x_t_inv = m.getVi( m.inv(x[t-1]).dot(x_t) )
    _, d_x_tminus1_x_t = m.Rt(m.inv(x[t-1]).dot(x_t))
    u_x_t = m.algi(m.logm(m.inv(x[t-1]).dot(x_t)))[:-1]
    norm = np.linalg.norm(u_x_t - V_x_tminus1_x_t_inv.dot(d_x_tminus1_x_t))
    assert norm < 1e-8, 'bad norm'

    x_t = util.make_rigid_transform(R_x_t, d_x_tminus1)
    V_x_t_x_tplus1_inv = m.getVi( m.inv(x_t).dot(x[t+1]) )
    _, d_x_t_x_tplus1 = m.Rt(m.inv(x_t).dot(x[t+1]))
    u_x_tplus1 = m.algi(m.logm(m.inv(x_t).dot(x[t+1])))[:-1]
    norm = np.linalg.norm(u_x_tplus1 - V_x_t_x_tplus1_inv.dot(d_x_t_x_tplus1))
    assert norm < 1e-8, 'bad norm'

  def test_xFwd_se2(s):
    T, K, N, grp = (6, 1, 0, 'se2')
    o, x, theta, E, S, Q, y, z = tu.GenerateRandomDataset(T, K, N, grp)
    m = getattr(lie, o.lie)

    # infer d_x_t in forward direction
    t = 5
    
    R_x_tminus1, d_x_tminus1 = m.Rt(x[t-1])
    R_x_t, _ = m.Rt(x[t])

    V_x_tminus1_x_t_inv = m.getVi( m.inv(x[t-1]).dot(x[t]) )
    H = V_x_tminus1_x_t_inv.dot(R_x_tminus1.T)
    b = -H.dot(d_x_tminus1)
    u = np.zeros(o.dxA)
    x2 = np.array([ m.algi(m.logm( m.inv(x[t-1]).dot(x[t]) ))[-1] ])

    mu, Sigma = SED.inferNormalConditional(x2, H, b, u, Q)

    # determine mu as MAP estimate
    Qi = np.linalg.inv(Q)
    def nllNormalConditional(d_x_t):
      x_t = util.make_rigid_transform(R_x_t, d_x_t)

      t1 = m.algi(m.logm( m.inv(x[t-1]).dot(x_t) ))
      nll = 0.5 * t1.dot(Qi).dot(t1)

      return nll

    g = nd.Gradient(nllNormalConditional)
    map_est = so.minimize(nllNormalConditional, np.zeros(2),
      method='BFGS', jac=g)

    norm = np.linalg.norm(map_est.x - mu)
    # print(f'norm: {norm:.12f}')
    assert norm < 1e-2, f'SED.inferNormalConditional bad, norm {norm:.2f}'

    # # grid search and contour plot   
    # nP = 100
    # vx0, vx1 = np.meshgrid(
    #   np.linspace(map_est.x[0]-3, map_est.x[0]+3,nP),
    #   np.linspace(map_est.x[1]-3, map_est.x[1]+3,nP)
    # )
    #
    # pts = np.vstack((vx0.flatten(), vx1.flatten())).T
    # nll = du.asShape(np.stack([nllNormalConditional(pt) for pt in pts]), vx0.shape)
    # plt.contour(vx0, vx1, -nll)
    # plt.scatter(*map_est.x, c='k', label='Optimum', s=30)
    # plt.scatter(*mu, c='b', label='Inferred', s=20)
    # plt.title(f'Norm: {norm:.2f}')
    # plt.legend()
    # plt.show()

  def test_xFwdBack_se2(s):
    T, K, N, grp = (7, 1, 0, 'se2')
    o, x, theta, E, S, Q, y, z = tu.GenerateRandomDataset(T, K, N, grp)
    m = getattr(lie, o.lie)

    # infer d_x_t in full conditional
    t = 5   
    
    R_x_tminus1, d_x_tminus1 = m.Rt(x[t-1])
    R_x_tplus1, d_x_tplus1 = m.Rt(x[t+1])
    R_x_t, _ = m.Rt(x[t])

    V_x_tminus1_x_t_inv = m.getVi( m.inv(x[t-1]).dot(x[t]) )
    H = V_x_tminus1_x_t_inv.dot(R_x_tminus1.T)
    b = -H.dot(d_x_tminus1)
    u = np.zeros(o.dxA)
    x2 = np.array([ m.algi(m.logm( m.inv(x[t-1]).dot(x[t]) ))[-1] ])

    V_x_t_x_tplus1_inv = m.getVi( m.inv(x[t]).dot(x[t+1]) )

    H_ = -V_x_t_x_tplus1_inv.dot(R_x_t.T)
    b_ = -H_.dot(d_x_tplus1)
    u_ = np.zeros(3)
    x2_ = np.array([ m.algi(m.logm( m.inv(x[t]).dot(x[t+1]) ))[-1] ])
    Q_ = Q

    mu, Sigma = SED.inferNormalConditionalNormal(
      x2, H, b, u, Q, x2_, H_, b_, u_, Q_)

    # determine mu as MAP estimate
    Qi = np.linalg.inv(Q)
    def nllNormalConditional2(d_x_t):
      x_t = util.make_rigid_transform(R_x_t, d_x_t)

      t1 = m.algi(m.logm( m.inv(x[t-1]).dot(x_t) ))
      nll = 0.5 * t1.dot(Qi).dot(t1)

      t2 = m.algi(m.logm( m.inv(x_t).dot(x[t+1]) ))
      nll += 0.5 * t2.dot(Qi).dot(t2)

      return nll

    g = nd.Gradient(nllNormalConditional2)
    map_est = so.minimize(nllNormalConditional2, np.zeros(2),
      method='BFGS', jac=g)

    norm = np.linalg.norm(map_est.x - mu)
    # print(f'norm: {norm:.12f}')
    assert norm < 1e-2, f'SED.inferNormalConditional bad, norm {norm:.2f}'

    # # grid search and contour plot   
    # nP = 100
    # vx0, vx1 = np.meshgrid(
    #   np.linspace(map_est.x[0]-3, map_est.x[0]+3,nP),
    #   np.linspace(map_est.x[1]-3, map_est.x[1]+3,nP)
    # )
    #
    # pts = np.vstack((vx0.flatten(), vx1.flatten())).T
    # nll = du.asShape(np.stack([nllNormalConditional2(pt) for pt in pts]), vx0.shape)
    # plt.contour(vx0, vx1, -nll)
    # plt.scatter(*map_est.x, c='k', label='Optimum', s=30)
    # plt.scatter(*mu, c='b', label='Inferred', s=20)
    # plt.title(f'Norm: {norm:.2f}')
    # plt.legend()
    # plt.show()


  def test_inferNormalConditionalNormal(s):
    dx1 = 2
    dx2 = 3
    d = dx1+dx2

    H = lie.so2.alg(np.random.rand())
    b = mvn.rvs(np.zeros(dx1), 5*np.eye(dx1))
    x2 = mvn.rvs(np.zeros(dx2), 5*np.eye(dx2))
    u = mvn.rvs(np.zeros(d), 5*np.eye(d))
    S = iw.rvs(2*d, 10*np.eye(d))
    Si = np.linalg.inv(S)

    H_ = lie.so2.alg(np.random.rand())
    b_ = mvn.rvs(np.zeros(dx1), 5*np.eye(dx1))
    x2_ = mvn.rvs(np.zeros(dx2), 5*np.eye(dx2))
    u_ = mvn.rvs(np.zeros(d), 5*np.eye(d))
    S_ = iw.rvs(2*d, 10*np.eye(d))
    S_i = np.linalg.inv(S_)

    mu, Sigma = SED.inferNormalConditionalNormal(x2, H, b, u, S, x2_, H_, b_, u_, S_)

    # determine mu as MAP estimate
    def nllNormalConditional2(v):
      val1 = np.concatenate((H.dot(v) + b, x2))
      t1 = val1 - u
      nll = 0.5*t1.dot(Si).dot(t1)

      val2 = np.concatenate((H_.dot(v) + b_, x2_))
      t2 = val2 - u_
      nll += 0.5*t2.dot(S_i).dot(t2)

      return nll

    g = nd.Gradient(nllNormalConditional2)
    map_est = so.minimize(nllNormalConditional2, np.zeros(dx1),
      method='BFGS', jac=g)

    norm = np.linalg.norm(map_est.x - mu)
    # print(f'norm: {norm:.12f}')
    assert norm < 1e-2, f'SED.inferNormalConditional bad, norm {norm:.6f}'

  def test_inferNormalConditional(s):
    dx1 = 2
    dx2 = 3
    d = dx1+dx2

    H = lie.so2.alg(np.random.rand())
    b = mvn.rvs(np.zeros(dx1), 5*np.eye(dx1))
    x2 = mvn.rvs(np.zeros(dx2), 5*np.eye(dx2))
    u = mvn.rvs(np.zeros(d), 5*np.eye(d))
    S = iw.rvs(2*d, 10*np.eye(d))
    Si = np.linalg.inv(S)

    # analytically infer x1 ~ N(mu, Sigma)
    mu, Sigma = SED.inferNormalConditional(x2, H, b, u, S)

    # determine mu as MAP estimate
    def nllNormalConditional(v):
      val = np.concatenate((H.dot(v) + b, x2))
      t1 = val - u
      return 0.5*t1.dot(Si).dot(t1)
    g = nd.Gradient(nllNormalConditional)
    map_est = so.minimize(nllNormalConditional, np.zeros(dx1),
      method='BFGS', jac=g)

    norm = np.linalg.norm(map_est.x - mu)
    # print(f'norm: {norm:.12f}')
    assert norm < 1e-2, f'SED.inferNormalConditional bad, norm {norm:.6f}'

  def test_inferNormalConditional_diag(s):
    dx1 = 2
    dx2 = 3
    d = dx1+dx2

    H = lie.so2.alg(np.random.rand())
    b = mvn.rvs(np.zeros(dx1), 5*np.eye(dx1))
    x2 = mvn.rvs(np.zeros(dx2), 5*np.eye(dx2))
    u = mvn.rvs(np.zeros(d), 5*np.eye(d))
    S = np.diag(np.diag(iw.rvs(2*d, 10*np.eye(d))))
    Si = np.linalg.inv(S)

    # analytically infer x1 ~ N(mu, Sigma)
    mu, Sigma = SED.inferNormalConditional(x2, H, b, u, S)

    # determine mu as MAP estimate
    def nllNormalConditional(v):
      val = np.concatenate((H.dot(v) + b, x2))
      t1 = val - u
      return 0.5*t1.dot(Si).dot(t1)
    g = nd.Gradient(nllNormalConditional)
    map_est = so.minimize(nllNormalConditional, np.zeros(dx1),
      method='BFGS', jac=g)

    norm = np.linalg.norm(map_est.x - mu)
    # print(f'norm: {norm:.12f}')
    assert norm < 1e-2, f'SED.inferNormalConditional bad, norm {norm:.6f}'

  def test_inferNormalConditionalNormal_diag(s):
    dx1 = 2
    dx2 = 3
    d = dx1+dx2

    H = lie.so2.alg(np.random.rand())
    b = mvn.rvs(np.zeros(dx1), 5*np.eye(dx1))
    x2 = mvn.rvs(np.zeros(dx2), 5*np.eye(dx2))
    u = mvn.rvs(np.zeros(d), 5*np.eye(d))
    S = np.diag(np.diag(iw.rvs(2*d, 10*np.eye(d))))
    Si = np.linalg.inv(S)

    H_ = iw.rvs(2*dx1, np.eye(dx1))
    H_i = np.linalg.inv(H_)
    b_ = mvn.rvs(np.zeros(dx1), 5*np.eye(dx1))
    x2_ = mvn.rvs(np.zeros(dx2), 5*np.eye(dx2))
    u_ = mvn.rvs(np.zeros(d), 5*np.eye(d))
    S_ = np.diag(np.diag(iw.rvs(2*d, 10*np.eye(d))))
    S_i = np.linalg.inv(S_)

    mu, Sigma = SED.inferNormalConditionalNormal(x2, H, b, u, S, x2_, H_, b_, u_, S_)

    # determine mu as MAP estimate
    def nllNormalConditional2(v):
      val1 = np.concatenate((H.dot(v) + b, x2))
      t1 = val1 - u
      nll = 0.5*t1.dot(Si).dot(t1)

      val2 = np.concatenate((H_.dot(v) + b_, x2_))
      t2 = val2 - u_
      nll += 0.5*t2.dot(S_i).dot(t2)

      return nll

    g = nd.Gradient(nllNormalConditional2)
    map_est = so.minimize(nllNormalConditional2, np.zeros(dx1),
      method='BFGS', jac=g)

    norm = np.linalg.norm(map_est.x - mu)
    # print(f'norm: {norm:.12f}')
    assert norm < 1e-2, f'SED.inferNormalConditional bad, norm {norm:.6f}'

  def test_xFwd_se3(s):
    T, K, N, grp = (6, 1, 0, 'se3')
    o, x, theta, E, S, Q, y, z = tu.GenerateRandomDataset(T, K, N, grp)
    m = getattr(lie, o.lie)

    # infer d_x_t in forward direction
    t = 5
    
    R_x_tminus1, d_x_tminus1 = m.Rt(x[t-1])
    R_x_t, _ = m.Rt(x[t])

    V_x_tminus1_x_t_inv = m.getVi( m.inv(x[t-1]).dot(x[t]) )
    H = V_x_tminus1_x_t_inv.dot(R_x_tminus1.T)
    b = -H.dot(d_x_tminus1)
    u = np.zeros(o.dxA)
    x2 = m.algi(m.logm( m.inv(x[t-1]).dot(x[t]) ))[o.dy:o.dxA]

    mu, Sigma = SED.inferNormalConditional(x2, H, b, u, Q)

    # determine mu as MAP estimate
    Qi = np.linalg.inv(Q)
    def nllNormalConditional(d_x_t):
      x_t = util.make_rigid_transform(R_x_t, d_x_t)

      t1 = m.algi(m.logm( m.inv(x[t-1]).dot(x_t) ))
      nll = 0.5 * t1.dot(Qi).dot(t1)

      return nll

    g = nd.Gradient(nllNormalConditional)
    map_est = so.minimize(nllNormalConditional, np.zeros(o.dy),
      method='BFGS', jac=g)

    norm = np.linalg.norm(map_est.x - mu)
    # print(f'norm: {norm:.12f}')
    assert norm < 1e-2, f'SED.inferNormalConditional bad, norm {norm:.2f}'

  def test_xFwdBack_se3(s):
    T, K, N, grp = (7, 1, 0, 'se3')
    o, x, theta, E, S, Q, y, z = tu.GenerateRandomDataset(T, K, N, grp)
    m = getattr(lie, o.lie)

    # infer d_x_t in full conditional
    t = 5   
    
    R_x_tminus1, d_x_tminus1 = m.Rt(x[t-1])
    R_x_tplus1, d_x_tplus1 = m.Rt(x[t+1])
    R_x_t, _ = m.Rt(x[t])

    V_x_tminus1_x_t_inv = m.getVi( m.inv(x[t-1]).dot(x[t]) )
    H = V_x_tminus1_x_t_inv.dot(R_x_tminus1.T)
    b = -H.dot(d_x_tminus1)
    u = np.zeros(o.dxA)
    x2 = m.algi(m.logm( m.inv(x[t-1]).dot(x[t]) ))[o.dy:o.dxA]

    V_x_t_x_tplus1_inv = m.getVi( m.inv(x[t]).dot(x[t+1]) )

    H_ = -V_x_t_x_tplus1_inv.dot(R_x_t.T)
    b_ = -H_.dot(d_x_tplus1)
    u_ = np.zeros(o.dxA)

    x2_ = m.algi(m.logm( m.inv(x[t]).dot(x[t+1]) ))[o.dy:o.dxA]
    Q_ = Q

    mu, Sigma = SED.inferNormalConditionalNormal(
      x2, H, b, u, Q, x2_, H_, b_, u_, Q_)

    # determine mu as MAP estimate
    Qi = np.linalg.inv(Q)
    def nllNormalConditional2(d_x_t):
      x_t = util.make_rigid_transform(R_x_t, d_x_t)

      t1 = m.algi(m.logm( m.inv(x[t-1]).dot(x_t) ))
      nll = 0.5 * t1.dot(Qi).dot(t1)

      t2 = m.algi(m.logm( m.inv(x_t).dot(x[t+1]) ))
      nll += 0.5 * t2.dot(Qi).dot(t2)

      return nll

    g = nd.Gradient(nllNormalConditional2)
    map_est = so.minimize(nllNormalConditional2, np.zeros(o.dy),
      method='BFGS', jac=g)

    norm = np.linalg.norm(map_est.x - mu)
    # print(f'norm: {norm:.12f}')
    assert norm < 1e-2, f'SED.inferNormalConditional bad, norm {norm:.2f}'
