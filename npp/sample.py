import numpy as np
from scipy.stats import expon
# import IPython as ip

def slice_univariate(f, x0, **kwargs):
  """ Univariate slice sampler.

  INPUT
    f (fcn): function f(x), evaluates log pdf
    x0 (float): initial point

  KEYWORD INPUT
    nSamples (int): number of samples, default 10
    w (float): doubling width, default 1
    P (int): max doubling iterations, default 10
    xmin (float): minimum for x support, default np.finfo(np.float).min
    xmax (float): maximum for x support, default np.finfo(np.float).max
    debug (bool): print debugging information, default False
  
  OUTPUT
    xs (ndarray, [nSamples,]): samples
    ll (ndarray, [nSamples,]): log-likelihood
  """
  w = kwargs.get('w', 1.0)
  nSamples = kwargs.get('nSamples', 10)
  P = kwargs.get('P', 10)
  xmin = kwargs.get('xmin', np.finfo(np.float).min)
  xmax = kwargs.get('xmax', np.finfo(np.float).max)
  debug = kwargs.get('debug', False)
  # debug = kwargs.get('debug', True)

  def doubling(x, z):
    # L = x - w*np.random.rand()
    # R = L + w
    L = np.maximum(x - w*np.random.rand(), xmin)
    R = np.minimum(L+w, xmax)
    K = P
    while K>0 and (z<f(L) or z<f(R)):
      if np.random.rand() < 0.5: L = np.maximum(L - (R-L), xmin)
      else: R = np.minimum(R + (R-L), xmax)
      if debug: print(f'In doubling loop, L: {L:.4f}, R: {R:.4f}, K: {K}')
      K = K - 1
    return L, R

  def accept(x0, x1, z, L, R):
    if debug: print(f'In accept: L: {L:.4f}, R: {R:.4f}, R-L: {(R-L):.4f}, 1.1*w: {(1.1*w):.4f}')
    Lhat, Rhat = (L, R)
    D = False
    while (Rhat - Lhat) > (1.1 * w):
      M = (Lhat + Rhat) / 2.
      if (x0 < M and x1 >= M) or (x0 >= M and x1 < M): D = True
      if x1 < M: Rhat = M
      else: Lhat = M

      if debug: print(f'In accept loop: Lhat: {Lhat:.4f}, Rhat: {Rhat:.4f}, M: {M:.4f}, Rhat-Lhat: {(Rhat-Lhat):.4f}, D: {D}')
      if D and z >= f(Lhat) and z >= f(Rhat): return False
    return True

  def shrinkage(x0, z, L, R):
    Lbar, Rbar = (L, R)   
    while True:
      if debug: print(f'In shrinkage loop, Lbar: {Lbar:.4f}, Rbar: {Rbar:.4f}')
      x1 = Lbar + np.random.rand() * (Rbar - Lbar)
      ll = f(x1)
      if z < ll and accept(x0, x1, z, Lbar, Rbar): break
      if x1 < x0: Lbar = x1
      else: Rbar = x1
    return x1, ll
  
  xs = np.zeros(nSamples)
  ll = np.zeros(nSamples)
  prevX = x0
  for nS in range(nSamples):
    z = f(prevX) - expon.rvs()
    L, R = doubling(prevX, z)
    xs[nS], ll[nS] = shrinkage(prevX, z, L, R)
    prevX = xs[nS]

  return xs, ll

if __name__ == "__main__":
  import matplotlib.pyplot as plt
  from scipy.stats import multivariate_normal as mvn
  import IPython as ip
  from functools import partial
  import du, du.stats
  D = 2
  Sigma = np.array([[10, 2], [2, 3]])
  mu = np.zeros(2)
  print('True Sigma\n', Sigma)
  print('True mu\n', mu)

  nSamples = 1000
  w = 1.0
  xs = np.zeros((nSamples, D))
  prevX = np.array([5.0,10.0])
  for nS in range(nSamples):
    for d in range(D):
      if d == 0:
        def f(x): return mvn.logpdf(np.array([x, prevX[1]]), mu, Sigma)
        prevX[0], _ = slice_univariate(f, prevX[0], w=w, nSamples=1)
      else:
        def f(x): return mvn.logpdf(np.array([prevX[0], x]), mu, Sigma)
        prevX[1], _ = slice_univariate(f, prevX[1], w=w, nSamples=1)
    xs[nS] = prevX.copy()
  
  xs = xs[nSamples//10:]
  print('Sample Sigma:\n', np.cov(xs.T))
  print('Sample mu:\n', np.mean(xs,axis=0))
  plt.scatter(xs[:,0], xs[:,1], s=0.1, c='b')
  plt.plot(*du.stats.Gauss2DPoints(mu, Sigma))
  plt.axis('equal')
  plt.show()
