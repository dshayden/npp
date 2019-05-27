import numpy as np
import scipy.linalg as sla
import itertools
import functools, du

def rigid_from_obs(y):
  # y is assumed to have trailing 1
  mu = np.mean(y[:,:-1], axis=0)
  sigma = np.cov(y[:,:-1].T)
  _, sigs = eigh_proper_all(sigma)

  R = np.stack(sigs)
  D = np.stack([ mu for r in R])
  return np.stack([ make_rigid_transform(r, d) for r, d in zip(R, D) ])

def eigh_proper_all(sigma):
  """ Get all 2^{D-1} eigendecompositions UDU^T s.t. det(U) = 1"""
  sigD, sigU = np.linalg.eigh(sigma)
  D = sigma.shape[0]

  # combinations
  combs = []
  for t in range(D+1):
    combs = combs + list(itertools.combinations(range(D), t))

  possible = np.zeros((2**(D-1), D, D))
  cnt = 0
  for c in combs:
    sU = sigU.copy()
    sU[:,c] = -sU[:,c]
    if np.linalg.det(sU) < 0: continue

    # UDU = sU.dot(np.diag(sigD)).dot(sU.T)
    # norm = np.linalg.norm( UDU - sigma )
    # assert norm < 1e-8, 'bad norm'

    possible[cnt] = sU
    cnt += 1
  return sigD, possible

def make_rigid_transform(R, d):
  return np.concatenate((np.concatenate((R, d[:,np.newaxis]), axis=1),
    np.concatenate((np.zeros((1, R.shape[1])), np.ones((1,1))), axis=1)))
