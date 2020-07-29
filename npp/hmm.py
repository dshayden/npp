import itertools
import numpy as np
from scipy.special import logsumexp
import du.stats

def build(t0, x0, ts, ks, val, costxx, costxy):
  """ Construct swap hmm for K targets.

  INPUT
    t0 (len-K list): start times for each target
    x0 (len-K list): start states for each target
    ts (len-T list): times inside swap window
    ks (len-K list): target indices
    val (fcn): of the form xt, yt, jt = val(t,k)
    costxx (fcn): of the form cx = costxx(t1,x1,t2,x2,k)
    costxy (fcn): of the form cy = costxy(t,k,x,y)

  OUTPUT
    perms (list): Index into permutations of ks (HMM states are in range(nPerms))
    pi (nPerms): Prior
    Psi (T-1, nPerms, nPerms): Transition Matrix
    psi (T, nPerms): Observation Matrix
  """
  T, K = ( len(ts), len(ks) )
  assert K > 1 and K <= 4, 'Combinatorics too large for K > 4'
  perms = list(itertools.permutations(range(K)))
  nPerms = len(perms)

  # len-T list of len-K list of tuples (xtk, ytk, jtk)
  vs = [ [val(t,k) for k in ks] for t in ts ]

  # build unnormalized log transition matrix
  Psi = np.zeros((T-1, nPerms, nPerms))
  for idx in range(T-1):
    tPrime, t = ( ts[idx], ts[idx+1] )
    for cPrime, pPrime in enumerate(perms):
      for c, p in enumerate(perms):
        for k in range(K): # compute over each k (pPrime[k], p[k])
          xtPrime, _, _ = vs[idx][pPrime[k]]
          xt, yt, _ = vs[idx+1][p[k]]
          Psi[idx, cPrime, c] += costxx(tPrime, xtPrime, t, xt, ks[k])
          Psi[idx, cPrime, c] += costxy(t, ks[k], xt, yt)

  # build unnormalized log prior
  pi = np.zeros(nPerms)
  idx = -1
  t = ts[idx+1]
  for c, p in enumerate(perms):
    for k in range(K): # compute over each k (pPrime[k], p[k])
      tPrime = t0[k]
      xtPrime = x0[k]
      xt, yt, _ = vs[idx+1][p[k]]
      pi[c] += costxx(tPrime, xtPrime, t, xt, ks[k])
      pi[c] += costxy(t, ks[k], xt, yt)

  # import IPython as ip, sys
  # ip.embed(); sys.exit()

  # normalize and return
  Psi = np.exp(Psi - logsumexp(Psi, axis=2, keepdims=True))
  pi = np.exp(pi - logsumexp(pi))
  psi = [ None for t in ts ]
  return perms, pi, Psi, psi

def ffbs(pi, Psi, psi):
  T, K = (Psi.shape[0]+1, Psi.shape[1])
  a = np.zeros((T, K)) # filter messages

  # filter messages
  if psi[0] is None: a[0] = pi
  else: a[0] = pi * psi[0]
  a[0] /= np.sum(a[0])

  for t in range(1,T):
    psi_t = psi[t] if psi[t] is not None else np.ones(K)
    a[t] = psi_t * np.dot(Psi[t-1].T, a[t-1])
    a[t] /= np.sum(a[t])

  # backward sampling
  x = -1 * np.ones(T, dtype=np.int)

  b = np.zeros((T, K))
  b[-1] = a[-1]

  x[-1] = du.stats.catrnd(b[-1][np.newaxis])
  for t in reversed(range(T-1)):
    j = x[t+1]

    psi_t = psi[t] if psi[t] is not None else np.ones(K)
    # pmf[i] is psi_t[i] * Psi[t,i,j] * a[t,i]
    #           ------------------------------
    #                      a[t,j]
    for i in range(K):
      if np.isclose(a[t+1,j], 0.0):
        b[t,i] = 0.0
        continue
      b[t,i] = psi_t[i] * Psi[t,i,j] * a[t,i] / a[t+1,j]
    b[t] /= np.sum(b[t])

    x[t] = du.stats.catrnd(b[t][np.newaxis])

  return [ int(x_) for x_ in x ]
