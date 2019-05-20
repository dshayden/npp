import numpy as np
import scipy.linalg as sla
import itertools
import networkx as nx, functools, du

# def shortest_rigid_path(y, m, dist):
#   T, N, D = y.shape
#   rd = np.stack([ rigid_from_obs(yy) for yy in y ])
#   
#   def _MakeTrellisEdge(t, i, k, x, nU, dist2):
#     node1 = t*nU + i
#     node2 = (t+1)*nU + k
#     weight = dist2( x[t, i], x[t+1, k] )
#     return (node1, node2, {'weight': weight})
#   nU = 2**(D-2)
#
#   MakeEdge = functools.partial(_MakeTrellisEdge, x=rd, nU=nU, dist2=dist)
#       
#   edges = du.For(MakeEdge, list(np.ndindex(T-1, nU, nU)), showProgress=False)
#   sourceEdges = [ ('source', k, {'weight': 0}) for k in range(nU) ]
#   sinkEdges = [ ('sink', T*nU - 1 - k, {'weight': 0}) for k in range(nU) ]
#
#   G = nx.Graph()
#   G.add_edges_from(sourceEdges + edges + sinkEdges)
#   path = nx.shortest_path(G, source='source', target='sink', weight='weight')
#   path = np.array(path[1:-1]) # remove source and sink nodes
#
#   x = rd[ (path - (path%nU)) // nU, path % nU ]
#
#   return x

def rigid_from_obs(y):
  # y is assumed to have trailing 1
  mu = np.mean(y[:,:-1], axis=0)
  sigma = np.cov(y[:,:-1].T)
  _, sigs = eigh_proper_all(sigma)
  # R = np.stack([ sig.T for sig in sigs ])
  # D = np.stack([ -r.dot(mu) for r in R])

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
