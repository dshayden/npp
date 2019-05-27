import numpy as np
import du, lie

def dist2_shortest(o, X, Y, c, d):
  """ Squared distance min_R d(XR,Y)^2 for R the symmetric rotations.

  INPUT
    o (argparse.Namespace): Algorithm options
    X (ndarray, dxGm): Element of SE(D)
    Y (ndarray, dxGm): Element of SE(D)
    c (float): Rotation distance scaling
    d (float): Translation distance scaling

  OUTPUT
    dist2 (float): Squared distance of min_R d(XR,Y)^2 for rotations R
  """
  m = getattr(lie, o.lie)
  if o.lie == 'se2': nPossible = 2
  elif o.lie == 'se3': nPossible = 4
  else: assert False, 'Only support se2 or se3'

  v = np.zeros((nPossible, m.dof)) # first row is identity
  for i in range(1,nPossible):
    v[i, o.dy+i-1] = np.pi
  dist2 = [ m.dist2(X.dot(m.expm(m.alg(vv))), Y, c, d) for vv in v ]
  return np.min(dist2)
