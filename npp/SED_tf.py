import numpy as np, tensorflow as tf, lie
from . import SED
tf.enable_eager_execution()
import IPython as ip, sys


#### New functions for Parts-ICP ####

# need ability to encode q_t as tensorflow variable to do optimization on
# need ability to encode all s_tk as tensorflow variables
# need objectives for global, parts


#### Old functions for attempting full trajectory optimization ####

def nll_traj(o, yh, pi, G, h_x, h_theta, T, K, Ei_tilde, Si, Qi, v):
  # everything is in tensorflow
  T, K = ( len(yh), len(Si) )
  logPi = tf.log(pi)
  factor = tf.constant(-0.5)

  # dynamics nll
  ll = tf.zeros(())

  ## x
  xvR = tf.reshape( v[:T*o.dxA], (T, o.dxA) )
  ll += factor * tf.reduce_sum(tf.multiply(tf.linalg.matvec(Qi, xvR), xvR))

  ## theta
  for k in range(K):
    ind1 = T*o.dxA + k*T*o.dxA
    ind2 = ind1 + T*o.dxA
    theta_kvR = tf.reshape( v[ind1:ind2], (T, o.dxA) )
    ll += factor * tf.reduce_sum(tf.multiply(tf.linalg.matvec(Si[k], theta_kvR), theta_kvR))

  # ## x
  # ll = tf.zeros(())
  # for t in range(T):
  #   ind = t*o.dxA
  #   v_x_t = v[ind:ind+o.dxA]
  #   ll += factor * tf.reduce_sum(tf.multiply(tf.linalg.matvec(Qi, v_x_t), v_x_t))
  #
  # ## theta
  # offset = T*o.dxA
  # for k in range(K):
  #   for t in range(T):
  #     ind = offset + k*T*o.dxA + t*o.dxA
  #     v_theta_tk = v[ind:ind+o.dxA]
  #     ll += factor * tf.reduce_sum(tf.multiply(tf.linalg.matvec(Si[k], v_theta_tk), v_theta_tk))

  # get world -> part transformations
  x, theta = vec2grp_tf(o, G, h_x, h_theta, T, K, v)
  T_part_world = [ [] for t in range(T) ]
  for t in range(T):
    T_part_world[t] = [ tf.linalg.inv(tf.linalg.matmul(x[t], theta[t][k])) for k in range(K) ]

  # T_part_world is len-T list, each item is len-K sublist
  zeroH = zeroH_tf(o)
  for t in range(T):
    yht = [tf.transpose(tf.matmul(T_pw_k, tf.transpose(yh[t]))) - zeroH
      for T_pw_k in T_part_world[t] ]
    # yth, yhtE is len-K list, each item is R^{N x dy+1}
    yhtE = [ tf.linalg.matmul( yht[k], Ei_tilde[k] ) for k in range(K) ]

    # alpha_t is K x N_t tensor of terms to do logsumexp over
    alpha_t = tf.stack( [ logPi[k] +
      factor * tf.reduce_sum(tf.multiply(yhtE[k], yht[k]), axis=1)
      for k in range(K) ]
    )
    lse_t = tf.math.reduce_logsumexp(alpha_t, axis=0)
    ll += tf.reduce_sum(lse_t)
  
  return -ll


def nll_traj_z(o, yh, z, G, h_x, h_theta, T, K, Ei_tilde, Si, Qi, v):
  # everything is in tensorflow
  T, K = ( len(yh), len(Si) )
  factor = tf.constant(-0.5)

  # dynamics nll
  ## x
  ll = tf.zeros(())
  for t in range(T):
    ind = t*o.dxA
    v_x_t = v[ind:ind+o.dxA]
    ll += factor * tf.reduce_sum(tf.multiply(tf.linalg.matvec(Qi, v_x_t), v_x_t))

  ## theta
  offset = T*o.dxA
  for k in range(K):
    for t in range(T):
      ind = offset + k*T*o.dxA + t*o.dxA
      v_theta_tk = v[ind:ind+o.dxA]
      ll += factor * tf.reduce_sum(tf.multiply(tf.linalg.matvec(Si[k], v_theta_tk), v_theta_tk))

  # get world -> part transformations
  x, theta = vec2grp_tf(o, G, h_x, h_theta, T, K, v)
  T_part_world = [ [] for t in range(T) ]
  for t in range(T):
    T_part_world[t] = [ tf.linalg.inv(tf.linalg.matmul(x[t], theta[t][k])) for k in range(K) ]

  # T_part_world is len-T list, each item is len-K sublist
  zeroH = zeroH_tf(o)
  for t in range(T):
    yht = [tf.transpose(tf.matmul(T_pw_k, tf.transpose(yh[t]))) - zeroH
      for T_pw_k in T_part_world[t] ]
    yhtE = [ tf.linalg.matmul( yht[k], Ei_tilde[k] ) for k in range(K) ]
    ll_t = [ factor*tf.reduce_sum(tf.multiply(yhtE[k], yht[k]))
      for k in range(K) ]
    Nt = yht[0].shape[0]
    for n in range(Nt):
      k = z[t][n]
      if k >= 0:
        ll += factor*tf.reduce_sum(tf.multiply(yhtE[k][n], yht[k][n]))

  return -ll

  
def vec2grp_tf(o, G, h_x, h_theta, T, K, v):
  ## first dxA is x[0], second dxA is x[1], ...
  xv, thetav = ( v[:T*o.dxA], v[T*o.dxA:] )

  xl = [ [] for t in range(T) ]
  ind = 0*o.dxA
  xl[0] = tf.matmul(ex(alg(G, h_x[1])), ex(alg(G, xv[ind:ind+o.dxA])))
  for t in range(1,T):
    ind = t*o.dxA
    xl[t] = tf.matmul(xl[t-1], ex(alg(G, xv[ind:ind+o.dxA])))

  ## store all times of part 0, then all times of part 1, ...
  thetal = [ [ [] for k in range(K) ] for t in range(T) ]
  for k in range(K):
    ind = k*T*o.dxA + 0*o.dxA
    theta_0kv = thetav[ind:ind+o.dxA]
    thetal[0][k] = tf.matmul(ex(alg(G, h_theta[1])), ex(alg(G, theta_0kv)))
    for t in range(1,T):
      ind = k*T*o.dxA + t*o.dxA
      theta_tkv = thetav[ind:ind+o.dxA]
      theta_tkV = ex(alg(G, theta_tkv))
      thetal[t][k] = tf.matmul( thetal[t-1][k], ex(alg(G, theta_tkv)))

  return xl, thetal

  
def grp2vec_tf(o, x, theta):
  m = getattr(lie, o.lie)
  T, K = theta.shape[:2]

  xv0 = m.algi(m.logm(m.inv(o.H_x[1]).dot(x[0])))
  xvT = np.concatenate([m.algi(m.logm(m.inv(x[t-1]).dot(x[t]))) for t in range(1,T)])
  xv = np.concatenate((xv0, xvT))
  assert len(xv) == T*o.dxA

  # parts are stored as [theta_{:k}, theta_{:(k+1)}, ...
  theta_k = [ ]
  for k in range(K):
    theta_0kv = m.algi(m.logm(m.inv(o.H_theta[1]).dot(theta[0,k])))
    theta_Tkv = np.concatenate([m.algi(m.logm(m.inv(theta[t-1,k]).dot(theta[t,k]))) for t in range(1,T)])
    theta_k.append(np.concatenate((theta_0kv, theta_Tkv)))
  thetav = np.concatenate(theta_k)
  assert len(thetav) == T*K*o.dxA

  v = np.concatenate((xv, thetav))
  return np2tf(v)

# return inverse part covariance in degenerate block form
def Ei_tf(o, E):
  m = getattr(lie, o.lie)
  Ei = np.linalg.inv(E)
  Ei_tilde = [ np2tf(np.block([
      [ _, np.zeros((m.n-1, 1)) ],
      [ np.zeros((1, m.n-1)), np.zeros((1,1)) ]]))
    for _ in Ei 
  ]
  return Ei_tilde

def Si_tf(o, S):
  m = getattr(lie, o.lie)
  Si = np.linalg.inv(S)
  return [ np2tf(_) for _ in Si ]

def obs_t_tf(yt):
  return np2tf(np.concatenate((yt, np.ones((yt.shape[0],1))), axis=1))

def prior2vec_tf(o):
  m = getattr(lie, o.lie)
  h_x = ('mvnL', np2tf(m.algi(m.logm(o.H_x[1]))), np2tf(o.H_x[2]))
  h_theta = ('mvnL', np2tf(m.algi(m.logm(o.H_theta[1]))), np2tf(o.H_theta[2]))
  return h_x, h_theta


def np2tf(x, dtype=tf.float32): return tf.convert_to_tensor(x, dtype=dtype)
def alg(G, v): return tf.tensordot(v, G, 1)
def ex(v): return tf.linalg.expm(v)
def generators_tf(o): return np2tf(getattr(lie, o.lie).G)
def obs_tf(y):
  return [np2tf(np.concatenate((yt, np.ones((yt.shape[0],1))), axis=1))
    for yt in y]
def params_tf(o, E, S, Q):
  m = getattr(lie, o.lie)
  Ei, Si, Qi = [ np.linalg.inv(_) for _ in (E, S, Q) ]
  Ei_tilde = [ np2tf(np.block([
      [ _, np.zeros((m.n-1, 1)) ],
      [ np.zeros((1, m.n-1)), np.zeros((1,1)) ]]))
    for _ in Ei 
  ]
  Si = [ np2tf(_) for _ in Si ]
  return Ei_tilde, Si, np2tf(Qi)
def assoc_tf(z): return [ np2tf(zt) for zt in z ]
def pi_tf(pi): return np2tf(pi)
def zeroH_tf(o):
  if o.dy == 2: return tf.constant([0.0, 0.0, 1.0])
  elif o.dy == 3: return tf.constant([0.0, 0.0, 0.0, 1.0])
  else: assert False, 'Not supported'
