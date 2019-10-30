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

def MakeLabelImgs(y, z, imgs):
  T = len(z)
  if type(imgs[0]) == str: h, w = du.imread(imgs[t]).shape[:2]
  else: h, w = imgs[0].shape[:2]

  labels = np.zeros((T,h,w),dtype=np.int)
  for t in range(T):
    ytInt = y[t].astype(np.int)
    for k in np.unique(z[t]):
      if k < 0: continue
      zk = z[t] == k
      labels[t, ytInt[zk,1], ytInt[zk,0]] = k+1

  return labels

# def mots(results, gt, **kwargs):
def mots(results, gt, **kwargs):
  iou = kwargs.get('iou', 0.3)

  # H = du.load(results)
  H = results

  # gtData = du.load(gt)
  gtData = gt

  G = gtData['labels']
  evalIdx = gtData['gtIdx']
  T = len(H)

  # G = m_{1:N} : ground-truth, N unique id
  # H = h_{1:M} : hypotheses, M unique ID
  gIds = np.setdiff1d(np.unique(G), 0)
  n_gId = len(gIds)

  hIds = np.setdiff1d(np.unique(H), 0)
  n_hId = len(hIds)

  IoU = np.zeros((len(evalIdx), n_gId, n_hId))
  c = [ dict() for t in range(len(evalIdx)) ] # h -> g
  ci = [ dict() for t in range(len(evalIdx)) ] # g -> h

  # Compute per-timestep matching 
  for cnt, t in enumerate(evalIdx):
    for n in range(n_gId):
      for m in range(n_hId):
        target = G[t] == gIds[n]
        prediction = H[t] == hIds[m]

        intersection = np.logical_and(target, prediction)
        union = np.logical_or(target, prediction)
        IoU[cnt, n, m] = np.sum(intersection) / np.sum(union)
    
    for m in range(n_hId):
      maxIoU = np.max(IoU[cnt,:,m])
      if maxIoU >= iou:
        ind = np.argmax(IoU[cnt,:,m])
        c[cnt][hIds[m]] = gIds[ind]

    # c^{-1}
    ci[cnt] = dict(reversed(item) for item in c[cnt].items())

  tp, fp, fn = (0, 0, 0)
  tilde_tp = 0
  for cnt, t in enumerate(evalIdx):
    tp += len(c[cnt])
    for h, g in c[cnt].items():
      hIdx = np.where(hIds == h)[0][0]
      gIdx = np.where(gIds == g)[0][0]
      # hIdx = np.argmin(hIds == h)
      # gIdx = np.argmin(gIds == g)
      tilde_tp += IoU[cnt][gIdx, hIdx]

    for h in hIds:
      if c[cnt].get(h, None) is None: fp += 1
    for g in gIds:
      if ci[cnt].get(g, None) is None: fn += 1

  ids = 0
  for cnt, t in enumerate(evalIdx):
    if cnt == 0: continue # no predecessor => no IDS   

    for g in gIds:
      # conditions from Equation (1) in MOTS paper
      id_now = ci[cnt].get(g, None)
      cond1 = id_now is not None # c^{-1}(m) != \emptyset
      cond2 = True # pred(m) != \emptyset

      id_prev = ci[cnt-1].get(g, None)
      cond3 = id_prev is not None

      cond4 = id_now != id_prev

      if cond1 and cond2 and cond3 and cond4: ids += 1

  motsa = (tp - fp - ids) / n_gId
  motsp = tilde_tp / tp
  s_motsa = (tilde_tp - fp - ids) / n_gId

  # def show(t):
  #   im = np.vstack((H[t], gt[t]))
  #   plt.imshow(im)
  # du.ViewPlots(range(T), show)
  # plt.show()

  return tp, fp, fn, ids, tilde_tp, motsa, motsp, s_motsa
