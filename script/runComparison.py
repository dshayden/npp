import numpy as np, matplotlib.pyplot as plt
import argparse
import du, du.stats
from npp import SED, drawSED, evalSED
import IPython as ip
import os

def main(args):
  # load previous sample
  o, alpha, z, pi, theta, E, S, x, Q, omega, mL, ll, subsetIdx, dataset = \
    SED.loadSample(args.sample)
  K = len(pi)-1
  assert dataset

  # load ground-truth labels
  gtData = du.load(f'{dataset}/gtLabels')
  gtLabels, gtIdx = ( gtData['labels'], gtData['gtIdx'] )
  T, h, w = gtLabels.shape

  # load dataset, take max assignments
  data = du.load(f'{dataset}/data')
  y = data['y']

  # make mL so it is the right size
  mL_const = np.mean([np.mean(mL[t]) for t in range(T)])
  mL = [ mL_const*np.ones(y[t].shape[0]) for t in range(T) ]
  z = [ SED.inferZ(o, y[t], pi, theta[t], E, x[t], omega, mL[t], max=True)
    for t in range(T) ]
  
  # precompute image indices for se3
  if o.lie == 'se3':
    yy, xx = np.meshgrid(range(h), range(w), indexing='ij')
    xy = np.stack((xx.flatten(),yy.flatten()), axis=1) # N x 2

  # construct stacked label images for sample  
  sampleLabels = np.zeros((T, h, w)) 
  for t in range(T):

    if o.lie == 'se3':
      idx, mask = ( data['idx'][t], data['mask'][t] )

      # idx_t = data['idx'][t] # indexes into xy
      # mask = data['mask'][t]
      xyFG = xy[mask.flatten()]
      xyFG = xyFG[ idx ] # lines up with z labels now

      xs, ys = xyFG.T     
    else:
      xs, ys = y[t].T.astype(np.int)

    for k in range(K):
      ztk = z[t]==k
      sampleLabels[t, ys[ztk], xs[ztk]] = k+1

  # du.ViewImgs(sampleLabels)
  # ip.embed()
  # sys.exit()

  # Have groundtruth and label images, call comparison
  iou = args.iou
  tp, fp, fn, ids, tilde_tp, motsa, motsp, s_motsa = \
    evalSED.mots(sampleLabels, gtData, iou=iou)

  name = du.fileparts(dataset)[1]
  header1 = f'{name}, iou: {iou:.1f}'
  header2 = 'tp & fp & fn & ids & tilde_tp & motsa & motsp & smotsa \\\\'
  values = f'{tp} & {fp} & {fn} & {ids:03} & {tilde_tp:.2f} & {motsa:.2f} & {motsp:.2f} & {s_motsa:.2f}'
  print(header1)
  print(header2)
  print(values)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='')
  parser.add_argument('sample', type=str, help='path to sample')
  parser.add_argument('--iou', type=float, default=0.5,
    help='IoU comparison')

  parser.set_defaults(func=main)

  args = parser.parse_args()
  args.func(args)
