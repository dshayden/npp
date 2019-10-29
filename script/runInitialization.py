import numpy as np, matplotlib.pyplot as plt
import argparse
import du, du.stats
import lie
from npp import SED, drawSED, icp
from tqdm import tqdm
import os
import IPython as ip

def main(args):
  data = du.load(f'{args.dataset_path}/data')
  yAll = data['y']
  T = len(yAll)
  ts = range(T)

  if args.maxObs > 0:
    subsetIdx = [ np.random.choice(
      range(len(yt)), min(args.maxObs, len(yt)), replace=False)
      for yt in yAll
    ]
    y = [ yt[subsetIdx[t]] for t, yt in enumerate(yAll) ]
  else:
    subsetIdx = [ np.arange(len(yt)) for yt in yAll ]
    y = yAll

  mL = [ args.mL * np.ones(y[t].shape[0]) for t in range(T) ]
  if args.se3: o = SED.opts(lie='se3')
  else: o = SED.opts(lie='se2')
  SED.initPriorsDataDependent(o, y,
    dfQ=args.dfQ, rotQ=args.rotQ, dfS=args.dfS, rotS=args.rotS,
    dfE=args.dfE, scale=args.scaleE, rotX=args.rotX
  )
  x_ = SED.initXDataMeans(o, y)
  Q = SED.inferQ(o, x_)
  theta_, E, S, z, pi = SED.initPartsAndAssoc(o, y[:1], x_, args.alpha, mL,
    maxBreaks=args.maxBreaks, nInit=args.nInit, nIter=args.nIter,
    tInit=args.tInit, fixedBreaks=args.fixedBreaks
  )
  K = len(pi) - 1

  # get parts and global for all time
  theta = np.zeros((T, K) + o.dxGm)
  theta[0] = theta_[0]
  x = np.zeros((T,) + o.dxGm)
  x[0] = x_[0]

  m = getattr(lie, o.lie)

  def estimate_global_then_parts(tPrev, tNext):
    # 0 -> 1, 1 -> 2, ...
    yPrev, yNext = ( y[tPrev], y[tNext] )
    xPrev = x[tPrev]
    thetaPrev = theta[tPrev]

    # Initialize relative body transformation from tPrev -> tNext as
    # translation between observation means (with no rotation).
    muDiff = np.mean(yNext, axis=0) - np.mean(yPrev, axis=0)
    Q_t0 = SED.MakeRd(np.eye(o.dy), muDiff)
    q_t0 = m.algi(m.logm(Q_t0))

    # Estimate body frame
    Q_t = icp.optimize_global(o, yNext, xPrev, thetaPrev, E, q_t=q_t0)
    x[tNext] = xNext = xPrev @ Q_t
    
    # Estimate parts with body frame
    S_t = icp.optimize_local(o, yNext, xNext, thetaPrev, E, S)
    for k in range(K): theta[tNext,k] = theta[tPrev,k] @ S_t[k]

  # iterate through t, t+1 to get x, theta estimates
  for tPrev, tNext in zip(ts[:-1], ts[1:]):
    estimate_global_then_parts(tPrev, tNext)

  # Specially handle t=0
  estimate_global_then_parts(0, 0)

  # Re-estimate z, pi; remove unused parts (if any)
  z = [ [] for t in range(T) ]
  for t in range(T):
    z[t] = SED.inferZ(o, y[t], pi, theta[t], E, x[t], mL[t])
  z, pi, theta, E, S = SED.consolidatePartsAndResamplePi(o, z, pi, args.alpha,
    theta, E, S)
  
  # Re-estimate Q, S, E
  Q = SED.inferQ(o, x)
  if K > 0: S = np.array([ SED.inferSk(o, theta[:,k]) for k in range(K) ])
  else: S = np.zeros((0, o.dxA, o.dxA))
  E = SED.inferE(o, x, theta, y, z)

  # Compute log-likelihood
  ll = SED.logJoint(o, y, z, x, theta, E, S, Q, args.alpha, pi, mL)

  # Save results
  SED.saveSample(args.outfile, o, args.alpha, z, pi, theta, E, S, x, Q, mL, ll,
    subsetIdx, args.dataset_path)

  # drawSED.draw(o, y=y, z=z, x=x, theta=theta, E=E)
  # plt.show()

  # ip.embed()

  
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='')

  # IO arguments
  parser.add_argument('dataset_path', type=str, help='dataset path')
  parser.add_argument('outfile', type=str, help='output file')

  # DP init argumets
  parser.add_argument('--maxBreaks', type=int, default=20, help='# max part breaks')
  parser.add_argument('--nInit', type=int, default=5, help='# max part breaks')
  parser.add_argument('--nIter', type=int, default=500, help='# dp initialization iterations')
  parser.add_argument('--tInit', type=int, default=0, help='Time to initialize parts on')
  parser.add_argument('--fixedBreaks', action='store_true', help='Force # of parts')
  parser.add_argument('--alpha', type=float, default=0.1, help='Concentration parameter')

  # Body/part prior arguments
  parser.add_argument('--dfQ', type=float, default=10.0, help='Q prior DoF')
  parser.add_argument('--rotQ', type=float, default=15.0,
    help='Q prior expected per-time rotation (in degrees)')
  parser.add_argument('--dfS', type=float, default=10.0, help='S prior DoF')
  parser.add_argument('--rotS', type=float, default=1.5,
    help='S prior expected per-time rotation (in degrees)')
  parser.add_argument('--dfE', type=float, default=10.0, help='E prior DoF')
  parser.add_argument('--scaleE', type=float, default=1.0,
    help='Percentage of mean dataset variance for part covariance prior.')
  parser.add_argument('--rotX', type=float, default=1.0,
    help='Expected initial body rotation (in degrees)')

  # Additional arguments
  parser.add_argument('--mL', type=float, default=-14.0,
    help='Approximate marginal likelihood with a constant')
  parser.add_argument('--se3', action='store_true',
    help='Use se3 dynamics')
  parser.add_argument('--maxObs', type=int, default=3000,
    help='Max number observations to view, 0 for all')

  parser.set_defaults(func=main)

  args = parser.parse_args()
  args.func(args)
