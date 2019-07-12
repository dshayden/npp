import numpy as np, matplotlib.pyplot as plt
import argparse
import du, du.stats
from npp import SED, drawSED
from tqdm import tqdm
import os

def main(args):
  dataset = args.dataset

  data = du.load(f'data/{dataset}/data')
  yAll = data['y']
  if args.maxObs > 0:
    y = [ yt[np.random.choice(range(len(yt)), min(args.maxObs, len(yt)), replace=False)] for yt in yAll ]
  else:
    y = yAll

  T = len(y)

  nSamples = args.nSamples
  alpha = args.alpha 
  nParticles = args.nParticles
  o = SED.opts()

  du.tic()
  SED.initPriorsDataDependent(o, y)
  x = SED.initXDataMeans(o, y)
  theta_, E_, S_ = SED.sampleKPartsFromPrior(o, T, nParticles)

  if nParticles == 0: mL = [ args.mL * np.ones(y[t].shape[0]) for t in range(T) ]
  else: mL = SED.logMarginalPartLikelihoodMonteCarlo(o, y, x, theta_, E_, S_)
  
  # parametric initialization
  theta, E, S, z, pi = SED.initPartsAndAssoc(o, y, x, alpha, mL,
    fixedBreaks=False, maxBreaks=args.nParts, nInit=5, nIter=500)
  Q = SED.inferQ(o, x)
  print(f'Init took {du.toc():.2f} seconds')

  try: os.makedirs(f'results/{dataset}/{args.runId}')
  except: None

  noNewPartBefore = args.noNewPartBefore
  newPartStep = args.newPartStep
  minNonAssoc = args.minNonAssoc

  ll = np.zeros(nSamples)
  for nS in range(nSamples):
    newPart = True if nS >= noNewPartBefore and nS % newPartStep == 0 else False

    z, pi, theta, E, S, x, Q, ll[nS] = SED.sampleStepFC(o, y, alpha, z, pi,
      theta, E, S, x, Q, mL, newPart=newPart, minNonAssoc=minNonAssoc)
    print(f'Run {args.runId}, Sample {nS:05}, #Parts: {theta.shape[1]:02}, ll: {ll[nS]:.2f}')
    filename = f'results/{dataset}/{args.runId}/sample-{nS:05}'
    SED.saveSample(filename, o, alpha, z, pi, theta, E, S, x, Q, ll[nS])

    # need to recompute mL each iteration
    if nParticles > 0:
      mL = SED.logMarginalPartLikelihoodMonteCarlo(o, y, x, theta_, E_, S_)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='')
  parser.add_argument('dataset', type=str, help='dataset name and path')
  parser.add_argument('runId', type=str, help='directory name to store samples')
  parser.add_argument('nParts', type=int, help='number of initialized parts')
  parser.add_argument('--nSamples', type=int, default=2000, help='number of samples drawn')
  parser.add_argument('--alpha', type=float, default=0.1, help='concentration')
  parser.add_argument('--nParticles', type=int, default=100,
    help='marginal likelihood #particles')
  parser.add_argument('--noNewPartBefore', type=int, default=1,
    help='No part proposals before this sample step')
  parser.add_argument('--newPartStep', type=int, default=1,
    help='Sample stride for allowing new parts')
  parser.add_argument('--minNonAssoc', type=int, default=1,
    help='Min number of unassociated observations to allow new part proposals')
  parser.add_argument('--mL', type=float, default=-7.0,
    help='Approximate marginal likelihood with a constant')
  parser.add_argument('--maxObs', type=int, default=0,
    help='Max number observations to view, 0 for all')

  parser.set_defaults(func=main)

  args = parser.parse_args()
  args.func(args)
