import numpy as np, matplotlib.pyplot as plt
import argparse
import du, du.stats
from npp import SED, drawSED
from tqdm import tqdm
import os

def main(args):
  # load previous sample
  o, alpha, z, pi, theta, E, S, x, Q, omega, mL, llInit, subsetIdx, dataset = \
    SED.loadSample(args.initialSample)

  # recursively make output path ignoring if it's already been created
  try: os.makedirs(args.outdir)
  except: pass

  # load data
  data = du.load(f'{dataset}/data')
  yAll = data['y']
  if subsetIdx is not None: y = [yt[subsetIdx[t]] for t, yt in enumerate(yAll)]
  else: y = yAll

  print(f'Initializing, LL: {llInit:.2f}, K: {len(pi)-1}')

  ll = np.zeros(args.nSamples) 

  # rjmcmc moves (todo: make as args)
  # pBirth, pDeath, pSwitch = (0.1, 0.1, 0.0)
  # pBirth, pDeath, pSwitch = (0.0, 0.0, 0.0)
  pBirth, pDeath, pSwitch = (args.pBirth, args.pDeath, args.pSwitch)

  # rjmcmc proposal tracking
  nBirthProp, nBirthAccept, nDeathProp, nDeathAccept = (0, 0, 0, 0)
  nSwitchProp, nSwitchAccept = (0, 0)

  sampleRng = range(args.firstSampleIndex, args.firstSampleIndex+args.nSamples)
  for cnt, nS in enumerate(sampleRng):
    z, pi, theta, E, S, x, Q, omega, mL, move, accept = SED.sampleRJMCMC(o, y,
      alpha, z, pi, theta, E, S, x, Q, omega, mL, pBirth, pDeath, pSwitch)
    ll[cnt] = SED.logJoint(o, y, z, x, theta, E, S, Q, alpha, pi, omega, mL)

    if move == 'birth':
      nBirthProp += 1
      if accept: nBirthAccept += 1
    elif move == 'death':
      nDeathProp += 1
      if accept: nDeathAccept += 1
    elif move == 'switch':
      nSwitchProp += 1
      if accept: nSwitchAccept += 1

    a = '+' if accept == True else '-'
    if not args.silent:
      print(
        f'Iter {nS:05}, LL: {ll[cnt]:.2f}, K: {len(pi)-1}, Move: {move[0]}{a}')

    if cnt % args.saveEvery == 0:
      filename = f'{args.outdir}/sample-{nS:08}'
      SED.saveSample(filename, o, alpha, z, pi, theta, E, S, x, Q, omega, mL,
        ll[cnt], subsetIdx, dataset)



if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='')
  parser.add_argument('initialSample', type=str, help='path to initial sample')
  parser.add_argument('outdir', type=str, help='path to store results')
  parser.add_argument('nSamples', type=int, help='number of samples to draw')
  parser.add_argument('--saveEvery', type=int, default=1,
    help='interval to save samples')
  parser.add_argument('--firstSampleIndex', type=int, default=1,
    help='first sample index, this number given to first saved sample')

  parser.add_argument('--pBirth', type=float, default=0.0,
    help='Birth move probability')
  parser.add_argument('--pDeath', type=float, default=0.0,
    help='Death move probability')
  parser.add_argument('--pSwitch', type=float, default=0.0,
    help='Switch move probability')

  parser.add_argument('--silent', action='store_true',
    help="don't print sampler status")

  parser.set_defaults(func=main)

  args = parser.parse_args()
  args.func(args)
