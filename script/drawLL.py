from npp import SED, drawSED, tmu
import argparse, du, numpy as np, matplotlib.pyplot as plt
from os.path import isdir
import os
import IPython as ip

def getSampleLL(sampleFile):
  o, alpha, z, pi, theta, E, S, x, Q, omega, mL, ll, subsetIdx, dataset = \
    SED.loadSample(sampleFile)
  return ll

def getSampleDatasetName(sampleFile):
  o, alpha, z, pi, theta, E, S, x, Q, omega, mL, ll, subsetIdx, dataset = \
    SED.loadSample(sampleFile)
  return du.fileparts(dataset)[1]

def main(args):
  samples = du.GetFilePaths(f'{args.resultPath}', 'gz')
  T = len(samples)
  if args.last == -1: args.last = T

  samples = samples[args.first:args.last:args.step]
  nSamples = len(samples)

  ll = np.array(du.ParforT(getSampleLL, samples))
  x = np.arange(args.first, args.last, args.step)
  plt.plot(x, ll)

  if not args.no_title:
    title = getSampleDatasetName(samples[0])
    plt.title(title)
  plt.xlabel('Samples'); plt.ylabel('Log Joint')
  
  if args.save: plt.savefig(args.save, dpi=300, bbox_inches='tight')
  else: plt.show()

  
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='')
  parser.add_argument('resultPath', type=str, help='directory with .gz sample files')
  parser.add_argument('--save', type=str, default='', help='filepath to save plot to')
  parser.add_argument('--first', type=int, default=0, help='first sample index to plot')
  parser.add_argument('--last', type=int, default=-1, help='last sample idnex to plot')
  parser.add_argument('--step', type=int, default=1, help='sample step parameter')
  parser.add_argument('--no_title', action='store_true', help="don't draw title")
  parser.set_defaults(func=main)

  args = parser.parse_args()
  args.func(args)
