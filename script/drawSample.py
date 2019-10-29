from npp import SED, drawSED
import argparse, du, numpy as np, matplotlib.pyplot as plt
import os

def main(args):
  yAll = du.load(f'{args.datasetPath}/data')['y']

  imgPath = f'{args.datasetPath}/imgs'
  if os.path.isdir(imgPath):
    imgPaths = du.GetImgPaths(imgPath)
    if len(imgPaths) > 100:
      du.imread(imgPaths[0])
      imgs = du.ParforT(du.imread, imgPaths)
    else:
      imgs = du.For(du.imread, imgPaths)
  else:
    imgs = None

  samples = du.GetFilePaths(f'{args.resultPath}', 'gz')

  o, alpha, z, pi, theta, E, S, x, Q, mL, ll, subsetIdx, dataset = \
    SED.loadSample(samples[args.sampleIdx])
  T = len(z)

  if args.no_resampleZ:
    if subsetIdx is not None:
      y = [yt[subsetIdx[t]] for t, yt in enumerate(yAll)]
    else:
      y = yAll
  else:
    # don't use subsetIdx
    y = yAll
    mL_const = np.mean([np.mean(mL[t]) for t in range(T)])
    mL = [ mL_const*np.ones(y[t].shape[0]) for t in range(T) ]

    if args.maxZ: max=True
    else: max=False
    z = [ SED.inferZ(o, y[t], pi, theta[t], E, x[t], mL[t], max=max)
      for t in range(T) ]

  if args.noE: E = None
  if args.noTheta: theta = None
  if args.noZ: z = None

  if args.save: fnames = [ f'{args.save}/img-{t:05}.png' for t in range(T) ]
  else: fnames = None

  drawSED.draw(o, y=y, x=x, theta=theta, E=E, img=imgs, z=z, filename=fnames)
  if not args.save: plt.show()

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='')
  parser.add_argument('datasetPath', type=str, help='dataset path')
  parser.add_argument('resultPath', type=str, help='result path')
  parser.add_argument('--sampleIdx', type=int, default=-1, help='sample index')
  parser.add_argument('--noE', action='store_true',
    help="don't draw part covariances")
  parser.add_argument('--noTheta', action='store_true',
    help="don't draw part frames of reference")
  parser.add_argument('--noZ', action='store_true',
    help="don't draw associations")
  parser.add_argument('--no_resampleZ', action='store_true',
    help="don't resample z")
  parser.add_argument('--maxZ', action='store_true',
    help="Take argmax assignments")
  parser.add_argument('--save', type=str, default='',
    help="Save plots to directory given by argument; draw to screen if empty")
  parser.set_defaults(func=main)

  args = parser.parse_args()
  args.func(args)
