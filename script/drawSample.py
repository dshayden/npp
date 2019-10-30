from npp import SED, drawSED, tmu
import argparse, du, numpy as np, matplotlib.pyplot as plt
from os.path import isdir
import os
import IPython as ip

def main(args):
  data = du.load(f'{args.datasetPath}/data')
  yAll = data['y']

  # if isdir(f'{args.datasetPath}/imgs'): imgPath = f'{args.datasetPath}/imgs'
  # elif isdir(f'{args.datasetPath}/rgb'): imgPath = f'{args.datasetPath}/rgb'
  # else: imgPath = ''
  # if os.path.isdir(imgPath):
  #   imgPaths = du.GetImgPaths(imgPath)
  #   if len(imgPaths) > 100:
  #     du.imread(imgPaths[0])
  #     imgs = du.ParforT(du.imread, imgPaths)
  #   else:
  #     imgs = du.For(du.imread, imgPaths)
  # else:
  #   imgs = None

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

  def getImgs(path):
    imgPaths = du.GetImgPaths(imgPath)
    if len(imgPaths) > 100:
      du.imread(imgPaths[0]); imgs = du.ParforT(du.imread, imgPaths)
    else:
      imgs = du.For(du.imread, imgPaths)
    return imgs
  
  # if isdir(f'{args.datasetPath}/imgs'): imgPath = f'{args.datasetPath}/imgs'
  # elif isdir(f'{args.datasetPath}/rgb'): imgPath = f'{args.datasetPath}/rgb'
  # else: imgPath = ''
  # if os.path.isdir(imgPath):
  #   imgPaths = du.GetImgPaths(imgPath)
  #   if len(imgPaths) > 100:
  #     du.imread(imgPaths[0])
  #     imgs = du.ParforT(du.imread, imgPaths)
  #   else:
  #     imgs = du.For(du.imread, imgPaths)
  # else:
  #   imgs = None


  # three cases
  #   se2, se3 + draw2d, se3
  if o.lie == 'se2':
    imgPath = f'{args.datasetPath}/imgs'
    if isdir(imgPath): imgs = getImgs(imgPath)
    else: imgs = None

    scenes_or_none = drawSED.draw(o, y=y, x=x, theta=theta, E=E, img=imgs, z=z,
      filename=fnames)
    if not args.save: plt.show()

  elif o.lie == 'se3' and not args.draw_3d_as_2d:
    scenes = drawSED.draw(o, y=y, x=x, theta=theta, E=E, z=z)
    transform = tmu.CameraFromScenes(scenes)

    # set transform as 1.25 of min z. This is specific to se3_marmoset for now
    #   multiply instead of divide because maybe camera is looking backwards?
    for scene in scenes:
      transform_t = scene.camera.transform.copy()
      transform_t[2,3] = transform[2,3] * 1.25
      scene.camera.transform = transform_t

    if args.save:
      for t in range(T): tmu.save_render(scenes[t], fnames[t])
    else:
      for t in range(T): scenes[t].show()

  elif o.lie == 'se3' and args.draw_3d_as_2d:
    imgPath = f'{args.datasetPath}/rgb'
    assert isdir(imgPath)
    imgs = getImgs(imgPath)

    yImg = [ ]

    for t in range(T):
      mask = data['mask'][t]

      # get ordered image indices
      h, w = imgs[0].shape[:2]
      yy, xx = np.meshgrid(range(h), range(w), indexing='ij')
      xy = np.stack((xx.flatten(),yy.flatten()), axis=1) # N x 2
      xyFG = xy[mask.flatten()]
      idx_t = data['idx'][t] # indexes into xy

      if subsetIdx is not None and args.no_resampleZ:
        subsetIdx_t = subsetIdx[t]
        idx_t = idx_t[subsetIdx_t]
      xyFG = xyFG[ idx_t ]
      yImg.append(xyFG)
    
    # make fake options with se2 for display
    oImg = SED.opts(lie='se2')
    drawSED.draw(oImg, y=yImg, z=z, img=imgs, filename=fnames)
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
  parser.add_argument('--draw_3d_as_2d', action='store_true',
    help="Draw 3d on image")
  parser.add_argument('--save', type=str, default='',
    help="Save plots to directory given by argument; draw to screen if empty")
  parser.set_defaults(func=main)

  args = parser.parse_args()
  args.func(args)
