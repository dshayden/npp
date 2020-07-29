from npp import SED, drawSED, tmu
import trimesh as tm
import argparse, du, numpy as np, matplotlib.pyplot as plt
from os.path import isdir
import os
import lie

def main(args):

  if args.sampleIdx is not None:
    samples = du.GetFilePaths(f'{args.resultPath}', 'gz')
    o, alpha, z, pi, theta, E, S, x, Q, omega, mL, ll, subsetIdx, datasetPath = \
      SED.loadSample(samples[int(args.sampleIdx)])
  else:
    o, alpha, z, pi, theta, E, S, x, Q, omega, mL, ll, subsetIdx, datasetPath = \
      SED.loadSample(args.resultPath)

  data = du.load(f'{datasetPath}/data')
  yAll = data['y']
  m = getattr(lie, o.lie)

  T = len(z)
  K = theta.shape[1]

  if args.drawMesh:
    meshFiles = du.GetFilePaths(f'{datasetPath}/mesh', 'obj')[:T]
    mesh = du.Parfor(tm.load, meshFiles)
    y = [ mesh[t].vertices for t in range(T) ]
    mL_const = np.mean([np.mean(mL[t]) for t in range(T)])
    mL = [ mL_const*np.ones(y[t].shape[0]) for t in range(T) ]

    for t in range(T):
      z[t] = SED.inferZ(o, y[t], pi, theta[t], E, x[t], omega, mL[t], max=args.maxZ)
  
  elif args.no_resampleZ:
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

    for t in range(T):
      z[t] = SED.inferZ(o, y[t], pi, theta[t], E, x[t], omega, mL[t], max=args.maxZ)

  # omegaTheta
  if args.omega:
    theta = np.tile(omega, (T,1,1,1))
  else:
    for t in range(T):
      for k in range(K):
        theta[t,k] = omega[k] @ theta[t,k]


  if args.noE: E = None
  if args.noTheta: theta = None
  if args.noZ: z = None
  noX = args.noX
  if args.noTitle: title = [ f'' for t in range(T) ]
  else: title = [ f'{t:05}' for t in range(T) ]

  if args.decimate > 0:
    nPts = args.decimate
    print('decimate')
    for t in range(T):
      decimateIdx = np.random.choice(range(len(y[t])), nPts)
      y[t] = y[t][decimateIdx]
      if z is not None: z[t] = z[t][decimateIdx]

  if args.wiggle:
    from scipy.stats import multivariate_normal as mvn
    for t in range(T):
      y[t] += mvn.rvs(np.zeros(3), args.wiggle_eps*np.eye(3), size=y[t].shape[0])
    # if theta is not None: theta[0,0][1,3] += 0.2

  if args.save:
    try: os.makedirs(args.save)
    except: pass
    fnames = [ f'{args.save}/img-{t:05}.png' for t in range(T) ]
  else:
    fnames = None

  def getImgs(path):
    imgPaths = du.GetImgPaths(imgPath)
    if len(imgPaths) > 100:
      du.imread(imgPaths[0]); imgs = du.ParforT(du.imread, imgPaths)
    else:
      imgs = du.For(du.imread, imgPaths)
    return imgs
  
  # three cases
  #   se2, se3 + draw2d, se3
  if o.lie == 'se2':
    imgPath = f'{datasetPath}/imgs'
    if isdir(imgPath): imgs = getImgs(imgPath)
    else: imgs = None

    scenes_or_none = drawSED.draw(o, y=y, x=x, theta=theta, E=E, img=imgs, z=z,
      filename=fnames, noX=noX, title=title)
    if not args.save: plt.show()

  elif o.lie == 'se3' and not args.draw2d and not args.drawMesh:
    scenes = drawSED.draw(o, y=y, x=x, theta=theta, E=E, z=z, noX=noX, title=title)
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

      if args.single_frame:
        t = 0
        tmu.show_scene_with_bindings(scenes[t], res=[1920,1080])
      else:
        for t in range(T): tmu.show_scene_with_bindings(scenes[t], res=[1920,1080])

  elif o.lie == 'se3' and args.drawMesh:
    meshFiles = du.GetFilePaths(f'{datasetPath}/mesh', 'obj')[:T]
    mesh = du.Parfor(tm.load, meshFiles)

    mL_const = np.mean([np.mean(mL[t]) for t in range(T)])
    zCols = (255*du.diffcolors(100, bgCols=[[1,1,1],[0,0,0]], alpha=1.0)).astype(np.uint8)
    zColsFloat = du.diffcolors(100, bgCols=[[1,1,1],[0,0,0]], alpha=1.0)

    for t in range(T):
      mesh[t].visual.vertex_colors = zCols[z[t]]

    # draw but don't render scenes just to get camera
    scenes = drawSED.draw(o, y=y)
    transform = tmu.CameraFromScenes(scenes)

    # same colors as drawSED
    for t in range(T):
      scene = tm.scene.Scene()
      scene.add_geometry(mesh[t])

      # scene = mesh[t]
      transform_t = scene.camera.transform.copy()
      transform_t[2,3] = transform[2,3] * 1.5
      transform_t[2,2] = transform[2,2]
      scene.camera.transform = transform_t

      if not args.noX:
        scene.add_geometry(tmu.MakeAxes(0.2, x[t], np.tile([0, 0, 0, 255],
          [4,1]).astype(np.int), minor=0.01))


      if not args.orbit:
        if args.save:
          tmu.save_render(scene, fnames[t], res=[1920,1080])
        else:
          scene.show()
      else:
        nCams = 8
        cams = tmu.MakeCamerasOrbit(scene, nCams)

        for i in range(nCams):
          print('Time {t}, Cam {i}')
          cam = cams[i] @ transform_t
          scene.camera.transform = cam
          fname = f'{args.save}/camera-{i:02}-img-{t:05}.png' 
          tmu.save_render(scene, fname, res=[1920,1080])
    
    # re-associate to mesh vertices

  elif o.lie == 'se3' and args.draw2d:
    imgPath = f'{datasetPath}/rgb'
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
      # ip.embed()
      xyFG = xyFG[ idx_t ]
      yImg.append(xyFG)
    
    # make fake options with se2 for display
    oImg = SED.opts(lie='se2')
    drawSED.draw(oImg, y=yImg, z=z, img=imgs, filename=fnames, noX=noX, title=title)
    if not args.save: plt.show()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='')
  parser.add_argument('resultPath', type=str, help='result file / path')
  parser.add_argument('--sampleIdx', help='sample index')
  parser.add_argument('--noE', action='store_true',
    help="don't draw part covariances")
  parser.add_argument('--noX', action='store_true',
    help="don't draw object coordinate frame")
  parser.add_argument('--noTheta', action='store_true',
    help="don't draw part frames of reference")
  parser.add_argument('--noZ', action='store_true',
    help="don't draw associations")
  parser.add_argument('--no_resampleZ', action='store_true',
    help="don't resample z")
  parser.add_argument('--maxZ', action='store_true',
    help="Take argmax assignments")
  parser.add_argument('--omega', action='store_true',
    help="draw canonical parts (static across time)")
  parser.add_argument('--draw2d', action='store_true',
    help="Draw 3d on image")
  parser.add_argument('--save', type=str, default='',
    help="Save plots to directory given by argument; draw to screen if empty")
  parser.add_argument('--noTitle', action='store_true', help="Empty titles")
  parser.add_argument('--drawMesh', action='store_true', help="Draw mesh for 3d data")
  parser.add_argument('--orbit', action='store_true', help="Draw mesh for 3d data")

  # only implemented for 3D non-mesh viewing
  parser.add_argument('--single_frame', action='store_true', help="Just show first frame")
  parser.add_argument('--decimate', type=int, default=-1, help="Reduce number of data points to this many")
  parser.add_argument('--wiggle', action='store_true', help="wiggle data points")
  parser.add_argument('--wiggle_eps', type=float, default=0.00001, help="wiggle noise")
  parser.set_defaults(func=main)

  args = parser.parse_args()
  args.func(args)
