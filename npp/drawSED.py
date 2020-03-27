import numpy as np
import matplotlib.pyplot as plt
import du, du.stats, lie
from . import tmu
from . import SED
import logging
logging.getLogger("trimesh").setLevel(logging.ERROR) # stop annoying trimesh logging
import trimesh as tm

def draw_t_SE3(o, **kwargs):
  """ Draw or save model drawing for given time t.

  INPUT
    o (Namespace): options

  KEYWORD INPUT
    y (ndarray, [N, m.n]): observations
    x (ndarray, [dxGf,]): latent global state
    z (ndarray, [N_t,]): Associations
    zCols (ndarray, [K,3/4]): Association Colors
    xTrue (ndarray, [dx,]): true latent global state
    theta (ndarray, [K_est, dt]): latent part state
    thetaTrue (ndarray, [K, dt]): latent part state
    E (ndarray, [K_est, dy]): observation noise extent
    ETrue (ndarray, [K, dy]): true observation noise extent
    xlim: min, max x plot limits
    ylim: min, max y plot limits
    title: plot title name
    filename: save path if desired, else returns figure reference
    style: string for modifying plot style. Currently ignored.
    noX: bool, default False
  """
  assert o.lie == 'se3'
  m = getattr(lie, o.lie)

  zCols = kwargs.get('zCols', None)
  if zCols is None:
    zCols = du.diffcolors(100, bgCols=[[1,1,1],[0,0,0]])
    # zCols[1] = zCols[7]
    # zCols[4] = zCols[11]
    # zColsInt = (zCols*255).astype(np.int)

  scene = tm.scene.Scene()

  y = kwargs.get('y', None)
  if y is not None:
    z = kwargs.get('z', None)
    if z is None:
      pc = tmu.ConstructPointCloud(y)
      scene.add_geometry(pc)
    else:
      instIdx = z>=0
      if np.sum(instIdx) > 0:
        zPart = z[instIdx]
        pc1 = tmu.ConstructPointCloud(y[instIdx], zCols[zPart])
        scene.add_geometry(pc1)

      noInstIdx = z==-1
      nNonInstantiated = np.sum(noInstIdx)
      if nNonInstantiated > 0:
        grayCols = np.tile(0.5*np.ones(3), [nNonInstantiated, 1])
        if nNonInstantiated == 1 and len(y[noInstIdx].shape) == 1:
          yNon = y[noInstIdx][np.newaxis]
        else: yNon = y[noInstIdx]
        pc2 = tmu.ConstructPointCloud(yNon, grayCols)
        scene.add_geometry(pc2)

      # pc = tmu.ConstructPointCloud(y, zCols[z])
    # scene.add_geometry(pc)

  # todo: add part majorAxix (60 / 20 / 2 / 0.2) as a parameter
  # else: marmoset needs ~60 and articulated meshes need ~0.2

  x = kwargs.get('x', None)
  noX = kwargs.get('noX', None)
  if x is not None and not noX:
    scene.add_geometry(tmu.MakeAxes(0.1, du.asShape(x, o.dxGm),
      np.tile([0, 0, 0, 255], [4,1]).astype(np.int), minor=0.01))
    # scene.add_geometry(tmu.MakeAxes(0.2, du.asShape(x, o.dxGm),
    #   np.tile([0, 0, 0, 255], [4,1]).astype(np.int), minor=0.01))

    # scene.add_geometry(tmu.MakeAxes(2.0, du.asShape(x, o.dxGm),
    #   np.tile([0, 0, 0, 255], [4,1]).astype(np.int), minor=0.01))
    # scene.add_geometry(tmu.MakeAxes(60.0, du.asShape(x, o.dxGm),
    #   np.tile([0, 0, 0, 255], [4,1]).astype(np.int), minor=0.01))

  theta = kwargs.get('theta', None)
  if theta is not None and x is not None:
    K = theta.shape[0]
    for k in range(K):
      c = zCols[k]
      if len(c) == 3: c = np.concatenate((c, np.ones(1)))
      c = np.tile( (c*255).astype(np.uint8), [4,1] )

      T_world_part = du.asShape(x, o.dxGm).dot(
        du.asShape(theta[k], o.dxGm))
      
      scene.add_geometry(tmu.MakeAxes(0.1, T_world_part, c, minor=0.01))
      # scene.add_geometry(tmu.MakeAxes(0.2, T_world_part, c, minor=0.01))

      # scene.add_geometry(tmu.MakeAxes(20.0, T_world_part, c, minor=0.01))
      # scene.add_geometry(tmu.MakeAxes(60.0, T_world_part, c, minor=0.01))
      # scene.add_geometry(tmu.MakeAxes(2.0, T_world_part, c, minor=0.01))

  E = kwargs.get('E', None)
  if E is not None and theta is not None and x is not None:
    K = theta.shape[0]
    for k in range(K):
      c = zCols[k]
      if len(c) == 3: c = np.concatenate((c, 0.25*np.ones(1)))
      c = (c*255).astype(np.uint8)

      T_world_part = du.asShape(x, o.dxGm).dot(
        du.asShape(theta[k], o.dxGm))

      # TODO: eigvals not sorted, this is probably bad idea
      # ell = tmu.MakeEllipsoid(T_world_part,
      #   np.sqrt(np.linalg.eigvals(E[k]))*2, c)
      ell = tmu.MakeEllipsoid(T_world_part,
        np.sqrt(np.linalg.eigvals(E[k]))*3, c)
      # setattr(ell, 'wire', True)
      scene.add_geometry(ell)

  filename = kwargs.get('filename', None)
  if filename is not None: tmu.save_render(scene, filename)
  else: return scene


def draw_t_SE2(o, **kwargs):
  """ Draw or save model drawing for given time t.

  INPUT
    o (Namespace): options

  KEYWORD INPUT
    y (ndarray, [N, m.n]): observations
    x (ndarray, dxGm): latent global state
    z (ndarray, [N_t,]): Associations
    zCols (ndarray, [K,3/4]): Association Colors
    theta (ndarray, [K_est, dxGm]): latent part state
    E (ndarray, [K_est, dy, dy]): observation noise extent
    isObserved (ndarray, [K,]): boolean array of whether parts are observed
    xlim: min, max x plot limits
    ylim: min, max y plot limits
    title: plot title name
    filename: save path if desired, else returns figure reference
    style: string for modifying plot style. Currently ignored.
    reverseY: boolean, default False
    noX: boolean, default False
    img (ndarray, [nY, nX, 3]): image to draw on

    linestyle (str): linestyle for x, theta arrows
    deviations (float): deviations to draw E at
  """
  assert o.lie == 'se2'
  m = getattr(lie, o.lie)

  zCols = kwargs.get('zCols', None)
  if zCols is None:
    zCols = du.diffcolors(100, bgCols=[[1,1,1],[0,0,0]])

  reverseY = kwargs.get('reverseY', False)
  noX = kwargs.get('noX', False)
  linestyle = kwargs.get('linestyle', '-')
  deviations = kwargs.get('deviations', 1.25)

  img = kwargs.get('img', None)
  if img is None:
    # fillStyle = kwargs.get('fillStyle', 'full')
    import matplotlib
    marker = matplotlib.markers.MarkerStyle('o', 'full')

    y = kwargs.get('y', None)
    if y is not None:
      z = kwargs.get('z', None)
      if z is None:
        plt.scatter( y[:,0], y[:,1], color='k', s=0.1, zorder=0.1 )
      else:
        assoc = z>=0
        plt.scatter(y[assoc,0], y[assoc,1], color=zCols[z[assoc]], s=0.1,
          zorder=0.1, marker=marker)
        noAssoc = np.logical_not(assoc)
        plt.scatter(y[noAssoc,0], y[noAssoc,1], color='gray', alpha=0.5, s=0.1,
          zorder=0.1, marker=marker)
  else:
    # assume y are foreground indices
    y = kwargs.get('y', None)
    if y is not None:
      z = kwargs.get('z', None)
      if z is not None:
        yInt = y.astype(np.int)
        if zCols.shape[1] == 3:
          zColsA = np.concatenate((zCols, 0.5*np.ones((zCols.shape[0], 1))),
            axis=1)
        else:
          zColsA = zCols

        for k in np.unique(z[z>=0]):
          yk = yInt[z==k]
          img = du.DrawOnImage(img, (yk[:,1], yk[:,0]), zColsA[k])
        gray = np.array([128, 128, 128, 0.5])
        yNoAssoc = yInt[z==-1]
        img = du.DrawOnImage(img, (yNoAssoc[:,1], yNoAssoc[:,0]), gray)
    plt.imshow(img)

  # both x, xTrue are black, need to handle linestyles
  x = kwargs.get('x', None)
  if x is not None and not noX: m.plot(x, np.tile([0,0,0], [2,1]), l=10.0, linestyle=linestyle)

  theta = kwargs.get('theta', None)
  if theta is not None and x is not None:
    K = theta.shape[0]
    isObserved = kwargs.get('isObserved', np.ones(K, dtype=np.bool))

    for k in range(K):
      if not isObserved[k]: continue
      T_world_part = x.dot(theta[k])
      m.plot(T_world_part, np.tile(zCols[k], [2,1]), l=10.0, linestyle=linestyle)

  zero = np.zeros(2)
  E = kwargs.get('E', None)
  if E is not None and theta is not None and x is not None:
    K = theta.shape[0]
    for k in range(K):
      if not isObserved[k]: continue
      T_world_part = x.dot(theta[k])
      yMu = SED.TransformPointsNonHomog(T_world_part, zero)
      R = T_world_part[:-1,:-1]
      ySig = R.dot(E[k]).dot(R.T)
      plt.plot( *du.stats.Gauss2DPoints(yMu, ySig, deviations=deviations), c=zCols[k], linestyle=linestyle)

  ax = plt.gca()
  xlim = kwargs.get('xlim', None)
  if xlim is not None: ax.set_xlim(xlim)
  elif img is not None:
    xlim = (0, img.shape[1])
    ax.set_xlim(xlim)

  ylim = kwargs.get('ylim', None)
  if ylim is not None: ax.set_ylim(ylim)
  elif img is not None:
    ylim = (img.shape[0], 0)
    ax.set_ylim(ylim)

  plt.gca().set_aspect('equal', 'box')
  plt.gca().set_xticks([])
  plt.gca().set_yticks([])
  plt.grid(color=[.5, .5, .5], alpha=0.25)

  title = kwargs.get('title', None)
  if title is not None: plt.title(title)

  if reverseY: plt.gca().invert_yaxis()

  filename = kwargs.get('filename', None)
  if filename is not None:
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
  else:
    return plt.gcf()

def draw_t(o, **kwargs):
  """ Draw or save model for given time t.

  INPUT
    o (Namespace): options

  KEYWORD INPUT
    y (ndarray, [N, m.n]): observations
    x (ndarray, [dxGf,]): latent global state
    z (ndarray, [N_t,]): Associations
    zCols (ndarray, [K,3/4]): Association Colors
    xTrue (ndarray, [dx,]): true latent global state
    theta (ndarray, [K_est, dt]): latent part state
    thetaTrue (ndarray, [K, dt]): latent part state
    E (ndarray, [K_est, dy]): observation noise extent
    ETrue (ndarray, [K, dy]): true observation noise extent
    xlim: min, max x plot limits
    ylim: min, max y plot limits
    title: plot title name
    filename: save path if desired, else returns figure reference
    style: string for modifying plot style. Currently ignored.
    img (ndarray, [nY, nX, 3]): image to draw on
  """
  if o.lie == 'se2': return draw_t_SE2(o, **kwargs)
  elif o.lie == 'se3': return draw_t_SE3(o, **kwargs) 
  else: assert False, 'Only support drawing SE2 or SE3'

def draw(o, **kwargs):
  """ Draw or save model for all times. Calls draw_t

  INPUT
    o (Namespace): options

  KEYWORD INPUT
    y (list of ndarray, [ [N_1, dy], [N_2, dy], ..., [N_T, dy] ]): Observations
    x (ndarray, [T, dx]): latent state
    z (list of ndarray, [ [N_1,], [N_2,], ..., [N_T,] ]): Associations
    zCols (ndarray, [K,3/4]): Association Colors
    theta (ndarray, [T, P, dt]): latent part state
    E (ndarray, [P, dy]): observation noise extent
    title (list, [T, ]): plot title names
    filename (list, [T, ]): save paths if desired, else invoke ViewPlots
    xlim: min, max x plot limits. If None, will be figured out.
    ylim: min, max y plot limits. If None, will be figured out.
    style: string for modifying plot style. One of [None, 'mot_wind']
    no_parallel (bool): don't use parfor
    img (list of ndarray, [nY, nX, 3]): images to draw on
    reverseY (bool): reverse y axis orientation
    noX (bool): don't draw x coordinate frame, even if provided
  """
  # time-varying
  y = kwargs.get('y', None)
  x = kwargs.get('x', None)
  z = kwargs.get('z', None)
  theta = kwargs.get('theta', None)

  title = kwargs.get('title', None)
  filename = kwargs.get('filename', None)
  no_parallel = kwargs.get('no_parallel', False)
  no_extents = kwargs.get('no_extents', False)
  img = kwargs.get('img', None)
  reverseY = kwargs.get('reverseY', False)
  noX = kwargs.get('noX', False)

  if x is None and y is None: return

  if x is not None: T = x.shape[0]
  elif y is not None: T = len(y)

  # single
  E = kwargs.get('E', None)
  zCols = kwargs.get('zCols', None)

  style = kwargs.get('style', None)

  xlim = kwargs.get('xlim', None)
  if xlim is None:
    if y is not None and img is None:
      xmin = np.min([ np.min(y[t][:,0]) for t in range(T) ])
      xmax = np.max([ np.max(y[t][:,0]) for t in range(T) ])
      diff = xmax - xmin
      xlim = (xmin - diff*0.1, xmax + diff*0.1)

  ylim = kwargs.get('ylim', None)
  if ylim is None:
    if y is not None and img is None:
      ymin = np.min([ np.min(y[t][:,1]) for t in range(T) ])
      ymax = np.max([ np.max(y[t][:,1]) for t in range(T) ])
      diff = ymax - ymin
      ylim = (ymax + diff*0.1, ymin - diff*0.1)

  # either save all or use ViewPlots
  if x is None: x = [ None for t in range(T) ]
  if theta is None: theta = [ None for t in range(T) ]
  if y is None: y = [ None for t in range(T) ]
  if z is None: z = [ None for t in range(T) ]
  if img is None: img = [ None for t in range(T) ]

  if title is None: title = [ f'{t:05}' for t in range(T) ]

  if filename is None and o.lie == 'se2':
    def _draw_t(t):
      draw_t(o, x=x[t], theta=theta[t], y=y[t], title=title[t], z=z[t],
        img=img[t], zCols=zCols, E=E, xlim=xlim, ylim=ylim, style=style,
        reverseY=reverseY, noX=noX)
    du.ViewPlots(range(T), _draw_t)
  elif filename is None and o.lie == 'se3':
    def pFunc(o, x, theta, y, title, z, zCols, E, xlim, ylim, style):
      import sys
      sys.path.append('code')
      # import igpSEN_relative as igp
      # scene = igp.draw_t(o, x=x, theta=theta, y=y, title=title, z=z,
      #   zCols=zCols, E=E, xlim=xlim, ylim=ylim, style=style)
      from npp import drawSED, tmu
      scene = drawSED.draw_t(o, x=x, theta=theta, y=y, title=title, z=z,
        zCols=zCols, E=E, xlim=xlim, ylim=ylim, style=style, noX=noX)
      scene.set_camera()
      return scene

    pArgs = [ (o, x[t], theta[t], y[t], title[t], z[t],
      zCols, E, xlim, ylim, style) for t in range(T) ]
    # return du.ParforD( pFunc, pArgs )
    return du.For( pFunc, pArgs, showProgress=False )
  else:
    def pFunc(o, x, theta, y, title, z, img, zCols, E, xlim, ylim, filename, style):
      # import sys
      # # sys.path.append('code')
      # sys.path.append('/data/rvsn/vp/projects/nonparametric_parts_relative/code')
      # import igpSEN_relative as igp
      # import matplotlib as mpl
      # mpl.use('Agg') # draw without x11

      # if no_extents: E = None

      # igp.draw_t(o, x=x, theta=theta, y=y, title=title, z=z, zCols=zCols, E=E,
      #   xlim=xlim, ylim=ylim, filename=filename, style=style)

      draw_t(o, x=x, theta=theta, y=y, title=title, z=z, img=img, zCols=zCols, E=E,
        xlim=xlim, ylim=ylim, filename=filename, style=style, reverseY=reverseY, noX=noX)

    pArgs = [ (o, x[t], theta[t], y[t], title[t], z[t], img[t], zCols, E, xlim,
      ylim, filename[t], style) for t in range(T) ]
    if __name__ == '__main__' and not no_parallel:
      du.ParforD( pFunc, pArgs )
    else:
      du.For( pFunc, pArgs, showProgress=False )
