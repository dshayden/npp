# trimesh utilities and helpers
import npp
import numpy as np
import trimesh as tm, trimesh.viewer
import du
import datetime
import pyglet, pyglet.gl # dependencies of trimesh
import os, pathlib
# import IPython as ip

currentPath = pathlib.Path(os.path.abspath(__file__))
spherePath = currentPath.parents[1] / 'data/sphere/sphere_ico5.obj'
assert spherePath.is_file(), "Can't find data/sphere/sphere_ico5.obj."
sphere = tm.load(str(spherePath))
nSV = sphere.vertices.shape[0]

def LoadObjs(objDir):
  """ Parallel load obj files in meshDir. """
  files = du.GetFilePaths(objDir, 'obj')
  return du.Parfor(tm.load, files, showProgress=False)


def MakeEllipsoid(Tx, scale, color=None):
  """ Make ellipsoidal mesh 

  INPUT
    Tx (ndarray, [4x4]): Transformation to apply
    scale (ndarray, [3,]): Scale x, y z
    color (ndarray, uint8, [3,] or [4,]): 0..255 for rgba

  OUTPUT
    ellipsoid (trimesh.Mesh): Mesh with transformation and vertex coloring
  """
  Txs = Tx.copy()
  # Txs[np.diag_indices(3)] *= scale
  ellipsoid = sphere.copy()

  # import IPython as ip
  # ip.embed()

  scale_transform = np.eye(4)
  for i in range(3): scale_transform[i,i] = scale[i]
  ellipsoid.apply_transform(scale_transform)
  ellipsoid.apply_transform(Tx)
  if color is None: return ellipsoid
  color = np.asarray(color, dtype=np.uint8)
  if len(color)==3: color = np.concatenate((color, [255]))
  ellipsoid.visual.vertex_colors = np.tile(color[np.newaxis,:], [nSV,1])
  return ellipsoid


def save_render(mesh_or_scene, fname, res=[1920,1080]):
  """ Offscreen render and save scene as png. Won't work through x11. """
  scene = GetScene(mesh_or_scene)
  # png = scene.save_image(resolution=res, visible=False)
  png = scene.save_image(resolution=res, visible=True)
  with open(fname, 'wb') as fid: fid.write(png)
  

def transform_points(Tx, pts): 
  """ Transform Nx3 set of points by 4x4 transform Tx. """
  return Tx[:3,:3].dot(pts.T).T + Tx[:3,3]


def show_scene_with_bindings(scene, res=[640,480], point_eps=0.25):
  """ Show a scene but with additional, custom behavior/bindings.

  WARNING: This relies on manually modified trimesh Scene code, so that
           scene.show() returns a handle to the SceneViewer object.
   
  TODO: Remove customization, find way to either manually invoke or to subclass
        Scene so that it can take a scene and do additional, custom behavior.

  INPUT
    scene (from trimesh)
    res (list): width x height
  """
  sv = tm.viewer.SceneViewer(scene, resolution=res, start_loop=False, background=[1.0, 1.0, 1.0, 0.0])
  # sv = tm.viewer.SceneViewer(scene, resolution=res, start_loop=False, background=[1.0, 1.0, 1.0, 0.0],
  #   gl.glDisable(gl.GL_CULL_FACE))
  global pointSize
  pointSize = 4.0

  # pyglet.gl.glPointSize(40)

  # sv = scene.show(resolution=res, start_loop=False)

  @sv.event
  def on_key_press(symbol, modifiers):
    now = datetime.datetime.now()
    fname = 'screenshot-{date:%Y-%m-%d-%H-%M-%S}.png'.format(date=now)
    global pointSize
    if symbol == ord('s') or symbol == ord('S'):
      pyglet.image.get_buffer_manager().get_color_buffer().save(fname)
      print('Saved: %s' % fname)
    # elif symbol == ord('c') or symbol == ord('C'):
    #   print('Camera')
    #   print(scene.graph['camera'])
    elif symbol == ord('+'):
      pointSize += point_eps
      print(f'pointSize: {pointSize}')
      pyglet.gl.glPointSize(pointSize)
    elif symbol == ord('-'):
      pointSize = max(point_eps, pointSize - point_eps)
      print(f'pointSize: {pointSize}')
      pyglet.gl.glPointSize(pointSize)

  pyglet.app.run()


def GetScene(mesh_or_scene):
  """ Helper function to return scene of mesh or scene. """
  if type(mesh_or_scene) == tm.base.Trimesh: return mesh_or_scene.scene()
  else: return mesh_or_scene


def GetCamera(mesh_or_scene):
  """ Return 4x4 transformation matrix of camera for a mesh or scene. """
  scene = GetScene(mesh_or_scene)
  try:
    c = scene.graph['camera'][0]
  except:
    c = np.eye(4)
  return c

def MakeCamerasOrbit(mesh_or_scene, nAngles):
  """ Generate camera matrices that orbit about mesh_or_scene's centroid.

  INPUT
    mesh_or_scene (trimesh): mesh or scene
    nAngles (int): number of angles to generate

  OUTPUT
    cameras (ndarray, [nAngles, 4, 4]): cameras orbiting around centroid,
                                        starting with current camera.
  """
  scene = GetScene(mesh_or_scene)
  cam = GetCamera(scene)

  angle = 360.0 // nAngles
  cams = np.zeros((nAngles, 4, 4))
  for idx, r in enumerate(np.arange(0, 360, 360.0 / nAngles)):
    rot = tm.transformations.rotation_matrix(np.radians(r), [0,1,0],
      scene.centroid)
    cams[idx] = cam.dot(rot)
  return cams


def SceneWithViews(mesh_or_scene, cameras):
  """ Return list of copies of scene, each with camera set to cameras[n]

  INPUT
    mesh_or_scene (trimesh): mesh or scene
    cameras (ndarray, [T, 4, 4]): T different cameras

  OUTPUT
    scenes (list of trimesh scenes): scenes[t] has cameras[t] set
  """
  scene = GetScene(mesh_or_scene).copy()

  scenes = [ [] for n in range(cameras.shape[0]) ]
  for idx, cam in enumerate(cameras):
    sceneI = scene.copy()
    sceneI.graph['camera'] = cam
    scenes[idx] = sceneI
  return scenes


def ConstructPointCloud(y, colors=None):
  """ Construct trimesh PointCloud for inclusion in a scene.

  INPUT
    y (ndarray, [N, D]): points
    colors (ndarray, float, [N, 4]): color for each point, black if None

  OUTPUT
    trimesh.points.PointCloud
  """
  pc = tm.points.PointCloud(y)
  if colors is not None:
    pc.colors = (255*colors).astype(np.uint8)
  else:
    pc.colors = np.zeros((y.shape[0], 4), dtype=np.uint8)
    pc.colors[:,3] = 255
  return pc


def CameraFromScenes(meshes_or_scenes):
  # cameras = np.stack([ GetScene(ms).graph['camera'][0]
  #   for ms in meshes_or_scenes ])
  cameras = np.stack([ GetScene(ms).camera.transform.copy()
    for ms in meshes_or_scenes ])
  minTranslation = np.min(cameras[:,:-1,-1], axis=0)
  cam = cameras[0].copy()
  cam[:-1,-1] = minTranslation
  return cam

def EllipsoidScene(axis_lens, Tx, colors):
  scene = tm.scene.Scene()
  P = axis_lens.shape[0]
  ellipsoids = [ MakeEllipsoid( Tx[p], axis_lens[p], colors[p] ) for p in
    range(P)]
  scene.add_geometry(ellipsoids)
  return scene

# def SetCameraForScenes(meshes_or_scenes):
#   scenes = np.stack([ GetScene(ms) for ms in meshes_or_scenes ])
#   cam = CameraFromScenes(meshes_or_scenes)
#   for scene in scenes:
#     scene.graph.update(

def MakeAxes(scale, transform=np.eye(4), color=None, minor=0.01):
  transform = np.asarray(transform)

  s = scale
  ms = minor * scale
  s2 = 0.5 * scale
  ms2 = 0.5 * ms
  msO = 10*ms

  # return list of trimesh Meshes
  Tx = transform.dot(np.array([
    [s, 0,  0, s2],
    [0, ms, 0, ms2],
    [0, 0, ms, ms2],
    [0, 0,  0, 1]]))
  Ty = transform.dot(np.array([
    [ms, 0, 0,  ms2],
    [0,  s, 0,  s2],
    [0,  0, ms, ms2],
    [0,  0, 0,  1]]))
  Tz = transform.dot(np.array([
    [ms, 0, 0, ms2],
    [0, ms, 0, ms2],
    [0, 0,  s, s2],
    [0, 0,  0, 1]]))
  To = transform.dot(np.array([
    [msO, 0, 0,   ms2],
    [0, msO, 0,   ms2],
    [0, 0,   msO, ms2],
    [0, 0,   0,   1]]))

  ax = tm.primitives.Box([1,1,1])
  ay = tm.primitives.Box([1,1,1])
  az = tm.primitives.Box([1,1,1])
  ao = tm.primitives.Box([1,1,1])

  ax.apply_transform(Tx)
  ay.apply_transform(Ty)
  az.apply_transform(Tz)
  ao.apply_transform(To)

  if color is not None:
    color = np.asarray(color)
    if color.ndim == 1: color = np.tile(color, [4,1])
    else: assert color.shape == (4,4)
  else:
    color = np.array([
      [255, 0, 0, 255],
      [0, 255, 0, 255],
      [0, 0, 255, 255],
      [0, 0, 0, 255]])
  ax.visual.vertex_colors = color[0]
  ay.visual.vertex_colors = color[1]
  az.visual.vertex_colors = color[2]
  ao.visual.vertex_colors = color[3]

  return [ax, ay, az, ao]
