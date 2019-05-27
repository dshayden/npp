import unittest
import numpy as np
from npp import SED, evalSED
import du

class test_se2_randomwalk3(unittest.TestCase):
  def setUp(s):
    data = du.load('data/se2_randomwalk3')
    s.o = SED.opts(lie='se2')
    s.T = len(data['x'])
    s.y = data['y']

    # Ground-truth
    s.KTrue = data['theta'].shape[0]
    s.xTrue = du.asShape(data['x'], (s.T,) + s.o.dxGm)
    s.thetaTrue = du.asShape(data['theta'],
      (s.T,s.KTrue) + s.o.dxGm)
    s.ETrue = data['E']
    s.STrue = data['S']
    s.QTrue = data['Q']
    s.zTrue = data['z']
    s.piTrue = data['pi']

  def testRotationFullConditionalX(s):
    t = 1
    o = s.o
    
    # set x_tF as previous time rotatation and this time translation
    x_tF = s.xTrue[t-1].copy()
    x_tF[:-1,-1] = s.xTrue[t,:-1,-1]

    # sample rotation for this time, forward only
    R_x_tF = SED.sampleRotationX(o, s.y[t], s.zTrue[t], x_tF, s.thetaTrue[t],
      s.ETrue, s.QTrue, s.xTrue[t-1], nRotationSamples=20)
    x_tF[:-1,:-1] = R_x_tF

    # set x_tFB as previous time rotatation and this time translation
    x_tFB = s.xTrue[t-1].copy()
    x_tFB[:-1,-1] = s.xTrue[t,:-1,-1]

    # sample rotation for this time, forward and back
    R_x_tFB = SED.sampleRotationX(o, s.y[t], s.zTrue[t], x_tF, s.thetaTrue[t],
      s.ETrue, s.QTrue, s.xTrue[t-1], x_tplus1=s.xTrue[t+1],
      nRotationSamples=20)
    x_tFB[:-1,:-1] = R_x_tFB

    distRF = evalSED.dist2_shortest(o, x_tF, s.xTrue[t], 1.0, 0.0)
    distRFB = evalSED.dist2_shortest(o, x_tFB, s.xTrue[t], 1.0, 0.0)

    assert distRF < 1e-3, 'Rotation distance overly large for forward filter'
    assert distRFB < 1e-3, 'Rotation distance overly large for backward sampler'

  def testRotationFullConditionalTheta(s):
    t = 1
    o = s.o

    K = s.KTrue
    distsF = np.zeros(K)
    for k in range(K):
      # set theta_tkF as previous time rotatation and this time translation
      theta_tkF = s.thetaTrue[t-1,k].copy()
      theta_tkF[:-1,-1] = s.thetaTrue[t,k,:-1,-1]

      y_tk = s.y[t][s.zTrue[t]==k]
      R_theta_tkF = SED.sampleRotationTheta(o, y_tk, theta_tkF, s.xTrue[t],
        s.STrue[k], s.ETrue[k], s.thetaTrue[t-1,k], nRotationSamples=20)
      theta_tkF[:-1,:-1] = R_theta_tkF
      distsF[k] = evalSED.dist2_shortest(o, theta_tkF, s.thetaTrue[t,k],
        1.0, 0.0)

    assert np.all(distsF < 1e-1), \
      'Rotation distance overly large for forward filter'

    distsFB = np.zeros(K)
    for k in range(K):
      # set theta_tkFB as previous time rotatation and this time translation
      theta_tkFB = s.thetaTrue[t-1,k].copy()
      theta_tkFB[:-1,-1] = s.thetaTrue[t,k,:-1,-1]

      y_tk = s.y[t][s.zTrue[t]==k]
      R_theta_tkFB = SED.sampleRotationTheta(o, y_tk, theta_tkFB, s.xTrue[t],
        s.STrue[k], s.ETrue[k], s.thetaTrue[t-1,k],
        theta_tplus1_k=s.thetaTrue[t+1,k], nRotationSamples=20)
      theta_tkFB[:-1,:-1] = R_theta_tkFB
      distsFB[k] = evalSED.dist2_shortest(o, theta_tkFB, s.thetaTrue[t,k],
        1.0, 0.0)

    assert np.all(distsFB < 1e-1), \
      'Rotation distance overly large for backward sampler'

  def testTranslationFullConditionalX(s):
    t = 1
    o = s.o

    # set x_tF as previous time translation and this time rotation
    x_tF = s.xTrue[t-1].copy()
    x_tF[:-1,:-1] = s.xTrue[t,:-1,:-1]

    # sample translation for this time, forward only
    d_x_tF = SED.sampleTranslationX(o, s.y[t], s.zTrue[t], x_tF, s.thetaTrue[t],
      s.ETrue, s.QTrue, s.xTrue[t-1])
    x_tF[:-1,-1] = d_x_tF

    # set x_tFB as previous time rotatation and this time translation
    x_tFB = s.xTrue[t-1].copy()
    x_tFB[:-1,:-1] = s.xTrue[t,:-1,:-1]

    # sample rotation for this time, forward and back
    d_x_tFB = SED.sampleTranslationX(o, s.y[t], s.zTrue[t], x_tFB, s.thetaTrue[t],
      s.ETrue, s.QTrue, s.xTrue[t-1], x_tplus1=s.xTrue[t+1])
    x_tFB[:-1,-1] = d_x_tFB

    distRF = evalSED.dist2_shortest(o, x_tF, s.xTrue[t], 0.0, 1.0)
    distRFB = evalSED.dist2_shortest(o, x_tFB, s.xTrue[t], 0.0, 1.0)

    assert distRF < 1.0, 'Translation distance overly large for forward filter'
    assert distRFB < 1.0, 'Translation distance overly large for backward sampler'

  def testTranslationFullConditionalTheta(s):
    t = 1
    o = s.o

    K = s.KTrue
    distsF = np.zeros(K)
    for k in range(K):
      # set theta_tkF as previous time translation and this time rotation
      theta_tkF = s.thetaTrue[t-1,k].copy()
      theta_tkF[:-1,:-1] = s.thetaTrue[t,k,:-1,:-1]

      y_tk = s.y[t][s.zTrue[t]==k]
      d_theta_tkF = SED.sampleTranslationTheta(o, y_tk, theta_tkF, s.xTrue[t],
        s.STrue[k], s.ETrue[k], s.thetaTrue[t-1,k])
      theta_tkF[:-1,-1] = d_theta_tkF
      distsF[k] = evalSED.dist2_shortest(o, theta_tkF, s.thetaTrue[t,k],
        0.0, 1.0)

    assert np.all(distsF < 1.0), \
      'Translation distance overly large for forward filter'

    distsFB = np.zeros(K)
    for k in range(K):
      # set theta_tkFB as previous time translation and this time rotation
      theta_tkFB = s.thetaTrue[t-1,k].copy()
      theta_tkFB[:-1,:-1] = s.thetaTrue[t,k,:-1,:-1]

      y_tk = s.y[t][s.zTrue[t]==k]
      d_theta_tkFB = SED.sampleTranslationTheta(o, y_tk, theta_tkFB, s.xTrue[t],
        s.STrue[k], s.ETrue[k], s.thetaTrue[t-1,k],
        theta_tplus1_k=s.thetaTrue[t+1,k])
      theta_tkFB[:-1,-1] = d_theta_tkFB
      distsFB[k] = evalSED.dist2_shortest(o, theta_tkFB, s.thetaTrue[t,k],
        0.0, 1.0)

    assert np.all(distsFB < 1.0), \
      'Translation distance overly large for backward sampler'
