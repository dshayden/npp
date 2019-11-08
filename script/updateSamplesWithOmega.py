import argparse
from pathlib import Path
import du, numpy as np, lie
from npp import SED

def main(args):
  files = Path(args.filespath).rglob('*.gz')
  
  for f in files:
    sample = du.load(str(f))
    o = sample['o']
    m = getattr(lie, o.lie)
    K = len(sample['pi']) - 1
    T = sample['theta'].shape[0]

    if K == 0:
      omega = np.zeros((T, 0,) + o.dxGm)
      sample['omega'] = omega
      du.save(str(f), sample)
      continue

    theta = sample['theta']
    omega = np.stack([ m.karcher(theta[:,k]) for k in range(K) ])
    omegaInv = np.stack([ m.inv(omega[k]) for k in range(K) ])
    for t in range(T):
      for k in range(K):
        theta[t,k] = omegaInv[k] @ theta[t,k]

    # re-estimate S
    if K > 0: S = np.array([ SED.inferSk(o, theta[:,k]) for k in range(K) ])
    else: S = np.zeros((0, o.dxA, o.dxA))

    # re-compute LL
    d = sample
    o, alpha, z, pi, E, x, Q, mL, llOld = \
      (d['o'], d['alpha'], d['z'], d['pi'], d['E'], d['x'],
       d['Q'], d['mL'], d['ll'])
    subsetIdx, dataset = ( d.get('subsetIdx', None), d.get('dataset', None) )

    data = du.load(f'{dataset}/data')
    yAll = data['y']
    if subsetIdx is not None:
      y = [yt[subsetIdx[t]] for t, yt in enumerate(yAll)]
    else:
      y = yAll

    ll = SED.logJoint(o, y, z, x, theta, E, S, Q, alpha, pi, omega, mL)
    # print(llOld, ll)
    
    path, base, _ = du.fileparts(str(f))

    SED.saveSample(f'{path}/{base}', o, alpha, z, pi, theta, E, S, x, Q, omega,
      mL, ll, subsetIdx, dataset)

    # SED.saveSample(str(f), o, alpha, z, pi, theta, E, S, x, Q, omega, mL, ll,
    #   subsetIdx, dataset)
  
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='')
  parser.add_argument('filespath', type=str, help='dataset path')
  parser.set_defaults(func=main)
  args = parser.parse_args()
  args.func(args)
