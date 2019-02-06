import numpy as np
import scipy.linalg as lin

# https://pcc.byu.edu/resources.html
# https://scicomp.stackexchange.com/a/27195
# https://github.com/AlexGrig/svd_update

def  addblock_svd_update( Uarg, Sarg, Varg, Aarg, force_orth = False):
  '''
    Given the SVD of

      X = U*S*V'

    update it to be the SVD of

      [X A] = Up*Sp*Vp'

    that is, add new columns (ie, data points).

    I have found that it is faster to add several (say, 200) data points
    at a time, rather than updating it incrementally with individual
    data points (for 500 dimensional vectors, the speedup was roughly
    25x). 

    The subspace rotations involved may not preserve orthogonality due
    to numerical round-off errors.  To compensate, you can set the
    "force_orth" flag, which will force orthogonality via a QR plus
    another SVD.  In a long loop, you may want to force orthogonality
    every so often.

    See Matthew Brand, "Fast low-rank modifications of the thin
    singular value decomposition".

  '''
  U = Varg
  V = Uarg
  # Sarg is a vector of singular values? 
  S = np.eye(len(Sarg),len(Sarg))*Sarg
  A = Aarg.T

  current_rank = U.shape[1]
  m = np.dot(U.T,A)
  p = A - np.dot(U,m)
  P = lin.orth(p)
  Ra = np.dot(P.T,p)
  z = np.zeros(m.shape)
  K = np.vstack(( np.hstack((S,m)), np.hstack((z.T,Ra)) ))
  tUp,tSp,tVp = lin.svd(K);
  tUp = tUp[:,:current_rank]
  tSp = np.diag(tSp[:current_rank])
  tVp = tVp[:,:current_rank]
  Sp = tSp
  Up = np.dot(np.hstack((U,P)),tUp)
  Vp = np.dot(V,tVp[:current_rank,:])
  Vp = np.vstack((Vp, tVp[current_rank:tVp.shape[0], :]))

  if force_orth:
    UQ,UR = lin.qr(Up,mode='economic')
    VQ,VR = lin.qr(Vp,mode='economic')
    tUp,tSp,tVp = lin.svd( np.dot(np.dot(UR,Sp),VR.T));
    tSp = np.diag(tSp)
    Up = np.dot(UQ,tUp)
    Vp = np.dot(VQ,tVp)
    Sp = tSp;

  Up1 = Vp;