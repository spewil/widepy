lsimport numpy as np
import scipy.linalg as lin
import matplotlib.pyplot as plt

# https://pcc.byu.edu/resources.html
# https://scicomp.stackexchange.com/a/27195
# https://github.com/AlexGrig/svd_update
# https://github.com/sioan/auto_xtc_hdf5_converter/blob/v0.2/user_created_analysis_functions/SVD_lib/manual_svd_update.py
# https://scicomp.stackexchange.com/questions/2678/updatable-svd-implementation-in-python-c-or-fortran
# https://pypi.org/project/gensim/0.5.0/

def pad(mat, padRow, padCol):
    """
    Add additional rows/columns to a numpy.matrix `mat`. The new rows/columns 
    will be initialized with zeros.
    """
    assert padRow >= 0, padCol >= 0
    rows, cols = mat.shape
    return np.bmat([[mat, np.matrix(np.zeros((rows, padCol)))],
                      [np.matrix(np.zeros((padRow, cols + padCol)))]])

def svd_add_cols(u, s, v, A, reorth = False):
        """
        this is adapted from gensim.lsimodel version 0.5.0

        Update singular value decomposition factors to take into account new 
        documents `A` -- columns of images?.
        
        This function corresponds to the general update of Brand (section 2), 
        specialized for `A = A.T` and `B` trivial (no update to matrix rows).

        The documents are assumed to be a list of full vectors (ie. not sparse 2-tuples).
        
        Compute new decomposition `u'`, `s'`, `v'` so that if the current matrix `X` decomposes to
        `u * s * v^T ~= X`, then
        `u' * s' * v'^T ~= [X A^T]`
        
        `u`, `s`, `v` and their new values `u'`, `s'`, `v'` are stored within `self` (ie. as 
        `u`, `v` etc.).
        
        `v` can be set to `None`, in which case it is completely ignored. This saves a
        bit of speed and a lot of memory, especially for huge corpora (size of `v` is
        linear in the number of added documents).

        k : number of singular components kept
        m : number of "features" or pixels 
        c : number of additional columns being added 


        """
        print("updating SVD with %i new documents" % len(A.T))
        keepV = v is not None
        if not keepV and reorth:
            raise TypeError("cannot reorthogonalize without the right singular vectors (v must not be None)")
        a = np.matrix(np.asarray(A))
        m, k = u.shape
        if keepV:
            n, k2 = v.shape
            assert k == k2, "left/right singular vectors shape mismatch! %i"
        m2, c = a.shape
        assert m == m2, "new documents must be in the same term-space as the original documents (old %s, new %s)" % (u.shape, a.shape)
        
        # construct orthogonal basis for (I - U * U^T) * A
        print("constructing orthogonal component")
        m = u.T * a # (k, m) * (m, c) = (k, c)
        print("computing orthogonal basis")
        P, Ra = np.linalg.qr(a - u * m) # equation (2)

        s = np.matrix(np.diag(s))
        print(s.shape)

        # allow re-orientation towards new data trends in the document stream, by giving less emphasis on old values
        # this is really cool! like an RL discount factor for old data in favor of newer data
        # s *= decay
        
        # now we're ready to construct K; K will be mostly diagonal and sparse, with
        # lots of structure, and of shape only (k + c, k + c), so its direct SVD 
        # ought to be fast for reasonably small additions of new documents (ie. tens 
        # or hundreds of new documents at a time).
        empty = pad(np.matrix([]).reshape(0, 0), c, k)
        K = np.bmat([[s, m], [empty, Ra]]) # (k + c, k + c), equation (4)
        print("computing %s SVD" % str(K.shape))
        uK, sK, vK = np.linalg.svd(K) # there is no python wrapper for partial svd => request all k + c factors :(
        lost = 1.0 - np.sum(sK[: k]) / np.sum(sK)
        print("discarding %.1f%% of data variation" % (100 * lost))
        
        # clip full decomposition to the requested rank
        uK = np.matrix(uK[:, :k])
        sK = np.matrix(np.diag(sK[:k]))
        vK = np.matrix(vK.T[:, :k]) # .T because np transposes the right vectors V, so we need to transpose it back: V.T.T = V
        
        # and finally update the left/right singular vectors
        print('rotating subspaces')
        s = sK
        
        # update U piece by piece, to avoid creating (huge) temporary arrays in a complex expression
        P = P * uK[k:]
        u = u * uK[:k]
        u += P # (m, k) * (k, k) + (m, c) * (c, k) = (m, k), equation (5)
        del P # free up memory
        
        if keepV:
            v = v * vK[:k, :] # (n + c, k) * (k, k) = (n + c, k)
            rot = vK[k:, :]
            v = np.bmat([[v], [rot]])
            
            if reorth:
                # The original article contains section 4.2 on keeping the rotations separate
                # from the subspaces (decomping V into Vsubspace * Vrotate), which further reduces 
                # complexity and improves numerical properties for rank-1 updates.
                #
                # I did not implement this step yet; instead, force the (expensive)
                # reorthogonalization explicitly from time to time, by setting reorth = True
                print("re-orthogonalizing singular vectors")
                uQ, uR = np.linalg.qr(u)
                vQ, vR = np.linalg.qr(v)
                uK, sK, vK = np.linalg.svd(uR * s * vR.T)#, full_matrices = False)
                uK = np.matrix(uK[:, :k])
                sK = np.matrix(np.diag(sK[: k]))
                vK = np.matrix(vK.T[:, :k])
                
                print("adjusting singular values by %f%%" % 
                              (100.0 * np.sum(np.abs(s - sK)) / np.sum(np.abs(s))))
                u = uQ * uK
                s = sK
                v = vQ * vK
        print("added %i documents" % len(A.T))

        return u,np.diagonal(s),v

if __name__ == '__main__':

  # noisy matrix with 0,999,998,...1 on diag 
  d = np.zeros(1000)
  d = np.arange(1000,0,-1)
  X = np.diag(d)
  N = np.random.randn(1000)
  N
  X = X+N
  X = X[:,:10]

  k = 10

  u,s,v = lin.svd(X)
  # kick out extra dimensions
  u = u[:,:k]
  print(u.shape)
  print(s.shape)
  print(v.T.shape)
  plt.plot(s,'o')
  # plt.show()

  a = np.zeros((1000,10))
  a[0,0] = 5000 
  # print(A.shape)
  u,s,v = svd_add_cols(u,s,v,a,reorth=False)
  plt.plot(s,'r-')
  plt.show()
