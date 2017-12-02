
import numpy as np

def advection_diffusion(C, u, v, w, BC, del_t=0.01, del_x = 1000, del_y = 1000, del_z = 1000, D=1e4):

    # Cijkn+1 - Cijkn = (-u∆t/4∆x)      (Ci+1jkn+1 - Ci-1jkn+1)
    #               +   (-u∆t/4∆x)      (Ci+1jkn - Ci-1jkn)
    #               +   (-v∆t/4∆x)      (Cij+1kn+1 - Cij-1kn+1)
    #               +   (-v∆t/4∆x)      (Cij+1kn - Cij-1kn)
    #               +   (-w∆t/4∆x)      (Cijk+1n+1 - Cijk-1n+1)
    #               +   (-w∆t/4∆x)      (Cijk+1n - Cijk-1n)
    #               +   (D∆t/2(∆x)^2)   (Ci+1jkn+1 + Ci-1jkn+1 + Cij+1kn+1 + Cij-1kn+1 + Cijk+1n+1 + Cijk-1n+1 - 6Cijkn+1)
    #               +   (D∆t/2(∆x)^2)   (Ci+1jkn + Ci-1jkn + Cij+1kn + Cij-1kn + Cijk+1n + Cijk-1n - 6Cijkn)

    rx = D * del_t/(del_x^2)
    Crx = u * del_t/(del_x)
    ry = D * del_t/(del_y^2)
    Cry = v * del_t/(del_y)
    rz = D * del_t/(del_z^2)
    Crz = v * del_t/(del_z)

    leftdiags =  [1 + rx + ry + rz,  - ry/2 + Cry/4, - rx/2 - Crx/4,  - ry/2 + Cry/4, - ry/2 - Cry/4, - rz/2 + Crz/4, - rz/2 - Crz/4]
    rightdiags = [1 + rx - ry - rz,  ry/2 + Cry/4, rx/2 - Crx/4, ry/2 + Cry/4, ry/2 - Cry/4,  rz/2 + Crz/4, rz/2 - Crz/4]

    return cranknicolson(C, rightdiags, leftdiags, BC)

def cranknicolson(C, rightdiags, leftdiags, BC):

    # diags in form :
    # [diagonal,  xsuperdiag, xsubdiag,
    #             ysuperdiag, ysubdiag,
    #             zsuperdiag, zsubdiag]

    diagonal_right =    rightdiags[0]
    xsuperdiag_right =  rightdiags[1]
    xsubdiag_right =    rightdiags[2]
    ysuperdiag_right =  rightdiags[3]
    ysubdiag_right =    rightdiags[4]
    zsuperdiag_right =  rightdiags[5]
    zsubdiag_right =    rightdiags[6]

    diagonal_left =    leftdiags[0]
    xsuperdiag_left =  leftdiags[1]
    xsubdiag_left =    leftdiags[2]
    ysuperdiag_left =  leftdiags[3]
    ysubdiag_left =    leftdiags[4]
    zsuperdiag_left =  leftdiags[5]
    zsubdiag_left =    leftdiags[6]

    nx,ny,nz = C.shape
    n = nx*ny*nz

    xindex = lambda i: i / (ny * nz)
    yindex = lambda i: (i%(ny * nz))/nz
    zindex = lambda i: (i%(ny * nz))%nz

    R = np.zeros([n,n])
    L = np.zeros([n,n])
    boundary = np.zeros(n)

    flat_C = np.ones(n)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # flattens to form a vector from matrix
                # so that can loop over the single n
                # variable to get the relevant value
                index = i*ny*nz + j*nz + k
                flat_C[index] = C[i,j,k]

    for i in range(n):
      for j in range(n):
          if i == j:
                R[i,j] = diagonal_right
                L[i,j] = diagonal_left
          if i + 1 == j:
              if xindex(j) != nx - 1:
                R[i,j] = zsuperdiag_right
                L[i,j] = zsuperdiag_left
              else:
                boundary[i] += (zsuperdiag_right - zsuperdiag_left) * BC['+z']
          if i - 1 == j:
              # need to implement a wall boundary condition
              if xindex(j) != 0:
                L[i,j] = zsubdiag_left
                R[i,j] = zsubdiag_right
              else:
                # BC['-z'] can't be set equal to zero as this would affect
                # the concentration (dilution represented by mixing with a lower)
                # concentration, and having a BC = 0 would lead to diffusion
                # out of the regime into the -z BC, to remove affects along this
                # boundary, it can be set equal to the lowest level
                boundary[i] += (zsubdiag_right - zsubdiag_left) * flat_C[i]
          if i + (nz - 1) == j:
              if yindex(j) != ny - 1:
                R[i,j] = ysuperdiag_right
                L[i,j] = ysuperdiag_left
              else:
                boundary[i] += (ysuperdiag_right - ysuperdiag_left) * BC['+y']
          if i - (nz - 1) == j:
              if yindex(j) != 0:
                R[i,j] = ysubdiag_right
                L[i,j] = ysubdiag_left
              else:
                boundary[i] += (ysubdiag_right - ysubdiag_left) * BC['-y']
          if i + (nz*ny - 1) == j:
              if zindex(j) != nz - 1:
                R[i,j] = xsuperdiag_right
                L[i,j] = xsuperdiag_left
              else:
                boundary[i] += (xsuperdiag_right - xsuperdiag_left) * BC['+x']
          if i - (nz*ny - 1) == j:
              if xindex(j) != 0:
                R[i,j] = xsubdiag_right
                L[i,j] = xsubdiag_left
              else:
                boundary[i] += (xsubdiag_right - xsubdiag_left) * BC['-x']


    right = np.dot(R, flat_C) + boundary
    flat_result = np.linalg.solve(L,right)
    result = np.zeros([nx, ny, nz])

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # flattens to form a vector from matrix
                # so that can loop over the single n
                # variable to get the relevant value
                index = i*ny*nz + j*nz + k
                result[i,j,k] = flat_result[index]
    return result
