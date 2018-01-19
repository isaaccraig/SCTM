
import numpy as np

def advection_diffusion(C, u, v, w, BC, del_t=0.01, del_x = 1000, del_y = 1000, del_z = 1000, D=1e4):

    # Cijkn+1 + (u∆t/4∆x) (Ci+1jkn+1 - Ci-1jkn+1) ... - (D∆t/2(∆x)^2) (Ci+1jkn+1 + Ci-1jkn+1 - 2 Cijkn+1) ... =
    # Cijkn - (u∆t/4∆x) (Ci+1jk - Ci-1jkn) ... + (D∆t/2(∆x)^2) (Ci+1jkn + Ci-1jkn - 2 Cijkn) ...

    # Wall Boundary Condition for z dimension : include a ghost point to inforce zero flux

    # no mass flux :       0 = v/2∆x * (Ck+1 - Ck-1) ---> Ck+1 = Ck-1 --> Cij(-1) = Cij1
    # no diffusion :       0 = D/∆x^2 * (Ck+1 + Ck-1 - 2Ck) ---> Ck+1 + Ck-1 = 2Ck ---> 2Ck+1 = 2Ck ---> Ck = Ck+1
    # Ck+1 = Ck-1 = Ck

    #       Cij0n+1 + (w∆t/4∆x) (Cij1n+1 - Cij(-1)n+1) ... - (D∆t/2(∆z)^2) (Cij1n+1 + Cij(-1)n+1 - 2 Cij0n+1) ... =
    #       Cij0n - (w∆t/4∆x) (Cij1 - Cij(-1)n) ... + (D∆t/2(∆z)^2) (Cij1n + Cij(-1)n - 2 Cij0n) ...
    # using Ck+1 = Ck-1
    #       Cij0n+1 + (w∆t/4∆x) (0) ... - (D∆t/2(∆z)^2) (2Cij1n+1 - 2 Cij0n+1) ... =
    #       Cij0n - (w∆t/4∆x) (0) ... + (D∆t/2(∆z)^2) (2Cij1n - 2 Cij0n) ...
    #
    # there exists diagonal changes from the removal of w∆t/4∆x = Crz/4
    #      leftdiag = [1 + crx + cry + crz] ---> [1 + crx + cry]
    #      rightdiag = [1 - crx - cry - crz] ---> [1 - crx - cry]

    # now a double dependance on Cij1n ---> zsuperdiag = [ Crz/2 ]
    # there exists no boundary dependance on Cij0n+1 from Cij0n such that this boundary term is zero : boundary[i] still = 0

    rx = D * del_t/(del_x^2)
    Crx = u * del_t/(del_x)
    ry = D * del_t/(del_y^2)
    Cry = v * del_t/(del_y)
    rz = D * del_t/(del_z^2)
    Crz = v * del_t/(del_z)

    leftdiags =  [1 + rx + ry + rz,     - ry/2 + Cry/4,    - rx/2 - Crx/4,     - ry/2 + Cry/4,     - ry/2 - Cry/4,     - rz/2 + Crz/4,     - rz/2 - Crz/4]
    rightdiags = [1 + rx - ry - rz,     ry/2 + Cry/4,      rx/2 - Crx/4,       ry/2 + Cry/4,       ry/2 - Cry/4,       rz/2 + Crz/4,       rz/2 - Crz/4]

    noflux_diagonal_right = 1 + rx + ry
    noflux_diagonal_left = 1 + rx - ry

    return cranknicolson(C, rightdiags, leftdiags, noflux_diagonal_right, noflux_diagonal_left, BC)

def cranknicolson(C, rightdiags, leftdiags, noflux_diagonal_right, noflux_diagonal_left, BC):

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
              if zindex(j) != 0:
                R[i,j] = diagonal_right
                L[i,j] = diagonal_left
              else: # remove rz from diagonal dependance for boundary condition
                R[i,j] = noflux_diagonal_right
                L[i,j] = noflux_diagonal_left
          if i + 1 == j:
              if zindex(j) != nz - 1: # positive z boundary
                R[i,j] = zsuperdiag_right
                L[i,j] = zsuperdiag_left
              else:
                boundary[i] += (zsuperdiag_right - zsuperdiag_left) * BC['+z']
          if i - 1 == j:
              if zindex(j) != 0: # negative z boundary
                L[i,j] = zsubdiag_left
                R[i,j] = zsubdiag_right
              else:
                pass # subdiagonal and superdiagonal for z are zero here, no BC needed
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
              if xindex(j) != nx - 1:
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
