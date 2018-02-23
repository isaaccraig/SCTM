
import numpy as np
import pdb

def advection_diffusion(C, u, v, w, BC, del_t, del_x, del_y, del_z, D, stop = False):

    rx = D * del_t/(del_x**2)
    Crx = u * del_t/(del_x) # m/s * sec/m
    ry = D * del_t/(del_y**2) # sec/m^2 * m^2/s
    Cry = v * del_t/(del_y)
    rz = D * del_t/(del_z**2)
    Crz = v * del_t/(del_z)

    noflux = { '-x': True,'+x': True,'-y': True,'+y': True,'-z': True, '+z': True}

    leftdiags =  [      1 + rx + ry + rz,
                        - rx/2 + Crx/4,
                        - rx/2 - Crx/4,
                        - ry/2 + Cry/4,
                        - ry/2 - Cry/4,
                        - rz/2 + Crz/4,
                        - rz/2 - Crz/4  ]

    rightdiags = [      1 - rx - ry - rz,
                        rx/2 - Crx/4,
                        rx/2 + Crx/4,
                        ry/2 - Cry/4,
                        ry/2 + Cry/4,
                        rz/2 - Crz/4,
                        rz/2 + Crz/4  ]

    return cranknicolson(C, rightdiags, leftdiags, BC, noflux, stop)

###################################################################
### For the no flux condition, dC/dz = 0 and so on at the edges
### Therefore dC/dx = [C(i-1,j,k) - C(i+1,j,k)]/2*delx = 0,
### therefore C(i-1,j,k) = C(i+1,j,k) for all j,k
### therefore BC[-x] = C(i+1,j,k) for all j,k
### this insures that sum over all i,j,k of C(i,j,k,t=t') is constant
### this was tested successfully with the code below
###################################################################

def cranknicolson(C, rightdiags, leftdiags, BC, noflux, stop = False):

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

    # expands for full, such that can use same syntax as the
    # no flux boundary case, in which the boundary condition for
    # x may depend on the values of y and z

    BC =      { '-x': BC['-x'] * np.ones([ny, nz]), # dim = ny * nz
                '+x': BC['-x'] * np.ones([ny, nz]), # dim = ny * nz
                '-y': BC['-y'] * np.ones([nx, nz]), # dim = nx * nz
                '+y': BC['+y'] * np.ones([nx, nz]), # dim = nx * nz
                '-z': BC['-z'] * np.ones([nx, ny]), # dim = nx * ny
                '+z': BC['+z'] * np.ones([nx, ny])} # dim = nx * ny

    # no flux boundary conditions
    # assumes that the outter layers are uniform, rarely true
    if noflux['-x']:
        BC['-x'] = C[0,:,:]
    if noflux['+x']:
        BC['+x'] = C[nx-1,:,:]
    if noflux['-y']:
        BC['-y'] = C[:,0,:]
    if noflux['+y']:
        BC['+y'] = C[:,ny-1,:]
    if noflux['-z']:
        BC['-z'] = C[:,:,0]
    if noflux['+z']:
        BC['+z'] = C[:,:,nz-1]

    xindex = lambda i: i // (ny * nz)
    yindex = lambda i: (i%(ny * nz))//nz
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

          if i == j: # diagonal
            R[i,j] = diagonal_right
            L[i,j] = diagonal_left
        # if the x indeces and y indeces are equal but the column index is one more than the row
        # index, then are on a zsuperdiagonal element for both L and R matricies
          if zindex(i) + 1 == zindex(j) and xindex(i) == xindex(j) and yindex(i) == yindex(j):

             #   when on this zsuperdiagonal element, if the row index is NZ - 1 (final element)
             #   then at a boundary condition, as would depend on NZ (outside bounds),
             #   then, add the BC condition to the BC vector at this index
            R[i,j] = zsuperdiag_right
            L[i,j] = zsuperdiag_left

            if zindex(j) == nz - 1: # positive z boundary
                boundary[j] += (zsuperdiag_right - zsuperdiag_left) * (BC['+z'])[xindex(i),yindex(i)]

          if zindex(i) - 1 == zindex(j) and xindex(i) == xindex(j) and yindex(i) == yindex(j):

             L[i,j] = zsubdiag_left
             R[i,j] = zsubdiag_right

             if zindex(j) == 0: # negative z boundary
                boundary[j] += (zsubdiag_right - zsubdiag_left) * (BC['-z'])[xindex(i),yindex(i)]

          if yindex(i) + 1 == yindex(j) and xindex(i) == xindex(j) and zindex(i) == zindex(j):

             R[i,j] = ysuperdiag_right
             L[i,j] = ysuperdiag_left

             if yindex(j) == ny - 1:
                 boundary[j] += (ysuperdiag_right - ysuperdiag_left) * (BC['+y'])[xindex(i),zindex(i)]

          if yindex(i) - 1 == yindex(j) and xindex(i) == xindex(j) and zindex(i) == zindex(j):

             R[i,j] = ysubdiag_right
             L[i,j] = ysubdiag_left

             if yindex(j) == 0:
                 boundary[j] += (ysubdiag_right - ysubdiag_left) * (BC['-y'])[xindex(i),zindex(i)]

          if xindex(i) + 1 == xindex(j) and yindex(i) == yindex(j) and zindex(i) == zindex(j):

             R[i,j] = xsuperdiag_right
             L[i,j] = xsuperdiag_left

             if xindex(j) == nx - 1:
                 boundary[j] += (xsuperdiag_right - xsuperdiag_left) * (BC['+x'])[yindex(i),zindex(i)]

          if xindex(i) - 1 == xindex(j) and yindex(i) == yindex(j) and zindex(i) == zindex(j):

             R[i,j] = xsubdiag_right
             L[i,j] = xsubdiag_left

             if xindex(j) == 0:
                 boundary[j] += (xsubdiag_right - xsubdiag_left) * (BC['-x'])[yindex(i),zindex(i)]

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

    if stop:
        pdb.set_trace()

    return result
