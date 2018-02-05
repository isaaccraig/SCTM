
import numpy as np
import pdb

# work through math with explicit case first before moving to CN
# then math for CN
# then ghost points


def advection_diffusion(C, u, v, w, BC, del_t, del_x, del_y, del_z, D):

    rx = D * del_t/(del_x**2)
    Crx = u * del_t/(del_x)
    ry = D * del_t/(del_y**2)
    Cry = v * del_t/(del_y)
    rz = D * del_t/(del_z**2)
    Crz = v * del_t/(del_z)

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
                boundary[j] += (zsuperdiag_right - zsuperdiag_left) * BC['+z']

          if zindex(i) - 1 == zindex(j) and xindex(i) == xindex(j) and yindex(i) == yindex(j):

             L[i,j] = zsubdiag_left
             R[i,j] = zsubdiag_right

             if zindex(j) == 0: # negative z boundary
                boundary[j] += (zsubdiag_right - zsubdiag_left) * BC['-z']

          if yindex(i) + 1 == yindex(j) and xindex(i) == xindex(j) and zindex(i) == zindex(j):

             R[i,j] = ysuperdiag_right
             L[i,j] = ysuperdiag_left

             if yindex(j) == ny - 1:
                 boundary[j] += (ysuperdiag_right - ysuperdiag_left) * BC['+y']

          if yindex(i) - 1 == yindex(j) and xindex(i) == xindex(j) and zindex(i) == zindex(j):

             R[i,j] = ysubdiag_right
             L[i,j] = ysubdiag_left

             if yindex(j) == 0:
                 boundary[j] += (ysubdiag_right - ysubdiag_left) * BC['-y']

          if xindex(i) + 1 == xindex(j) and yindex(i) == yindex(j) and zindex(i) == zindex(j):

             R[i,j] = xsuperdiag_right
             L[i,j] = xsuperdiag_left

             if xindex(j) == nx - 1:
                 boundary[j] += (xsuperdiag_right - xsuperdiag_left) * BC['+x']

          if xindex(i) - 1 == xindex(j) and yindex(i) == yindex(j) and zindex(i) == zindex(j):

             R[i,j] = xsubdiag_right
             L[i,j] = xsubdiag_left

             if xindex(j) == 0:
                 boundary[j] += (xsubdiag_right - xsubdiag_left) * BC['-x']

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
                #if (result[i,j,k] < 0):
                    #pdb.set_trace();

    return result
