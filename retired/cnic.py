
import numpy as np

# work through math with explicit case first before moving to CN
# then math for CN
# then ghost points

def tests():
    C = np.ones([2,2,1])

    C[0,0,0] = 10
    C[0,1,0] = 15
    C[1,0,0] = 20
    C[1,1,0] = 25

    BC={ '-x': 1,'+x': 2,'-y': 1,'+y': 2,'-z': 1, '+z': 2 }

    rx = 10
    Crx = 10
    ry = 10
    Cry = 10
    rz = 10
    Crz = 10

    leftdiags =  [ 1 + rx + ry + rz, - rx/2 + Crx/4, - rx/2 - Crx/4, - ry/2 + Cry/4, - ry/2 - Cry/4, - rz/2 + Crz/4, - rz/2 - Crz/4  ]
    rightdiags = [ 1 - rx - ry - rz, rx/2 - Crx/4, rx/2 + Crx/4, ry/2 - Cry/4, ry/2 + Cry/4, rz/2 - Crz/4, rz/2 + Crz/4  ]

    R = np.ones([4,4]);
    L = np.ones([4,4]);

    R[0,:] = [-29,2.5,2.5,0];
    R[1,:] = [7.5,-29,0, 2.5];
    R[2,:] = [7.5, 0, -29, 2.5];
    R[3,:] = [0,7.5,7.5,-29];

    right = np.dot(R, [10, 15, 20, 25]) + [30, 25, 25, 20];

    L[0,:] = [31, -2.5, -2.5, 0];
    L[1,:] = [-7.5, 31, 0, -2.5];
    L[2,:] = [-7.5, 0, 31, -2.5];
    L[3,:] = [0, -7.5, -7.5, 31];

    flat_result = (-1) * np.linalg.solve(L,right)
    print(flat_result)

    return cranknicolson(C, rightdiags, leftdiags, BC)

def CN(C, u=1, v=1, w=1, BC={ '-x': 1,'+x': 1,'-y': 1,'+y': 1,'-z': 1, '+z': 1 }, del_t=0.01, del_x = 1000, del_y = 1000, del_z = 1000, D=1e4):

    rx = D * del_t/(del_x^2)
    Crx = u * del_t/(del_x)
    ry = D * del_t/(del_y^2)
    Cry = v * del_t/(del_y)
    rz = D * del_t/(del_z^2)
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

    return cranknicolson_padded(C, rightdiags, leftdiags, BC)

def pad(C, BC={ '-x': 1,'+x': 1,'-y': 1,'+y': 1,'-z': 1, '+z': 1 }):

    nx,ny,nz = C.shape
    padded_C = np.zeros([nx+2, ny+2, nz+2])

    padded_C[1:nx+1, 1:ny+1, 1:nz+1] = C;

    padded_C[0, :, :] = BC['-x']
    padded_C[:, 0, :] = BC['-y']
    padded_C[:, :, 0] = BC['-z']

    padded_C[nx+1, :, :] = BC['+x']
    padded_C[:, ny+1, :] = BC['+y']
    padded_C[:, :, nz+1] = BC['+z']

    return padded_C

def depad(C):

    nx,ny,nz = C.shape
    unpadded_C = C[1:nx-1, 1:ny-1, 1:nz-1];
    return unpadded_C

def cranknicolson_padded(C, rightdiags, leftdiags, BC):

    # diags in form :
    # [diagonal,  xsuperdiag, xsubdiag,
    #             ysuperdiag, ysubdiag,
    #             zsuperdiag, zsubdiag]

    C = pad(C, BC)

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
    yindex = lambda i: (i % (ny * nz)) // nz
    zindex = lambda i: (i % (ny * nz)) % nz

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
          if zindex(i) + 1 == zindex(j) and xindex(i) == xindex(j) and yindex(i) == yindex(j):
                R[i,j] = zsuperdiag_right
                L[i,j] = zsuperdiag_left
          if zindex(i) - 1 == zindex(j) and xindex(i) == xindex(j) and yindex(i) == yindex(j):
                L[i,j] = zsubdiag_left
                R[i,j] = zsubdiag_right
          if yindex(i) + 1 == yindex(j) and xindex(i) == xindex(j) and zindex(i) == zindex(j):
                R[i,j] = ysuperdiag_right
                L[i,j] = ysuperdiag_left
          if yindex(i) - 1 == yindex(j) and xindex(i) == xindex(j) and zindex(i) == zindex(j):
                R[i,j] = ysubdiag_right
                L[i,j] = ysubdiag_left
          if xindex(i) + 1 == xindex(j) and yindex(i) == yindex(j) and zindex(i) == zindex(j):
                R[i,j] = xsuperdiag_right
                L[i,j] = xsuperdiag_left
          if xindex(i) - 1 == xindex(j) and yindex(i) == yindex(j) and zindex(i) == zindex(j):
                R[i,j] = xsubdiag_right
                L[i,j] = xsubdiag_left

    print("R[0,:]={}".format(R[0,:]))
    print("L[0,:]={}".format(L[0,:]))

    right = np.dot(R, flat_C)
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

    return depad(result)

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

            if zindex(j) != nz - 1: # positive z boundary
                boundary[j] += (zsuperdiag_right - zsuperdiag_left) * BC['+z']

          if zindex(i) - 1 == zindex(j) and xindex(i) == xindex(j) and yindex(i) == yindex(j):

             L[i,j] = zsubdiag_left
             R[i,j] = zsubdiag_right

             if zindex(j) != 0: # negative z boundary
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

    print("R={}".format(R))
    print("L={}".format(L))
    print("boundary={}".format(boundary))

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
                result[i,j,k] = -1 * flat_result[index]

    return result
