import numpy as np

def cr3bp(t, s, mu):
    x, y, z, vx, vy, vz = s

    ds = np.array([vx, vy, vz, 0, 0, 0])

    r1 = np.sqrt((x+mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x-1+mu)**2 + y**2 + z**2)

    ds[3] = 2*vy + x - ((1-mu)/r1**3)*(x+mu) + (mu/r2**3)*(1-mu-x)
    ds[4] = -2*vx + y - ((1-mu)/r1**3)*y - (mu/r2**3)*y
    ds[5] = -((1-mu)/r1**3)*z - (mu/r2**3)*z

    return ds

def jac_cr3bp(t, s, mu):
    x, y, z, vx, vy, vz = s

    r1 = np.sqrt((x+mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x-1+mu)**2 + y**2 + z**2)

    omega = np.array([[0 , 1, 0],
                      [-1, 0, 0],
                      [0 , 0, 0]])
    
    I = np.eye(3)

    U_xx = 1 - (1-mu)*(r1**2 - 3*(x+mu)**2) / r1**5 - mu*(r2**2 - 3*(x - (1-mu))**2) / r2**5
    U_yy = 1 - (1-mu)*(r1**2 - 3*(y)**2) / r1**5 - mu*(r2**2 - 3*(y)**2) / r2**5
    U_zz = -(1-mu)*(r1**2 - 3*(z)**2) / r1**5 - mu*(r2**2 - 3*(z)**2) / r2**5 

    U_xy = 3*y*(1-mu)*(x+mu) / r1**5 + 3*mu*y*(x - (1-mu)) / r2**5
    U_xz = 3*z*(1-mu)*(x+mu) / r1**5 + 3*mu*z*(x - (1-mu)) / r2**5
    U_yz = 3*y*z*(1-mu) / r1**5 + 3*mu*y*z / r2**5

    U = np.array([[U_xx, U_xy, U_xz],
                  [U_xy, U_yy, U_yz],
                  [U_xz, U_yz, U_zz]])
    Z = np.zeros(shape=(3,3))

    A = np.block([[Z, I],
                  [U, 2*omega]])
    
    return A