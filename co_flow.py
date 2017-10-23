from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import numpy as np


def solve_co_flow(input):
    nx = input['nx']
    ny = input['ny']
    lx = input['lx']
    ly = input['ly']
    Bi = input['Bi']
    R = input['R']
    NTU = input['NTU']
    x_0_boundary = input['x_0_boundary']

    # define constants and fields
    hx, hy = float(lx) / (nx - 1), float(ly) / (ny - 1)
    phi = np.array((nx, ny), dtype=float)
    x_coords, y_coords = np.meshgrid(np.linspace(0, lx, nx), np.linspace(0, ly, ny), indexing='ij')
    inds = np.reshape(a=np.arange(nx * ny), newshape=(nx, ny))

    # Assemble equations
    ij = []
    coeffs = []
    src = np.zeros((nx * ny))

    # Assemble the interior equations
    for i in xrange(1, nx - 1):
        for j in xrange(1, ny - 1):
            x, y = x_coords[i, j], y_coords[i, j]

            # Center coefficient
            ij.append((inds[i][j], inds[i][j]))
            coeffs.append(-2 * R(x, y) / hx ** 2 - 2 / hy ** 2)

            # y+ coefficient
            ij.append((inds[i][j], inds[i][j + 1]))
            coeffs.append(1 / hy ** 2)

            # y- coefficient
            ij.append((inds[i][j], inds[i][j - 1]))
            coeffs.append(1 / hy ** 2)

            # x+ coefficient
            ij.append((inds[i][j], inds[i + 1][j]))
            coeffs.append(R(x, y) / hx ** 2)

            # x- coefficient
            ij.append((inds[i][j], inds[i - 1][j]))
            coeffs.append(R(x, y) / hx ** 2)

            # Advection
            if i == 1:
                ij.append((inds[i, j], inds[i, j]))
                ij.append((inds[i, j], inds[i - 1, j]))
                ij.append((inds[i, j], inds[i - 1, j]))

                coeffs.append(-1.5 * (Bi(x, y) / NTU(x, y)) / hx)
                coeffs.append(2 * (Bi(x, y) / NTU(x, y)) / hx)
                coeffs.append(-0.5 * (Bi(x, y) / NTU(x, y)) / hx)
            else:
                ij.append((inds[i, j], inds[i, j]))
                ij.append((inds[i, j], inds[i - 1, j]))
                ij.append((inds[i, j], inds[i - 2, j]))
                coeffs.append(-1.5 * (Bi(x, y) / NTU(x, y)) / hx)
                coeffs.append(2 * (Bi(x, y) / NTU(x, y)) / hx)
                coeffs.append(-0.5 * (Bi(x, y) / NTU(x, y)) / hx)

    # Assemble x* = 0 boundary (fixed)
    for j in xrange(0, ny):
        y = y_coords[0, j]

        ij.append((inds[0, j], inds[0, j]))
        coeffs.append(1)
        src[inds[0, j]] += x_0_boundary(y)

    # Assemble x* = l boundary (infinite)
    for j in xrange(1, ny - 1):
        x, y = x_coords[-1, j], y_coords[-1, j]
        """
        ij.append((inds[-1, j], inds[-1, j]))
        ij.append((inds[-1, j], inds[-2, j]))
        ij.append((inds[-1, j], inds[-3, j]))
        coeffs.append(1.5)
        coeffs.append(-2)
        coeffs.append(0.5)
        """
        # hx coefficients
        ij.append((inds[-1, j], inds[-1, j]))
        ij.append((inds[-1, j], inds[-2, j]))
        ij.append((inds[-1, j], inds[-3, j]))
        ij.append((inds[-1, j], inds[-4, j]))
        ij.append((inds[-1, j], inds[-5, j]))

        coeffs.append(35. / 12. / hy ** 2)
        coeffs.append(-104. / 12. / hy ** 2)
        coeffs.append(114. / 12. / hy ** 2)
        coeffs.append(-56. / 12. / hy ** 2)
        coeffs.append(11. / 12. / hy ** 2)

        # hy coefficients
        ij.append((inds[-1, j], inds[-1, j]))
        ij.append((inds[-1, j], inds[-1, j + 1]))
        ij.append((inds[-1, j], inds[-1, j - 1]))

        coeffs.append(-2. * R(x, y) / hx ** 2)
        coeffs.append(R(x, y) / hx ** 2)
        coeffs.append(R(x, y) / hx ** 2)

        # Advection
        ij.append((inds[-1, j], inds[-1, j]))
        ij.append((inds[-1, j], inds[-2, j]))
        ij.append((inds[-1, j], inds[-3, j]))

        coeffs.append(-1.5 * (Bi(x, y) / NTU(x, y)) / hx)
        coeffs.append(2 * (Bi(x, y) / NTU(x, y)) / hx)
        coeffs.append(-0.5 * (Bi(x, y) / NTU(x, y)) / hx)

    # Assemble y* = 0 boundary (neumann)
    for i in xrange(1, nx):
        x = x_coords[i, 0]

        ij.append((inds[i, 0], inds[i, 0]))
        ij.append((inds[i, 0], inds[i, 1]))
        ij.append((inds[i, 0], inds[i, 2]))
        coeffs.append(1.5)
        coeffs.append(-2.)
        coeffs.append(0.5)

    # Assemble y* = 1 boundary (robin)
    for i in xrange(1, nx):
        x = x_coords[i, -1]

        ij.append((inds[i, -1], inds[i, -1]))
        ij.append((inds[i, -1], inds[i, -2]))
        ij.append((inds[i, -1], inds[i, -3]))

        coeffs.append(Bi(x, 1) + 1.5 / hy)
        coeffs.append(-2. / hy)
        coeffs.append(0.5 / hy)

    mat = csr_matrix((coeffs, zip(*ij)), shape=(nx * ny, nx * ny))
    print 'Finished assembly. Solving...'
    phi = np.reshape(spsolve(mat, src), newshape=(nx, ny))

    return x_coords, y_coords, phi


def main(input):
    pass


if __name__ == '__main__':
    from math import sin, pi, exp


    def Bi(x, y):
        if y == 1:
            return 0.1 + 20. * exp(-0.5 * x)
        else:
            return 1.


    def R(x, y):
        return 0.


    def NTU(x, y):
        return 1.


    def x_0_boundary(y):
        return 1.


    input = {
        'nx': 801,
        'ny': 801,
        'lx': 1,
        'ly': 1,
        'x_0_boundary': x_0_boundary,
        'Bi': Bi,
        'R': R,
        'NTU': NTU,
    }

    x, y, phi = solve_co_flow(input=input)

    """
    hy = input['ly'] / (input['ny'] - 1.)
    for i in xrange(1, input['nx']):
        print '{} = {}'.format(-(1.5 * phi[i, -1] - 2. * phi[i, -2] + 0.5 * phi[i, -3]) / hy / phi[i, -1],
                               Bi(x[i, -1], 1.))
    """

    with open('coflow.dat', 'w') as f:
        f.write('Title = "CoFlow Exchanger"\n')
        f.write('Variables = "x", "y", "phi"\n')
        f.write('Zone T = "Domain" J = {} I = {} F = BLOCK\n'.format(*phi.shape))

        for x_row in x:
            f.write('{}\n'.format(' '.join(map(str, x_row))))

        for y_row in y:
            f.write('{}\n'.format(' '.join(map(str, y_row))))

        for phi_row in phi:
            f.write('{}\n'.format(' '.join(map(str, phi_row))))

        f.write('Zone T = "y* = 0 Boundary" I = {} J = {} F = BLOCK\n'.format(phi.shape[0], 1))
        f.write('{}\n'.format('\n'.join(map(str, x[:, 0]))))
        f.write('{}\n'.format('\n'.join(map(str, y[:, 0]))))
        f.write('{}\n'.format('\n'.join(map(str, phi[:, 0]))))

        ly, ny = input['ly'], input['ny']
        hy = ly / (ny - 1.)

        for fixed_y in 0., 0.25, 0.5, 0.75, 1.:
            i = int(fixed_y / hy)
            f.write('Zone T = "y* = {} Boundary" I = {} J = {} F = BLOCK\n'.format(fixed_y, phi.shape[0], 1))
            f.write('{}\n'.format('\n'.join(map(str, x[:, i]))))
            f.write('{}\n'.format('\n'.join(map(str, y[:, i]))))
            f.write('{}\n'.format('\n'.join(map(str, phi[:, i]))))

        lx = input['lx']
        nx = input['nx']
        hx = lx / (nx - 1.)

        for fixed_x in 0., 0.25, 0.5, 0.75, 1.:
            i = int(fixed_x / hx)
            f.write('Zone T = "x* = {} Exchanger Midpoint" I = {} J = {} F = BLOCK\n'.format(fixed_x, 1, phi.shape[1]))
            f.write('{}\n'.format('\n'.join(map(str, x[i, :]))))
            f.write('{}\n'.format('\n'.join(map(str, y[i, :]))))
            f.write('{}\n'.format('\n'.join(map(str, phi[i, :]))))
