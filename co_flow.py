from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import numpy as np
import h5py


def solve_co_flow(input):
    nx = input['nx']
    ny = input['ny']
    lx = input['lx']
    ly = input['ly']
    Bi = input['Bi']
    R = input['R']
    NTU = input['NTU']
    C = input['C']
    x_0_boundary = input['x_0_boundary']
    qs = input.get('Qs', lambda x, y: 0.)
    qf = input.get('Qf', lambda x: 0.)

    # define constants and fields
    hx, hy = float(lx) / (nx - 1), float(ly) / (ny - 1)
    x_coords, y_coords = np.meshgrid(np.linspace(0, lx, nx), np.linspace(0, ly, ny), indexing='ij')

    solid_inds = np.reshape(a=np.arange(nx * ny), newshape=(nx, ny))
    fluid_inds = np.arange(nx * ny, nx * ny + ny)

    # Assemble equations
    ij = []
    coeffs = []
    src = np.zeros((nx * ny + ny))

    # Assemble the solid equations
    for i in xrange(1, nx - 1):
        for j in xrange(1, ny - 1):
            x, y = x_coords[i, j], y_coords[i, j]

            # Center coefficient
            ij.append((solid_inds[i, j], solid_inds[i, j]))
            coeffs.append(-2 * R(x, y) / hx ** 2 - 2 / hy ** 2)

            # y+ coefficient
            ij.append((solid_inds[i, j], solid_inds[i, j + 1]))
            coeffs.append(1 / hy ** 2)

            # y- coefficient
            ij.append((solid_inds[i, j], solid_inds[i, j - 1]))
            coeffs.append(1 / hy ** 2)

            # x+ coefficient
            ij.append((solid_inds[i, j], solid_inds[i + 1, j]))
            coeffs.append(R(x, y) / hx ** 2)

            # x- coefficient
            ij.append((solid_inds[i, j], solid_inds[i - 1, j]))
            coeffs.append(R(x, y) / hx ** 2)

            # Advection
            if i == 1:
                ij.append((solid_inds[i, j], solid_inds[i + 1, j]))
                ij.append((solid_inds[i, j], solid_inds[i - 1, j]))

                coeffs.append(-Bi(x, y) / NTU(x, y) / hx / 2.)
                coeffs.append(Bi(x, y) / NTU(x, y) / hx / 2.)
            else:
                ij.append((solid_inds[i, j], solid_inds[i, j]))
                ij.append((solid_inds[i, j], solid_inds[i - 1, j]))
                ij.append((solid_inds[i, j], solid_inds[i - 2, j]))
                coeffs.append(-1.5 * (Bi(x, y) / NTU(x, y)) / hx)
                coeffs.append(2 * (Bi(x, y) / NTU(x, y)) / hx)
                coeffs.append(-0.5 * (Bi(x, y) / NTU(x, y)) / hx)

            # Src
            src[solid_inds[i, j]] = -qs(x, y)

    # Assemble x* = 0 boundary (fixed)
    for j in xrange(0, ny):
        y = y_coords[0, j]

        ij.append((solid_inds[0, j], solid_inds[0, j]))
        coeffs.append(1)
        src[solid_inds[0, j]] += x_0_boundary(y)

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
        ij.append((solid_inds[-1, j], solid_inds[-1, j]))
        ij.append((solid_inds[-1, j], solid_inds[-2, j]))
        ij.append((solid_inds[-1, j], solid_inds[-3, j]))
        ij.append((solid_inds[-1, j], solid_inds[-4, j]))
        ij.append((solid_inds[-1, j], solid_inds[-5, j]))

        coeffs.append(35. / 12. / hy ** 2)
        coeffs.append(-104. / 12. / hy ** 2)
        coeffs.append(114. / 12. / hy ** 2)
        coeffs.append(-56. / 12. / hy ** 2)
        coeffs.append(11. / 12. / hy ** 2)

        # hy coefficients
        ij.append((solid_inds[-1, j], solid_inds[-1, j]))
        ij.append((solid_inds[-1, j], solid_inds[-1, j + 1]))
        ij.append((solid_inds[-1, j], solid_inds[-1, j - 1]))

        coeffs.append(-2. * R(x, y) / hx ** 2)
        coeffs.append(R(x, y) / hx ** 2)
        coeffs.append(R(x, y) / hx ** 2)

        # Advection
        ij.append((solid_inds[-1, j], solid_inds[-1, j]))
        ij.append((solid_inds[-1, j], solid_inds[-2, j]))
        ij.append((solid_inds[-1, j], solid_inds[-3, j]))

        coeffs.append(-1.5 * (Bi(x, y) / NTU(x, y)) / hx)
        coeffs.append(2 * (Bi(x, y) / NTU(x, y)) / hx)
        coeffs.append(-0.5 * (Bi(x, y) / NTU(x, y)) / hx)

    # Assemble y* = 0 boundary (neumann)
    for i in xrange(1, nx):
        x = x_coords[i, 0]

        ij.append((solid_inds[i, 0], solid_inds[i, 0]))
        ij.append((solid_inds[i, 0], solid_inds[i, 1]))
        ij.append((solid_inds[i, 0], solid_inds[i, 2]))
        coeffs.extend([1.5, -2, 0.5])

    # Assemble y* = 1 boundary (robin)
    for i in xrange(1, nx):
        x, y = x_coords[i, -1], y_coords[i, -1]

        ij.append((solid_inds[i, -1], solid_inds[i, -1]))
        ij.append((solid_inds[i, -1], solid_inds[i, -2]))
        ij.append((solid_inds[i, -1], solid_inds[i, -3]))
        coeffs.extend([1.5 / hy, -2. / hy, 0.5 / hy])

        ij.append((solid_inds[i, -1], fluid_inds[i]))
        ij.append((solid_inds[i, -1], solid_inds[i, -1]))
        coeffs.extend([-Bi(x, y), Bi(x, y)])

    # Assemble the fluid equations
    # x* = 0 boundary
    ij.append((fluid_inds[0], fluid_inds[0]))
    coeffs.append(1.)
    src[fluid_inds[0]] = 0.

    for i in xrange(1, nx):
        if i == 1:
            ij.append((fluid_inds[i], fluid_inds[i + 1]))
            ij.append((fluid_inds[i], fluid_inds[i - 1]))
            coeffs.extend([1. / hx / 2., -1. / hx / 2.])
        else:
            ij.append((fluid_inds[i], fluid_inds[i]))
            ij.append((fluid_inds[i], fluid_inds[i - 1]))
            ij.append((fluid_inds[i], fluid_inds[i - 2]))
            coeffs.extend([1.5 / hx, -2. / hx, 0.5 / hx])

        # ij.append((fluid_inds[i], fluid_inds[i]))
        # ij.append((fluid_inds[i], fluid_inds[i - 1]))
        # coeffs.extend([1. / hx, -1. / hx])

        x, y = x_coords[i, -1], y_coords[i, -1]

        ij.append((fluid_inds[i], solid_inds[i, -1]))
        ij.append((fluid_inds[i], fluid_inds[i]))
        coeffs.extend([-NTU(x, y) * C(x), NTU(x, y) * C(x)])

        src[fluid_inds[i]] = qf(x)

    mat = csr_matrix((coeffs, zip(*ij)), shape=(nx * ny + ny, nx * ny + ny))

    print 'Finished assembly. Solving...'

    x = spsolve(mat, src)

    return x_coords, y_coords, np.reshape(x[:nx * ny], newshape=(nx, ny)), x[nx * ny:]


def main(input):
    pass


if __name__ == '__main__':
    from math import sin, cos, pi, exp

    input = {
        'nx': 1001,
        'ny': 1001,
        'lx': 1.,
        'ly': 1.,
        'x_0_boundary': lambda y: cos(pi * y / 3.),
        'Bi': lambda x, y: 10.,
        'R': lambda x, y: 0.,
        'NTU': lambda x, y: 20.,
        'C': lambda x: 0.5,
        'Qs': lambda x, y: cos(x) * sin(y),
        'Qf': lambda x: 0.
    }

    x, y, phi_s, phi_f = solve_co_flow(input=input)

    """
    hy = input['ly'] / (input['ny'] - 1.)
    for i in xrange(1, input['nx']):
        print '{} = {}'.format(-(1.5 * phi[i, -1] - 2. * phi[i, -2] + 0.5 * phi[i, -3]) / hy / phi[i, -1],
                               Bi(x[i, -1], 1.))
    """

    with h5py.File('coflow.h5', 'w') as f:
        group = f.create_group('solid')
        group.create_dataset('x', data=x)
        group.create_dataset('y', data=y)
        group.create_dataset('phi_s', data=phi_s)
        group.create_dataset('Qs', data=np.vectorize(input['Qs'])(x, y))
        group.create_dataset('Bi', data=np.vectorize(input['Bi'])(x, y))
        group.create_dataset('R', data=np.vectorize(input['R'])(x, y))
        group.create_dataset('NTU', data=np.vectorize(input['NTU'])(x, y))

        group = f.create_group('fluid')
        group.create_dataset('x', data=x[:, -1])
        group.create_dataset('phi_f', data=phi_f)

        lx, ly = input['lx'], input['ly']
        nx, ny = input['nx'], input['ny']
        hx, hy = lx / (nx - 1), ly / (ny - 1)

        for fixed_x in 0, 0.25, 0.5, 0.75, 1.:
            i = int(fixed_x / ((lx / (nx - 1))))
            group = f.create_group('x* = {}'.format(fixed_x))
            group.create_dataset(name='x', data=x[i, :])
            group.create_dataset(name='y', data=y[i, :])
            group.create_dataset(name='phi', data=phi_s[i, :])

        for fixed_y in 0, 0.25, 0.5, 0.75, 1.:
            j = int(fixed_y / ((ly / (ny - 1))))
            group = f.create_group('y* = {}'.format(fixed_y))
            group.create_dataset(name='x', data=x[:, j])
            group.create_dataset(name='y', data=y[:, j])
            group.create_dataset(name='phi', data=phi_s[:, j])
