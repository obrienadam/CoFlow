from setup import input
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import numpy as np
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pylab as plt

def assemble(input, func):
    """
    :param input: An input dictionary
    :return: Sparse matrix and rhs vector
    """
    nx = input['nx']
    ny = input['ny']
    lx = input['length_x']
    ly = input['length_y']
    R = input['R']
    Bi = input['Bi']
    NTU = input['NTU']

    hx = lx/(nx - 1)
    hy = ly/(ny - 1)

    n = nx*ny

    a_p = -2/hy**2 - 2*R/hx**2 - Bi/(NTU*hx)
    a_e = 1/hy**2
    a_w = 1/hy**2

    a_n = R/hx**2
    a_s = R/hx**2 + Bi/(NTU*hx)

    a_p = np.array([a_p]*n, order='F')
    a_e = np.array([a_e]*n, order='F')
    a_w = np.array([a_w]*n, order='F')
    a_n = np.array([a_n]*n, order='F')
    a_s = np.array([a_s]*n, order='F')

    diags = [a_s, a_w, a_p, a_e, a_n]
    # Set the rows that correspond to third kind boundary to 0 (east side)
    for d in diags:
        d[ny - 1::ny] = 0.

    # Set the rows that correspond to the second boundary to 0 (west side)
    for d in diags:
        d[0::ny] = 0.
    diags[2][0::ny] = -1.
    diags[3][0::ny] = 1.

    # Make sure the correct west boundaries are also 0 (west side)
    #a_w[::ny] = 0.

    # Trim the diagonals
    a_e = a_e[:-1]
    a_w = a_w[1:]
    a_n = a_n[:n - ny]
    a_s = a_s[ny:]

    diags = [a_s, a_w, a_p, a_e, a_n]

    mat = sp.diags(diags, [-ny,-1,0,1,ny], format='lil')

    # Set up the equations for the third-kind boundary problem
    a_p = 1./hy + Bi
    a_w = -1./hy

    for i in xrange(ny - 1, n, ny):
        mat[i,i] = a_p
        mat[i,i-1] = a_w

    # Start assembling the rhs
    rhs = np.zeros(n)

    # At x = 0, we have some function y
    rhs[1:ny-1] = -a_s[1]*func(np.linspace(0, input['length_y'], input['ny']))[1:-1]

    return sp.csr_matrix(mat), rhs

def write_csv(input, x, y, phi):
    file_name = '{}_data_R{}_Bi{}_NTU_{}.csv'.format(input['Case name'], input['R'], input['Bi'], input['NTU'])
    file_name = file_name.replace('.', ',', 3)

    with open(file_name, 'w') as out_file:
        for j in xrange(phi.shape[1]):
            out_file.write(', '.join(map(str, phi[:, j])) + '\n')

        out_file.write('\n')

        for j in xrange(phi.shape[1]):
            out_file.write(', '.join(map(str, y[:, j])) + '\n')

        out_file.write('\n')

        for j in xrange(phi.shape[1]):
            out_file.write(', '.join(map(str, x[:, j])) + '\n')


if __name__ == '__main__':
    nx, ny = input['nx'] - 1, input['ny']
    lx, ly = input['length_x'], input['length_y']
    hx, hy = input['length_x']/(nx - 1), input['length_y']/(ny - 1)

    # Adjust
    input['nx'] -= 1
    input['length_x'] -= hx

    # Solve the problem
    print "Assembling the global matrix..."
    mat, rhs = assemble(input, input['F(y)'])
    print "Finished assembling matrix."

    x, y = np.meshgrid(np.linspace(0, ly, ny), np.linspace(0, lx, nx + 1), indexing='ij')

    phi = np.zeros((ny, nx + 1), order='F')
    print "Solving the global system of equations..."
    phi[:, 1:] = np.reshape(spla.spsolve(mat, rhs), (ny, nx), order='F')
    print "Finished solving the global system of equations."

    phi[:, 0] = input['F(y)'](np.linspace(0, ly, ny))

    print "Writing solution to csv file..."
    write_csv(input, x, y, phi)
    print "Finished writing solution to csv file."

    print "Plotting solution and saving image to .png format..."
    fig = plt.figure()
    plt.contourf(x, y, phi, input['Contour levels'])
    plt.axis('equal')
    plt.grid(True)
    plt.colorbar()
    fig.savefig('{}.png'.format(input['Case name']), dpi=fig.dpi)
    print "Finished plotting solution and saving to .png format."

    if input['Show plot']:
        plt.show()
