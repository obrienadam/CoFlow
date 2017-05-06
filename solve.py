from setup import input
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.integrate as integrate
import numpy as np
import matplotlib
#matplotlib.use('Qt4Agg')
import matplotlib.pylab as plt

def solve(input, func):
    """
    (dict, function) -> np_array
    """
    nx = input['nx']
    ny = input['ny']
    lx = input['length_x']
    ly = input['length_y']
    R = input['R']
    Bi_input = input['Bi']
    NTU = input['NTU']

    hx = lx/(nx - 1)
    hy = ly/(ny - 1)

    print "Mesh spacing: ({}x{})".format(hx, hy)

    n = nx*ny

    Bi = np.array([0.]*n, order='F')
    for i in xrange(0, n, ny):
        x = (int(i)/int(ny) + 1)*hx

        for u_loc, Bi_val in Bi_input:
            if x <= u_loc:
                Bi[i:i + ny] = Bi_val
                break

    a_p = -2/hy**2 - 2*R/hx**2 #- Bi/(NTU*hx)
    a_e = 1/hy**2
    a_w = 1/hy**2

    a_n = R/hx**2
    a_s = R/hx**2 #+ Bi/(NTU*hx)

    a_p_diag = np.array([a_p]*n, order='F') - Bi/(NTU*hx)
    a_e_diag = np.array([a_e]*n, order='F')
    a_w_diag = np.array([a_w]*n, order='F')
    a_n_diag = np.array([a_n]*n, order='F')
    a_s_diag = np.array([a_s]*n, order='F') + Bi/(NTU*hx)

    diags = [a_s_diag, a_w_diag, a_p_diag, a_e_diag, a_n_diag]
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
    a_e_diag = a_e_diag[:-1]
    a_w_diag = a_w_diag[1:]
    a_n_diag = a_n_diag[:n - ny]
    a_s_diag = a_s_diag[ny:]

    diags = [a_s_diag, a_w_diag, a_p_diag, a_e_diag, a_n_diag]

    mat = sp.diags(diags, [-ny,-1,0,1,ny], format='lil')

    # Set up the equations for the third-kind boundary problem
    a_p = 1./hy # + Bi
    a_w = -1./hy

    Bi_input = [(input['Bi_limit'], Bi_input[0][1]), (10., 0.)]
    Bi = np.array([0.]*n, order='F')
    for i in xrange(0, n, ny):
        x = (int(i)/int(ny) + 1)*hx

        for u_loc, Bi_val in Bi_input:
            if x <= u_loc:
                Bi[i:i + ny] = Bi_val
                break

    for i in xrange(ny - 1, n, ny):
        mat[i,i] = a_p + Bi[i]
        mat[i,i-1] = a_w

    # Start assembling the rhs
    rhs = np.zeros(n)

    # At x = 0, we have some function y
    rhs[1:ny-1] = -a_s_diag[1]*func(np.linspace(0, input['length_y'], input['ny']))[1:-1]

    phi = np.zeros((ny, nx + 1), order='F')
    print "Solving the global system of equations..."
    phi[:, 1:] = np.reshape(spla.spsolve(mat, rhs), (ny, nx), order='F')
    print "Finished solving the global system of equations."
    phi[:, 0] = input['F(y)'](np.linspace(0, ly, ny))

    return phi


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


def main(input=input, show_plot=True):
    nx, ny = input['nx'] - 1, input['ny']
    lx, ly = input['length_x'], input['length_y']
    hx, hy = input['length_x']/(nx - 1), input['length_y']/(ny - 1)

    # Adjust
    input['nx'] -= 1
    input['length_x'] -= hx

    # Solve the problem
    phi = solve(input, input['F(y)'])

    x, y = np.meshgrid(np.linspace(0, ly, ny), np.linspace(0, lx, nx + 1), indexing='ij')

    # Write solution at x = 1
    i = int(1./hx) + 1
    phi_out = phi[:, i]

    with open('phi_out_x_1.txt', 'w') as f:
        f.write('\n'.join(map(str, phi_out)))

    j = int(1./hy)
    phi_wall = phi[-1,:i+1]
    y_wall = y[-1,:i+1]

    with open('y_wall.txt', 'w') as f:
        f.write('\n'.join(map(str, y_wall)))

    with open('phi_wall.txt', 'w') as f:
        f.write('\n'.join(map(str, phi_wall)))

    wall_flux = integrate.simps(phi_wall, y_wall)

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
    print "Min value: {}    Max value: {}".format(np.min(np.min(phi)), np.max(np.max(phi)))

    print show_plot
    if show_plot:
        plt.show()

    return wall_flux

if __name__ == '__main__':
    main()