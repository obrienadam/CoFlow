import numpy as np
from solve import main
from setup import input

if __name__ == '__main__':
    bis = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1., 2., 5., 10., 20., 50., 100., 200., 500., 1000.]

    with open('bi_change.dat', 'w') as f:

        f.write('T = "Flux Deviation Bi" VARIABLES = "Bi", "Bi = Bi x*->inf", "Bi = Bi x* < 1", "% Deviation"\n')
        f.write('Zone T = "Data"\n')

        for bi in bis:
            input['Bi'] = [(10., bi)]

            input['Bi_limit'] = 10.
            f1 = main(input=input, show_plot=False)

            input['Bi_limit'] = 1.
            f2 = main(input=input, show_plot=False)

            f.write('{} {} {} {}\n'.format(bi, f1, f2, (f1 - f2)/f1*100.))