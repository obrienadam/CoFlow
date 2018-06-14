import argparse
from math import sqrt

import numpy as np
import h5py

parser = argparse.ArgumentParser(description='Converts .txt files to hdf5.')

parser.add_argument('--filename',
                    dest='filename',
                    type=str,
                    required=True,
                    help='The name of the file.')

parser.add_argument('-lx',
                    dest='lx',
                    type=float,
                    default=1.,
                    help='Length of the domain in the x direction.')

parser.add_argument('-ly',
                    dest='ly',
                    type=float,
                    default=1.,
                    help='Length of the domain in the y direction.')

parser.add_argument('--ouptut-filename',
                    dest='output_filename',
                    type=str,
                    default='',
                    help='Name of the output file.')

args = parser.parse_args()

phi_s = np.loadtxt(fname=args.filename, delimiter=',')

x, y = np.meshgrid(np.linspace(0., args.lx, phi_s.shape[1]),
                   np.linspace(0., args.ly, phi_s.shape[0]),
                   indexing='ij')



with h5py.File(args.output_filename if args.output_filename else '{}.h5'.format(args.filename), 'w') as f:
    group = f.create_group('Solid')
    group.create_dataset('x', data=x)
    group.create_dataset('y', data=y)
    group.create_dataset('phi_s', data=phi_s)

    group = f.create_group('y=0')
    group.create_dataset('x', data=x[:, 0])
    group.create_dataset('phi_s', data=phi_s[:, 0])

    group = f.create_group('y=0.5')
    group.create_dataset('x', data=x[:, x.shape[1] // 2])
    group.create_dataset('phi_s', data=phi_s[:, phi_s.shape[1] // 2])

    group = f.create_group('y=1')
    group.create_dataset('x', data=x[:, -1])
    group.create_dataset('phi_s', data=phi_s[:, -1])