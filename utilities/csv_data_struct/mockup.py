import numpy as np
import sys
import csv_loader
import pickle as pkl


def main():
    fname = sys.argv[1]
    sys.argv.pop(1)


    csv = np.loadtxt(fname, dtype='str', delimiter=',')
    data = csv_loader.load_csv(csv, fname)

    closest_row = data.find_closest_row(1481747621, 281100000)


if __name__ == '__main__':
    main()

