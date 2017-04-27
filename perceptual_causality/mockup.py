import os
import sys
import scipy.io


def main():
    mat_dir = sys.argv[1]

    for mat_file in os.listdir(mat_dir):
        if mat_file.endswith('_mat.mat'):
            data = scipy.io.load(mat_file)
        if mat_file.endswith('_tf_ordering.mat'):
            tf_ordering = scipy.io.load(mat_file)
        if mat_file.endswith('_times.mat'):
            times = scipy.io.load(mat_file)



if __name__ == '__main__':
    main()