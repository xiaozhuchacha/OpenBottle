import string
import numpy as np
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import scipy.io


class TF:
    def __init__(self, f_id, cf_id):
        self.frame_id = f_id
        self.child_frame_id = cf_id


def main():
    data_dir = os.path.expanduser('~') +'/Dropbox/Documents/VCLA/SIMPLEX/datacollection/11_29_data_local/proc/3_bottle69_open_bottle_palm_2_tf_convert_merged_successes_proc/'
    data_file = 'hand_only_with_tf_labels'

    # csv_to_mat(data_dir, data_file)
    mat_to_csv(data_dir, data_file)

    return
    print "Loaded array with %i samples and %i features" % data.shape

    normalized_data, column_maxs, column_mins = normalize(data)

    pca = PCA()
    pca.fit(normalized_data)

    plt.figure(1, figsize=(4, 3))
    plt.clf()
    plt.axes([.2, .2, .7, .7])
    plt.plot(pca.explained_variance_, linewidth=2)
    plt.axis('tight')
    plt.xlabel('n_components')
    plt.ylabel('explained_variance')

    reconstruction = np.zeros(data.shape, dtype='f8')
    errors = np.zeros(data.shape, dtype='f8')
    mean_pose = pca.mean_

    eigenvalues = pca.explained_variance_
    eigenvectors = pca.components_
    num_components = 10 # number of eigenvectors to use
    for i in range(0, data.shape[0]):
        reconstruction[i, :] = mean_pose
        for j in range(0, num_components):
            reconstruction[i, :] += np.dot(data[i, :], eigenvectors[j, :]) * eigenvalues[j]
            # undo normalization
            reconstruction[i, :] *= (column_maxs - column_mins)
            reconstruction[i, :] += column_mins
            print "iter: %, err_sum: %f" % (j, )
        errors[i, :] = (reconstruction[i, :] - data[i, :])**2

    write_csv(data_dir + 'reconstructed.csv', reconstruction, tf_ordering, time_sec, time_nsec, image_id)

    plt.show()

    print "All done!"


def normalize(data):
    column_maxs = np.amax(data, axis=0)
    column_mins = np.amin(data, axis=0)
    for i in range(0, column_maxs.shape[0]):
        # prevent dividing by 0 when we normalize
        if column_maxs[i] == column_mins[i] :
            column_maxs[i] = 1
            column_mins[i] = 0

    # normalize data
    normalized_data = np.zeros(data.shape)
    for i in range(0, data.shape[0]):
        normalized_data[i, :] = (data[i, :] - column_mins) / (column_maxs - column_mins)
    return normalized_data, column_maxs, column_mins


def csv_to_mat(data_dir, data_file):
    print "Loading from file: " + data_dir + data_file + '.csv'

    data, time_sec, time_nsec, image_id, tf_ordering = load_csv(data_dir + data_file + '.csv')

    times = np.vstack((time_sec, time_nsec))
    times = np.vstack((times, image_id))
    times = np.transpose(times)

    write_mat(data_dir, data_file, data, times, tf_ordering)

    print "Wrote to .mats with %i samples and %i features" % data.shape


def mat_to_csv(data_dir, data_base):
    data_file = data_dir + data_base + '_data_reconstructed.mat'

    # cheat and use the same times, image_ids, and tfs as the original CSV so we only load the data mat
    data, time_sec, time_nsec, image_id, tf_ordering = load_csv(data_dir + data_base + '.csv')

    data = scipy.io.loadmat(data_file)

    write_csv(data_dir + data_base + '_reconstructed.csv', data['reconstructed_data'], tf_ordering, time_sec,
              time_nsec, image_id)



# loads a CSV into a numpy matrix, ignores any tf with strings in ignored_children as their child_frame_id
# and ignores any tf with strings in ignored_parents as their frame_id (in the csv the format is frame_id,
# child_framed_id)
def load_csv(file):
    # treat the tf identifiers as comments, ignore
    loaded = np.loadtxt(file, dtype='str', delimiter=',')

    num_cols = 0
    num_rows = loaded.shape[0]
    # count the number of usable columns to preallocate, skip first three (times + image_id)
    for i in range(3, loaded.shape[1]):
        if is_float_cell(loaded[0, i]):
            # print loaded[0, i]
            num_cols += 1

    # preallocate and fill data matrix
    data = np.zeros((num_rows, num_cols))
    column = np.zeros((num_rows, 1), dtype='|S16')
    j = 0
    tf_ordering = [] # stores array of tf's in order of the original file
    # first 3 entries are time_s, time_ns, image_id
    time_sec = np.array(loaded[:, 0], dtype='|S16')
    time_nsec = np.array(loaded[:, 1], dtype='|S16')
    image_id = np.array(loaded[:, 2], dtype='|S16')
    i = 3
    data_idx = 0
    while i < loaded.shape[1]:
        entry_type, tf_ordering = check_entry(loaded, i, tf_ordering)
        if entry_type == 'tf':
            for j in range(i+2, i+9):
                column = loaded[:, j]
                data[:, data_idx] = np.array(column)
                data_idx += 1
            i += 9
        elif entry_type == 'force':
            for j in range(i, i+26):
                data[:, data_idx] = np.array(loaded[:, j])
                data_idx += 1
            break # finished after force

    return data, time_sec, time_nsec, image_id, tf_ordering


def check_entry(loaded, col, tf_ordering):
    # check if cell is a tf
    cell = loaded[0, col]
    if loaded[0, col][0].isalpha() or loaded[0, col][0] == '/':
        tf_ordering.append(TF(loaded[0, col], loaded[0, col+1]))
        return 'tf', tf_ordering
    else:
        return 'force', tf_ordering


def write_mat(dir, fname, data, times, tfs):
    scipy.io.savemat(dir + fname + '_data.mat', mdict={'data': data})
    scipy.io.savemat(dir + fname + '_times.mat', mdict={'times': times})
    scipy.io.savemat(dir + fname + '_tf_order.mat', mdict={'tfs': tfs})


def load_mat(dir, datafname):
    data = scipy.io.loadmat(dir + datafname)

    return data


def write_csv(filename, data, tf_ordering, time_sec, time_nsec, image_id):
    # construct array with all data
    write_arr = np.empty(shape=(data.shape[0], 3 + data.shape[1] + 2 * len(tf_ordering)), dtype=object)
    write_arr[:, 0] = time_sec
    write_arr[:, 1] = time_nsec
    write_arr[:, 2] = image_id
    data_idx = 0
    write_idx = 3
    for tf_idx in range(0, len(tf_ordering)):
        # write tf
        write_arr[:, write_idx] = np.full((data.shape[0]), tf_ordering[tf_idx].frame_id, dtype=object)
        write_idx += 1
        write_arr[:, write_idx] = np.full((data.shape[0]), tf_ordering[tf_idx].child_frame_id, dtype=object)
        write_idx += 1
        print "tf_idx: %i Writing tf %s -> %s" % (tf_idx, tf_ordering[tf_idx].frame_id, tf_ordering[
            tf_idx].child_frame_id)
        # write the next 7 columns
        for j in range(0, 7):
            write_arr[:, write_idx] = data[:, data_idx]
            print "tf_idx: %i w_idx: %i d_idx: %i" % (tf_idx, write_idx, data_idx)
            data_idx += 1
            write_idx += 1
    # write remaining forces
    for i in range(data_idx, data.shape[1]):
        write_arr[:, write_idx] = data[:, data_idx]
        data_idx += 1
        write_idx += 1
    # actually write the file
    np.savetxt(filename, write_arr, delimiter=',', fmt='%s')


# determines if a string cell contains a floating point number (not robustly!)
def is_float_cell(cell):
    return len(cell) > 0 and (cell[0].isdigit() or cell[0] == '-')


if __name__ == "__main__":
    main()
