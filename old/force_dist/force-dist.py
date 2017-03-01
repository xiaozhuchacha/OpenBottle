
import sys
import scipy.io
import numpy as np
import pickle
import matplotlib.pyplot as plt
import math

output_dir = '/home/mark/Desktop'


class ActionContent:
    def __init__(self, name):
        self.name = name
        self.times = []
        self.time_idx = []
        self.palm_forces = []
        self.palm_avgs = []
        self.finger_forces = []
        self.finger_avgs = []
        self.thumb_forces = []
        self.thumb_avgs = []

    def add_entry(self, time, time_idx, palm_force, finger_force, thumb_force):
        self.times.append(time)
        self.time_idx.append(time_idx)
        self.palm_forces.append(palm_force)
        self.finger_forces.append(finger_force)
        self.thumb_forces.append(thumb_force)

    def compute_histograms(self):
        for i in range(0, len(self.times)):
            self.palm_avgs.append(np.mean(self.palm_forces[i]))
            self.finger_avgs.append(np.mean(self.finger_forces[i]))
            self.thumb_avgs.append(np.mean(self.thumb_forces[i]))

        self.palm_avgs = np.nan_to_num(self.palm_avgs)
        self.finger_avgs = np.nan_to_num(self.finger_avgs)
        self.thumb_avgs = np.nan_to_num(self.thumb_avgs)

        plot = True
        bin_num = 20
        # stack the potential force sources and average column wise (avg across trials)
        total = np.vstack((self.palm_avgs, self.finger_avgs, self.thumb_avgs))
        total = np.mean(total, axis=0)
        # total = reject_outliers(total, m=4)
        norm_total = total / max(total)
        self.total_histogram = np.histogram(total, bins=bin_num, range=(0, max(total)))
        self.norm_total_histogram = np.histogram(norm_total, bins=bin_num, range=(0, max(norm_total)))
        if plot:
            plt.figure(2)
            plt.hist(total, bins=bin_num, range=(0, max(total)))
            plt.title(self.name + " combined histogram")
            plt.xlabel("Force")
            plt.ylabel("Frequency")
            plt.savefig('/home/mark/Desktop/' + self.name + '_total.png')
            plt.close()

            plt.figure(3)
            plt.hist(norm_total, bins=bin_num, range=(0, max(norm_total)))
            plt.title(self.name + " combined normalized histogram")
            plt.xlabel("Force")
            plt.ylabel("Frequency")
            plt.savefig('/home/mark/Desktop/' + self.name + '_normalized_total.png')
            plt.close()


        # if self.name == 'pinch':
        #     stack the potential force sources and average column wise (avg across trials)
            # total = np.vstack((self.finger_avgs, self.thumb_avgs))
            # total = np.mean(total, axis=0)
            # self.histogram = np.histogram(total, bins=bin_num, range=(0, max(total)))
            # if plot:
            #     plt.figure(2)
            #     plt.hist(total, bins=bin_num, range=(0, max(total)))
            #     plt.title(self.name + " combined histogram")
            #     plt.xlabel("Force")
            #     plt.ylabel("Frequency")
            #     plt.savefig('/home/mark/Desktop/' + self.name + '_total.png')
            #     plt.close()

        if plot:
            plt.figure(1)
            bins = np.linspace(min([min(self.palm_avgs), min(self.finger_avgs), min(self.thumb_avgs)]),
                               max([max(self.palm_avgs), max(self.finger_avgs), max(self.thumb_avgs)]),
                               bin_num)
            print self.name
            plt.hist(self.palm_avgs, bins=bins, alpha=0.5, label="palm")
            plt.hist(self.finger_avgs, bins=bins, alpha=0.5, label="finger")
            plt.hist(self.thumb_avgs, bins=bins, alpha=0.5, label="thumb")
            plt.title(self.name + " histogram")
            plt.xlabel("Force")
            plt.ylabel("Frequency")
            plt.legend(loc='upper right')
            plt.savefig('/home/mark/Desktop/' + self.name + '_divided_hist.png')
            plt.close()

    def sample(self):
        prob = self.total_histogram[0].astype(float) / sum(self.total_histogram[0])
        vals = self.total_histogram[1][:len(self.total_histogram[1])-1]
        return np.random.choice(vals, p=prob)


class TrialContent:
    def __init__(self, label, forces, poses, times, labels):
        self.label = label

        # find indices that match this label
        minor_group = np.where(self.label[1] == labels[:, 1])[0]
        major_group = np.where(self.label[0] == labels[:, 0])[0]
        group = np.intersect1d(minor_group, major_group)

        self.forces = forces[group]
        self.poses = poses[group]
        self.times = times[group]


def main():
    # input_force_mat = sys.argv[1]
    # input_pose_mat = sys.argv[2]
    # input_times = sys.argv[3]
    # input_annotation_mat = sys.argv[4]

    input_force_mat = '/home/mark/Dropbox/Documents/SIMPLEX/DataCollection/11_29_data_local/glovedata/pca_prepostconditions' \
                    '/hand_only_csvs/all/bottle_success_corrected_order3-3.2-4-9-8-12-13_forces.mat'
    input_times = '/home/mark/Dropbox/Documents/SIMPLEX/DataCollection/11_29_data_local/glovedata/pca_prepostconditions' \
                  '/hand_only_csvs' \
                '/all/bottle_success_corrected_order3-3.2-4-9-8-12-13_times.mat'
    input_labels = '/home/mark/Dropbox/Documents/SIMPLEX/DataCollection/11_29_data_local/glovedata/pca_prepostconditions/hand_only_csvs' \
                '/all/bottle_success_corrected_order3-3.2-4-9-8-12-13_label.mat'
    input_annotation_mat = '/home/mark/Dropbox/Documents/SIMPLEX/DataCollection/11_29_data_local/glovedata' \
                           '/annotations/all/annotation.mat'

    annotation_mapping = parse_mapping('/home/mark/Dropbox/Documents/SIMPLEX/DataCollection/11_29_data_local'
                                       '/glovedata/annotations/annotation_mapping.txt')

    forces = scipy.io.loadmat(input_force_mat)['forces']
    times = scipy.io.loadmat(input_times)['times']
    labels = scipy.io.loadmat(input_labels)['label']
    annotations = scipy.io.loadmat(input_annotation_mat)

    # convert to ints/strings
    # NOTE: msec is really nsec, mistake in the mat file
    annotations['end_time_msec'] = np.array(annotations['end_time_msec'], dtype=int)
    annotations['end_time_sec'] = np.array(annotations['end_time_sec'], dtype=int)
    annotations['start_time_msec'] = np.array(annotations['start_time_msec'], dtype=int)
    annotations['start_time_sec'] = np.array(annotations['start_time_sec'], dtype=int)
    annotations['action'] = np.array(np.char.strip(annotations['action']), dtype=str)

    annotation_dict = build_annotation_dict(annotations)

    fill_annotations(annotation_dict, annotations, forces, times, labels)

    for annotation in annotation_dict:
        annotation_dict[annotation].compute_histograms()

    # dump_annotations(annotation_dict)

    print "All done!"


# MAD with IQR outlier rejection
# https://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list
def reject_outliers(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s<m]


# create the annotation mapping between
def parse_mapping(file):
    with open(file) as f:
        content = f.readlines()

    annotation_mapping = dict()
    for mapping in content:
        pair = mapping.split(',')
        annotation_mapping[pair[0]] = int(pair[1])

    return annotation_mapping


def build_annotation_dict(annotations):
    unique_action_labels = np.unique(annotations['action'])

    annotation_dict = dict()

    for action_label in unique_action_labels:
        action_label = action_label.strip()
        annotation_dict[action_label] = ActionContent(action_label)

    return annotation_dict


def fill_annotations(annotation_dict, annotations, forces, times, labels):
    action_arr = annotations['action']
    start_time_sec = annotations['start_time_sec']
    start_time_nsec = annotations['start_time_msec']
    end_time_sec = annotations['end_time_sec']
    end_time_nsec = annotations['end_time_msec']
    annoation_labels = annotations['file_index']

    trials = dict()
    load = True
    if load:
        trials = pickle.load(open('/home/mark/Desktop/trials.pkl', 'rb'))
    else:
        for annoation_label in annoation_labels:
            trials[str(annoation_label[0]) + '.' + str(annoation_label[1])] = TrialContent(annoation_label, forces,
                                                                                           poses, times,
                                                                                           labels)
        pickle.dump(trials, open('/home/mark/Desktop/trials.pkl', 'wb'))


    for idx in range(0, len(action_arr)):
        action_name = action_arr[idx]
        start_sec = start_time_sec[idx]
        start_nsec = start_time_nsec[idx]
        end_sec = end_time_sec[idx]
        end_nsec = end_time_nsec[idx]

        # get index of the start and end time
        start_idx = find_time_idx(times, start_sec, start_nsec)
        end_idx = find_time_idx(times, end_sec, end_nsec)

        time = times[start_idx:end_idx+1, :]
        time_idx = [i for i in range(start_idx, end_idx+1)]

        palm_force, finger_force, thumb_force = extract_forces(start_idx, end_idx+1, forces)

        annotation_dict[action_arr[idx]].add_entry(time, time_idx, palm_force, finger_force, thumb_force)

    return annotation_dict


# extracts the palm forces, finger force, and thumb force separately
def extract_forces(start_idx, end_idx, forces):
    palm_forces = forces[start_idx:end_idx, 10:26]
    finger_forces = forces[start_idx:end_idx, 2:10]
    thumb_forces = forces[start_idx:end_idx, 0:2]
    return palm_forces, finger_forces, thumb_forces


def find_time_idx(times, time_sec, time_nsec):
    sec = times[:, 0]
    nsec = times[:, 1]
    prev_sec = time_sec - 1
    idx = 0

    sec_idxs = np.where(sec == time_sec)
    sec_idxs = sec_idxs[0]
    prev_sec_idxs = np.where(sec == prev_sec)
    prev_sec_idxs = prev_sec_idxs[0]

    if len(sec_idxs) > 0 and len(prev_sec_idxs) > 0:
        nsecs_of_matching_sec = nsec[sec_idxs]
        nsecs_of_matching_prev_sec = nsec[prev_sec_idxs]
        closest_idx, check_prev = find_match_cur_sec(sec, nsec, sec_idxs, time_nsec, nsecs_of_matching_sec)
        # if this nsec is the lowest for this second, check the previous second's highest nano second
        if check_prev:
            closest_prev_sec_idx = find_match_prev_sec(sec, nsec, prev_sec_idxs, time_nsec, nsecs_of_matching_prev_sec)
            closest_idx = determine_closest(nsec, time_nsec, closest_idx, closest_prev_sec_idx)
    # subtract the nanosecond from the current times, find the min (this is the closet timestamp)
    elif len(sec_idxs) > 0:
        nsecs_of_matching_sec = nsec[sec_idxs]
        closest_idx, _ = find_match_cur_sec(sec, nsec, sec_idxs, time_nsec, nsecs_of_matching_sec)
    # subtract the nanosecond previous second, find the max (this is the closet timestamp)
    elif len(prev_sec_idxs) > 0:
        nsecs_of_matching_prev_sec = nsec[prev_sec_idxs]
        closest_idx, _ = find_match_prev_sec(sec, nsec, prev_sec_idxs, time_nsec, nsecs_of_matching_prev_sec)
    else:
        return -1

    return closest_idx


# finds the indices of the closest time stamp
def find_match_cur_sec(sec, nsec, sec_idxs, time_nsec, nsecs_of_matching_sec):
    nsec_sub = abs(nsecs_of_matching_sec - time_nsec)
    min_nsec = np.argmin(nsec_sub)
    closest_nsec = nsecs_of_matching_sec[min_nsec]
    # find matching nsec idxs
    nsec_idxs = np.where(nsec == closest_nsec)
    # find final idx by matching the nsec_idx with sec_idxs
    idxs = np.nonzero(np.in1d(sec_idxs, nsec_idxs))[0]

    check_prev_second = False
    if closest_nsec == np.min(nsecs_of_matching_sec):
        check_prev_second = True

    return sec_idxs[idxs[0]], check_prev_second


# for the previous second, the max nsec will always be closes to the next second
def find_match_prev_sec(sec, nsec, prev_sec_idxs, time_nsec, nsecs_of_matching_prev_sec):
    max_nsec = np.argmax(nsecs_of_matching_prev_sec)

    return prev_sec_idxs[max_nsec]


# determines if the previous second's time entry is closer than the current time entry
# compares the nanoseconds to the target and returns the closer
def determine_closest(nsec, time_nsec, cur_sec_idx, prev_sec_idx):
    ns = 1e9 # nanosecond
    # distance to cut time = cur_time_ns - target_time_nsec
    # distance to prev time = ns - prev_time_ns + tart_time_nsec
    if nsec[cur_sec_idx] - time_nsec < (ns - nsec[prev_sec_idx] + 05):
        return cur_sec_idx
    else:
        return prev_sec_idx


def dump_annotations(annotation_dict):
    for annotation in annotation_dict:
        anno_obj = annotation_dict[annotation]
        anno_obj_len = len(anno_obj.end_times)
        window_len = anno_obj.pose_prewindows[0].shape[0]
        length = anno_obj_len * window_len*2
        pose_window_width = anno_obj.pose_prewindows[0].shape[1]
        force_window_width = anno_obj.force_prewindows[0].shape[1]
        width = anno_obj.end_times[0].shape[0] + pose_window_width + force_window_width + 1
        write_arr = np.zeros((length, width))
        idx = 0
        for i in range(0, anno_obj_len):
            # write times
            write_arr[idx, 0:3] = anno_obj.start_times[i]
            write_arr[idx+1, 0:3] = anno_obj.end_times[i]

            # write pre conditions
            window = anno_obj.pose_prewindows[i]
            begin_col = 3
            end_col = 3 + pose_window_width
            write_arr[idx:idx+window_len, begin_col:end_col] = window

            window = anno_obj.force_prewindows[i]
            begin_col = end_col
            end_col += force_window_width
            write_arr[idx:idx+window_len, begin_col:end_col] = window

            # write post conditions below
            window = anno_obj.pose_postwindows[i]
            begin_col = 3
            end_col = 3 + pose_window_width
            write_arr[idx+window_len:idx+2*window_len, begin_col:end_col] = window

            window = anno_obj.force_postwindows[i]
            begin_col = end_col
            end_col += force_window_width
            write_arr[idx+window_len:idx+2*window_len, begin_col:end_col] = window

            write_arr[idx, end_col] = anno_obj.next_actions[i]

            idx = idx + window_len * 2  # row offset of this window

        scipy.io.savemat(output_dir + "/" + annotation + "_windows.mat", {'windows': write_arr, 'pose_window_width':
            pose_window_width, 'force_window_width': force_window_width, 'window_len': window_len})
        print "Wrote mat to : %s" % str(output_dir + '/' + annotation + "_windows.mat")


if __name__ == "__main__":
    main()
