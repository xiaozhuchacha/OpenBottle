import sys
import scipy.io
import numpy as np
import pickle
from collections import Counter
from decimal import *

output_dir = '/home/mark/Desktop'

class Window:
    def __init__(self, window_len, window_width):
        self.content = np.zeros((window_len, window_width))
    def __init__(self, content):
        self.content = content


class ActionAnnotation:
    def __init__(self, name):
        self.name = name
        self.prewindow_times = []
        self.sample_times = []
        self.postwindow_times = []
        self.pose_prewindows = []   # precondition window
        self.pose_samples = []      # samples in between pre and post condition
        self.pose_postwindows = []  # post condition window
        self.force_prewindows = []  # precondition window
        self.force_samples = []     # samples in between pre and post condition
        self.force_postwindows = [] # post condition window\

    def add_entry(self, prewindow_time, sample_time, postwindow_time, pose_prewindow, pose_sample, pose_postwindow,
                  force_prewindow, force_sample, force_postwindow):
        self.prewindow_times.append(prewindow_time)
        self.sample_times.append(sample_time)
        self.postwindow_times.append(postwindow_time)
        self.pose_prewindows.append(pose_prewindow)
        self.pose_samples.append(pose_sample)
        self.pose_postwindows.append(pose_postwindow)
        self.force_prewindows.append(force_prewindow)
        self.force_samples.append(force_sample)
        self.force_postwindows.append(force_postwindow)


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
    use_pca_data = False
    sampling_rate = 10          # hertz

    if use_pca_data:
        input_force_mat = '/home/mark/Dropbox/Documents/SIMPLEX/DataCollection/11_29_data_local/glovedata/pca' \
                        '/hand_only_csvs/all/bottle_success_corrected_order3-3.2-4-9-8-12-13_poses_pca_scores.mat'
        input_pose_mat = '/home/mark/Dropbox/Documents/SIMPLEX/DataCollection/11_29_data_local/glovedata/pca' \
                        '/hand_only_csvs/all/bottle_success_corrected_order3-3.2-4-9-8-12-13_forces_pca_scores.mat'
        input_times = '/home/mark/Dropbox/Documents/SIMPLEX/DataCollection/11_29_data_local/glovedata/pca/hand_only_csvs' \
                    '/all/bottle_success_corrected_order3-3.2-4-9-8-12-13_times.mat'
        input_labels = '/home/mark/Dropbox/Documents/SIMPLEX/DataCollection/11_29_data_local/glovedata/pca/hand_only_csvs' \
                    '/all/bottle_success_corrected_order3-3.2-4-9-8-12-13_label.mat'
        forces = scipy.io.loadmat(input_force_mat)['scores']
        poses = scipy.io.loadmat(input_pose_mat)['scores']
        times = scipy.io.loadmat(input_times)['times']
        labels = scipy.io.loadmat(input_labels)['label']
    else:
        input_times_data = '/home/mark/Dropbox/Documents/SIMPLEX/DataCollection/11_29_data_local/glovedata/pca' \
                        '/bottle_csvs/all/bottle_success_corrected_order3-3.2-4-9-8-12-13_times_data.mat'
        input_labels = '/home/mark/Dropbox/Documents/SIMPLEX/DataCollection/11_29_data_local/glovedata/pca' \
                       '/bottle_csvs/all/bottle_success_corrected_order3-3.2-4-9-8-12-13_label.mat'

        times_data = scipy.io.loadmat(input_times_data)['times_data']
        labels = scipy.io.loadmat(input_labels)['label']

        # extract times, poses, and forces
        num_forces = 26
        times = times_data.astype(int)[:, 0:3]
        data = times_data[:, 3:times_data.shape[1]]
        poses = data[:, 0:(data.shape[1] - num_forces)]
        forces = data[:, (data.shape[1] - num_forces):data.shape[1]]


    input_annotation_mat = '/home/mark/Dropbox/Documents/SIMPLEX/DataCollection/11_29_data_local/glovedata' \
                          '/annotations/all/annotation.mat'


    annotations = scipy.io.loadmat(input_annotation_mat)

    # convert to ints/strings
    # NOTE: msec is really nsec, mistake in the mat file
    annotations['end_time_msec'] = np.array(annotations['end_time_msec'], dtype=int)
    annotations['end_time_sec'] = np.array(annotations['end_time_sec'], dtype=int)
    annotations['start_time_msec'] = np.array(annotations['start_time_msec'], dtype=int)
    annotations['start_time_sec'] = np.array(annotations['start_time_sec'], dtype=int)
    annotations['action'] = np.array(np.char.strip(annotations['action']), dtype=str)

    annotation_dict = build_annotation_dict(annotations)

    fill_annotations(annotation_dict, annotations, forces, poses, times, labels, sampling_rate)

    dump_annotations(annotation_dict)

    print "All done!"


def build_annotation_dict(annotations):
    unique_action_labels = np.unique(annotations['action'])

    annotation_dict = dict()

    for action_label in unique_action_labels:
        action_label = action_label.strip()
        annotation_dict[action_label] = ActionAnnotation(action_label)

    return annotation_dict


def fill_annotations(annotation_dict, annotations, forces, poses, times, labels, sampling_rate):
    action_arr = annotations['action']
    start_time_sec = annotations['start_time_sec']
    start_time_nsec = annotations['start_time_msec']
    end_time_sec = annotations['end_time_sec']
    end_time_nsec = annotations['end_time_msec']
    annoation_labels = annotations['file_index']

    window_len = 10

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
        cur_label = annoation_labels[idx]
        start_sec = start_time_sec[idx]
        start_nsec = start_time_nsec[idx]
        end_sec = end_time_sec[idx]
        end_nsec = end_time_nsec[idx]

        cur_trial = trials[str(cur_label[0]) + '.' + str(cur_label[1])]

        # get index of the start and end time
        start_time_idx = find_time_idx(cur_trial.times, start_sec, start_nsec)
        end_time_idx = find_time_idx(cur_trial.times, end_sec, end_nsec)

        # ignore and warn about times that are out of range
        if start_time_idx == -1:
            print("Out of range start time at %i.%i" % (start_sec, start_nsec))
        if end_time_idx == -1:
            print("Out of range end time at %i.%i" % (end_sec, end_nsec))
        if end_time_idx == -1 or start_time_idx == -1:
            continue

        start_time = cur_trial.times[start_time_idx]
        end_time = cur_trial.times[end_time_idx]
        # print("Actual start: %i.%i Computed start: %i.%i" % (start_sec, start_nsec, start_time[0], start_time[1]))
        # print("Actual end: %i.%i Computed end: %i.%i" % (end_sec, end_nsec, end_time[0], end_time[1]))

        force_prewindow_times, force_prewindow = build_window(start_time_idx, cur_trial.forces, cur_trial.times,
                                                              window_len)
        force_postwindow_times, force_postwindow = build_window(end_time_idx-window_len, cur_trial.forces,
                                                            cur_trial.times,
                                                window_len)
        pose_prewindow_times, pose_prewindow = build_window(start_time_idx, cur_trial.poses, cur_trial.times,
                                                           window_len)
        pose_postwindow_times, pose_postwindow = build_window(end_time_idx-window_len, cur_trial.poses,
                                                             cur_trial.times, window_len)

        # sample in between windows
        pose_sample_times, pose_samples = build_sample(start_time_idx+window_len, end_time_idx-window_len,
                                                        cur_trial.poses, cur_trial.times, sampling_rate)
        force_sample_times, force_samples = build_sample(start_time_idx+window_len, end_time_idx-window_len,
                                                          cur_trial.forces, cur_trial.times, sampling_rate)

        annotation_dict[action_arr[idx]].add_entry(pose_prewindow_times, pose_sample_times, pose_postwindow_times,
                                                   pose_prewindow, pose_samples, pose_postwindow, force_prewindow,
                                                   force_samples, force_postwindow)

    return annotation_dict


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
    if nsec[cur_sec_idx] - time_nsec < (ns - nsec[prev_sec_idx] + time_nsec):
        return cur_sec_idx
    else:
        return prev_sec_idx


# builds window by row
def build_window(idx, data, times, window_size):
    return times[idx:(idx + window_size), :], data[idx:(idx + window_size), :]


def build_sample(start_idx, end_idx, data, times, sampling_rate):
    sample_times = []
    sample_data = []

    # time change between samples in seconds
    time_delta = Decimal(float(sampling_rate)/60)

    sample_sec = times[start_idx, 0]
    sample_nsec = times[start_idx, 1]
    sample_time = Decimal(str(sample_sec) + '.' + str(sample_nsec))

    idx = start_idx
    # sample between start idx and end idx
    while idx < end_idx:
        cur_sec = times[idx, 0]
        cur_nsec = times[idx, 1]

        # skip this time idx if it's a duplicate/has the same stamp
        if sample_sec == cur_sec and sample_nsec == cur_nsec:
            idx += 1
            continue

        cur_time = Decimal(str(cur_sec) + '.' + str(cur_nsec))

        if cur_time > sample_time + time_delta:
            sample_times.append(times[idx])
            sample_data.append(data[idx])
            sample_time = sample_time + time_delta
            sample_sec = cur_sec
            sample_nsec = cur_nsec
        idx += 1

    return sample_times, sample_data


def dump_annotations(annotation_dict):
    for annotation in annotation_dict:
        anno_obj = annotation_dict[annotation]
        anno_obj_len = len(anno_obj.end_times)
        window_len = anno_obj.pose_prewindows[0].shape[0]
        length = anno_obj_len * window_len
        pose_window_width = anno_obj.pose_prewindows[0].shape[1]
        force_window_width = anno_obj.force_prewindows[0].shape[1]
        width = anno_obj.end_times[0].shape[0] + 2 * pose_window_width + 2 * force_window_width
        write_arr = np.zeros((length, width))
        idx = 0
        for i in range(0, anno_obj_len):
            # write times
            write_arr[idx, 0:3] = anno_obj.start_times[i]
            write_arr[idx+1, 0:3] = anno_obj.end_times[i]

            # write pre conditions
            window = np.matrix(anno_obj.pose_prewindows[i])
            begin_col = 3
            end_col = 3 + pose_window_width
            write_arr[idx:idx+window_len, begin_col:end_col] = window

            window = np.matrix(anno_obj.force_prewindows[i])
            begin_col = end_col
            end_col += force_window_width
            write_arr[idx:idx+window_len, begin_col:end_col] = window

            # write post conditions
            window = np.matrix(anno_obj.pose_postwindows[i])
            begin_col = end_col
            end_col += pose_window_width
            write_arr[idx:idx+window_len, begin_col:end_col] = window

            window = np.matrix(anno_obj.force_postwindows[i])
            begin_col = end_col
            end_col += force_window_width
            write_arr[idx:idx+window_len, begin_col:end_col] = window

            idx = idx + window_len  # row offset of this window

        scipy.io.savemat(output_dir + "/" + annotation + "_windows.mat", {'windows': write_arr, 'pose_window_width':
            pose_window_width, 'force_window_width': force_window_width, 'window_len': window_len})


if __name__ == "__main__":
    main()