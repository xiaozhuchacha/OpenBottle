import sys
import scipy.io
import numpy as np
import pickle
from collections import Counter

output_dir = '/home/mark/Desktop'

class Window:
    def __init__(self, window_size, window_width):
        self.content = np.zeros((window_size, window_width))
    def __init__(self, content):
        self.content = content


class ActionAnnotation:
    def __init__(self, name):
        self.name = name
        self.start_times = []
        self.start_idxs = []
        self.end_times = []
        self.end_idxs = []
        self.pose_prewindows = []
        self.pose_postwindows = []
        self.force_prewindows = []
        self.force_postwindows = []
        self.next_actions = []

    def add_entry(self, start_time, start_idx, end_time, end_idx, pose_prewindow, pose_postwindow, force_prewindow,
                  force_postwindow, next_action):
        self.start_times.append(start_time)
        self.start_idxs.append(start_idx)
        self.end_times.append(end_time)
        self.end_idxs.append(end_idx)
        self.pose_prewindows.append(pose_prewindow)
        self.pose_postwindows.append(pose_postwindow)
        self.force_prewindows.append(force_prewindow)
        self.force_postwindows.append(force_postwindow)
        self.next_actions.append(next_action)


class TrialContent:
    def __init__(self, label, forces, poses, times, labels):
        self.label = label

        # find indices that match this label
        minor_group = np.where(self.label[1] == labels[:, 1])
        if minor_group is not None:
            minor_group = minor_group[0]
        else:
            return
        major_group = np.where(self.label[0] == labels[:, 0])
        if major_group is not None:
            major_group = major_group[0]
        else:
            return

        group = np.intersect1d(minor_group, major_group)
        if group.size == 0:
            self.forces = []
            self.poses = []
            self.times = []
        else:
            self.forces = forces[group]
            self.poses = poses[group]
            self.times = times[group]


def main():
    data_mat_file = sys.argv[1]
    times_file = sys.argv[2]
    tf_ordering_file = sys.argv[3]
    labels_file = sys.argv[4]
    # annotation_mat_file = sys.argv[3]

    annotation_mat_file = '/home/mark/datacollection/11_29_data_local/glovedata' \
                           '/annotations/all/annotation.mat'

    annotation_mapping = parse_mapping('/home/mark/datacollection/11_29_data_local'
                                       '/glovedata/annotations/annotation_mapping.txt')

    data = scipy.io.loadmat(data_mat_file)['data']
    forces = data[:, len(data[0])-26:len(data[0])]
    poses = data[:, 0:len(data[0])-26]
    times = scipy.io.loadmat(times_file)['times']
    tf_ordering = scipy.io.loadmat(tf_ordering_file)['tfs'][0]
    tf_ordering = clean_tf_ordering(tf_ordering)
    annotations = scipy.io.loadmat(annotation_mat_file)
    # labels should be the length of the data array, indicating which file the data originally came from
    labels = scipy.io.loadmat(labels_file)['label']

    # convert to ints/strings
    # NOTE: msec is really nsec, mistake in the mat file
    annotations['end_time_msec'] = np.array(annotations['end_time_msec'], dtype=int)
    annotations['end_time_sec'] = np.array(annotations['end_time_sec'], dtype=int)
    annotations['start_time_msec'] = np.array(annotations['start_time_msec'], dtype=int)
    annotations['start_time_sec'] = np.array(annotations['start_time_sec'], dtype=int)
    annotations['action'] = np.array(np.char.strip(annotations['action']), dtype=str)

    annotation_dict = build_annotation_dict(annotations)

    fill_annotations(annotation_dict, annotations, forces, poses, times, labels, annotation_mapping)

    dump_annotations(annotation_dict)

    print "All done!"


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
        annotation_dict[action_label] = ActionAnnotation(action_label)

    return annotation_dict


def fill_annotations(annotation_dict, annotations, forces, poses, times, labels, annotation_mapping):
    action_arr = annotations['action']
    start_time_sec = annotations['start_time_sec']
    start_time_nsec = annotations['start_time_msec']
    end_time_sec = annotations['end_time_sec']
    end_time_nsec = annotations['end_time_msec']
    annotation_labels = annotations['file_index']

    window_size = 1

    trials = dict()
    load = False
    if load:
        trials = pickle.load(open('/home/mark/Desktop/trials.pkl', 'rb'))
    else:
        for annotation_label in np.vstack({tuple(row) for row in annotation_labels}):
            trials[str(annotation_label[0]) + '.' + str(annotation_label[1])] = TrialContent(annotation_label, forces,
                                                                                           poses, times,
                                                                                           labels)
        pickle.dump(trials, open('/home/mark/Desktop/trials.pkl', 'wb'))

    for idx in range(0, len(action_arr)):
        cur_label = annotation_labels[idx]
        next_action_idx = -1
        # pull has no next action
        if action_arr[idx] != 'pull' and idx < len(action_arr)-1:
            next_action_idx = annotation_mapping[action_arr[idx+1]]
        start_sec = start_time_sec[idx]
        start_nsec = start_time_nsec[idx]
        end_sec = end_time_sec[idx]
        end_nsec = end_time_nsec[idx]

        cur_trial = trials[str(cur_label[0]) + '.' + str(cur_label[1])]

        # skip trials we aren't processing this time
        if len(cur_trial.times) == 0:
            continue

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

        force_prewindow = build_window(start_time_idx, cur_trial.forces, window_size)
        force_postwindow = build_window(end_time_idx-window_size, cur_trial.forces, window_size)
        pose_prewindow = build_window(start_time_idx, cur_trial.poses, window_size)
        pose_postwindow = build_window(end_time_idx-window_size, cur_trial.poses, window_size)

        annotation_dict[action_arr[idx]].add_entry(start_time, start_time_idx, end_time, end_time_idx,
                                                   pose_prewindow, pose_postwindow, force_prewindow, force_postwindow,
                                                   next_action_idx)

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
    if nsec[cur_sec_idx] - time_nsec < (ns - nsec[prev_sec_idx] + 05):
        return cur_sec_idx
    else:
        return prev_sec_idx


# builds window by row
def build_window(idx, data, window_size):
    return data[idx:(idx + window_size), :]


def dump_annotations(annotation_dict):
    for annotation in annotation_dict:
        anno_obj = annotation_dict[annotation]
        anno_obj_len = len(anno_obj.end_times)
        if anno_obj_len == 0: # continue if this action wasn't used (empty windows)
            continue
        window_size = anno_obj.pose_prewindows[0].shape[0]
        length = anno_obj_len * window_size*2
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
            write_arr[idx:idx+window_size, begin_col:end_col] = window

            window = anno_obj.force_prewindows[i]
            begin_col = end_col
            end_col += force_window_width
            write_arr[idx:idx+window_size, begin_col:end_col] = window

            # write post conditions below
            window = anno_obj.pose_postwindows[i]
            begin_col = 3
            end_col = 3 + pose_window_width
            write_arr[idx+window_size:idx+2*window_size, begin_col:end_col] = window

            window = anno_obj.force_postwindows[i]
            begin_col = end_col
            end_col += force_window_width
            write_arr[idx+window_size:idx+2*window_size, begin_col:end_col] = window

            write_arr[idx, end_col] = anno_obj.next_actions[i]

            idx = idx + window_size * 2  # row offset of this window

        scipy.io.savemat(output_dir + "/" + annotation + "_windows.mat", {'windows': write_arr, 'pose_window_width':
            pose_window_width, 'force_window_width': force_window_width, 'window_size': window_size})
        print "Wrote mat to : %s" % str(output_dir + '/' + annotation + "_windows.mat")


def clean_tf_ordering(tf_ordering):
    tf_ordering = [x for tf_ordering[0] in tf_ordering for x in tf_ordering[0]]
    tf_ordering = [x for tf_ordering[0] in tf_ordering for x in tf_ordering[0]]
    new_ordering = []
    for i in range(0, len(tf_ordering)):
        new_ordering.append([str(tf_ordering[i][0][0]), str(tf_ordering[i][1][0])])
    return new_ordering


if __name__ == "__main__":
    main()