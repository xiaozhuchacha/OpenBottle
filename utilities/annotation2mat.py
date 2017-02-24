import argparse
import glob
import re
import os
import scipy.io


def parser_annotation(root_dir):
    file_list = glob.glob(root_dir + '/*_segments.txt')
    file_list.sort(key=lambda var: [int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])

    file_index_all = []
    trial_index_all = []
    action_all = []
    start_time_sec_all = []
    start_time_msec_all = []
    end_time_sec_all = []
    end_time_msec_all = []

    for file in file_list:
        filename_split = file.split('/')[-1].split('_success_')
        file_index1 = [s for s in filename_split[0].split('_') if s.isdigit()]
        file_index2 = [s for s in filename_split[0].split('palm') if s.isdigit()]
        trial_index = [int(s) for s in filename_split[0].split('_') if s.isdigit()]
        assert(len(file_index1) == 1)
        assert(0 <= len(file_index2) <= 1)
        assert(len(trial_index) == 1)
        file_index = str(file_index1) + '.' + str(file_index2)

        # load file
        lines = [line.rstrip('\n') for line in open(file)]
        for line in lines:
            # comment
            if line.startswith('---'):
                continue
            # empty line
            elif not line.rstrip().lstrip():
                continue
            else:
                action, end_time_sec, end_time_msec = line.rstrip().lstrip().split(',')
                if action == 'start':
                    start_time_sec = end_time_sec
                    start_time_msec = end_time_msec
                    continue

                file_index_all.append(file_index.rstrip('.'))
                trial_index_all.append(trial_index)
                action_all.append(action)
                start_time_sec_all.append(start_time_sec)
                start_time_msec_all.append(start_time_msec)
                end_time_sec_all.append(end_time_sec)
                end_time_msec_all.append(end_time_msec)

                start_time_sec = end_time_sec
                start_time_msec = end_time_msec
    scipy.io.savemat('annotation.mat',
                     mdict={'file_index': file_index_all,
                            'trial_index': trial_index_all,
                            'action': action_all,
                            'start_time_sec': start_time_sec_all,
                            'start_time_msec': start_time_msec_all,
                            'end_time_sec': end_time_sec_all,
                            'end_time_msec': end_time_msec_all})


def main():
    # parse input
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir')
    parser.add_argument('-o', '--output')
    args = parser.parse_args()

    root_dir = args.input_dir
    assert (os.path.exists(root_dir))
    assert (os.path.isdir(root_dir))

    parser_annotation(root_dir)


if __name__ == '__main__':
    main()