import argparse
import glob
import re
import os
import scipy.io


def parser_annotation(root_dir):
    file_list = glob.glob(root_dir + '/*_segments.txt')
    file_list.sort(key=lambda var: [int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])

    filename_all = []
    start_time_sec_all = []
    start_time_msec_all = []

    for file in file_list:
        filename = file.split('/')[-1]

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
                action, start_time_sec, start_time_msec = line.rstrip().lstrip().split(',')
                if action == 'start':
                    filename_all.append(filename)
                    start_time_sec_all.append(start_time_sec)
                    start_time_msec_all.append(start_time_msec)
                    continue

    scipy.io.savemat('calibration.mat',
                     mdict={'filename': filename_all,
                            'start_time_sec': start_time_sec_all,
                            'start_time_msec': start_time_msec_all})


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