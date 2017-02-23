#!/usr/bin/env python

'''
Segments csv data file according to annotated action labels. Output is a set of
folders for each action, with each sequence as its own csv file. Timestamps are
currently converted from the [s,ns] format to a floating point in seconds, relative
to the start of the action.
'''

import glob
import os
import csv


data_dir = '/volumes/data/datasets/simplex/20170214/world_centric/'
anno_dir = '/volumes/data/datasets/simplex/20170214/anno/'
output_dir = '/volumes/data/datasets/simplex/20170214/action_segments'


def get_ftime(s,ns):
    return float(s) + 1e-9*float(ns)

def main():
    indices = {}

    anno_files = glob.glob(os.path.join(anno_dir, '*segments.txt'))
    for anno_file in anno_files:

        tokens = os.path.basename(anno_file).split('_')
        prefix = '_'.join(tokens[:4])
        suffix = tokens[-2]

        data_files = glob.glob(os.path.join(data_dir, '{}*_{}_p.csv'.format(prefix, suffix)))
        if len(data_files) == 0:
            print('Could not find data file to match {}*_{}_p.csv'.format(prefix, suffix))
            continue

        with open(data_files[0], 'rb') as f_data, open(anno_file, 'r') as f_anno:
            datareader = csv.reader(f_data, delimiter=',')
            anno_lines = f_anno.readlines()

            f_output = None
            datawriter = None

            print('Processing {} trial:{}'.format(prefix, suffix))

            # Set up first label to expect
            anno_idx = 1
            anno_next = anno_lines[anno_idx].strip().split(',')
            anno = anno_next
            anno_stop_ftime = get_ftime(anno_next[1], anno_next[2])
            ref_ftime = 0

            for row in datareader:

                # Trigger on passing the start time of the next expected label
                row_ftime = get_ftime(row[0],row[1])
                time_diff = row_ftime - anno_stop_ftime
                if time_diff > 0:
                    ref_ftime = row_ftime

                    # Close out old writer, if it exists
                    if f_output != None:
                        f_output.close()

                    # Advance to next label
                    anno_idx = anno_idx+1
                    anno = anno_next
                    if anno_idx < len(anno_lines):
                        anno_next = anno_lines[anno_idx].strip().split(',')
                        anno_stop_ftime = get_ftime(anno_next[1], anno_next[2])
                    else:
                        anno_stop_ftime = float('inf')

                    # Set label index to 0 if unseen
                    if anno[0] not in indices:
                        indices[anno[0]] = 0

                    # Create path if doesn't exist
                    output_subdir = os.path.join(output_dir, anno[0])
                    output_filename = os.path.join(output_subdir, '{}_{:05d}.csv'.format(anno[0], indices[anno[0]]))
                    indices[anno[0]] = indices[anno[0]]+1
                    if not os.path.exists(output_subdir):
                        os.mkdir(output_subdir)

                    # Open up a writer
                    f_output = open(output_filename, 'wb')
                    datawriter = csv.writer(f_output, delimiter=',')

                # Write out the rows. Substitude float time insntead of the original [s, ns]
                if datawriter != None:
                    datawriter.writerow([row_ftime - ref_ftime] + row)

if __name__ == '__main__':
    main()
