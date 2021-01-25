import argparse
import csv
import json
import math
from pathlib import Path

import pandas as pd

# intput
# image_dir_path
#     action_label
#        video_file_name
#            image_jpg_files

# output
#  label_split{1,2,3}.csv
#    video_file_name {0,1,2}
def generate_hmdb51_csv(image_dir_path, annotation_dir_path):
    for action_label_path in image_dir_path.iterdir():
        action_label = action_label_path.name
        video_list = []
        for video_file_name in action_label_path.iterdir():
            video_list.append(video_file_name.name)
        num = len(video_list)
        for i in range(1, 4):
            csv_filename = annotation_dir_path / '{}_split{}.csv'.format(action_label, i)
            if i == 1:
                tmp_video_list = video_list[:math.ceil(num * 0.2)]
            elif i == 2:
                tmp_video_list = video_list[math.ceil(num * 0.2):math.ceil(num * 0.8)]
            else:
                tmp_video_list = video_list[math.ceil(num * 0.8):]
                
            print("generating ", csv_filename, " ...")
            with csv_filename.open('w') as csv_file:
                for video in tmp_video_list:
                    csv_file.write('{} {} \n'.format(video, i - 1))
        # total_csv_filename = annotation_dir_path / '{}_total.csv'.format(action_label)
        # print("generating ", total_csv_filename, " ...")
        # with total_csv_filename.open('w') as total_csv_file:
        #     for video in video_list:
        #         total_csv_file.write(video + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path',
                        default=None,
                        type=Path,
                        help=('Path of video directory (jpg).'
                              'Using to get n_frames of each video.'))
    parser.add_argument('annotation_path',
                        default=None,
                        type=Path,
                        help='Directory path of HMDB51 annotation files.')

    args = parser.parse_args()

    generate_hmdb51_csv(args.image_path, args.annotation_path)
