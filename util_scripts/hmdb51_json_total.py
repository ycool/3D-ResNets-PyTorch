import argparse
import json
from pathlib import Path

import pandas as pd

from .utils import get_n_frames


def convert_csv_to_dict(csv_dir_path):
    database = {}
#    print(csv_dir_path)
    for file_path in csv_dir_path.iterdir():
        filename = file_path.name
        # print(filename)
        data = pd.read_csv(csv_dir_path / filename, delimiter=' ', header=None)
#        print(data)
        keys = []
        subsets = []
        for i in range(data.shape[0]):
            row = data.iloc[i, :]
            if row[1] == 0: 
                subset = 'test'
            elif row[1] == 1:
                subset = 'training'
            elif row[1] == 2:
                subset = 'validation'

            keys.append(row[0].split('.')[0])
            subsets.append(subset)

        for i in range(len(keys)):
            key = keys[i]
            database[key] = {}
            database[key]['subset'] = subsets[i]
#            label = '_'.join(filename.split('_')[:-2])
            label = '_'.join(filename.split('_')[:-1])
            database[key]['annotations'] = {'label': label}

    return database


def get_labels(csv_dir_path):
    labels = []
    for file_path in csv_dir_path.iterdir():
#        label = '_'.join(file_path.name.split('_')[:-2])
        label = '_'.join(file_path.name.split('_')[:-1])
        if len(label) == 0:
            continue
        labels.append(label)
    # print(len(set(labels)))
    return sorted(list(set(labels)))


def convert_hmdb51_csv_to_json(csv_dir_path, video_dir_path,
                               dst_json_path):
    labels = get_labels(csv_dir_path)
    # print("labels:", labels)
    database = convert_csv_to_dict(csv_dir_path)
    # print('database:', database)
    dst_data = {}
    dst_data['labels'] = labels
    dst_data['database'] = {}
    dst_data['database'].update(database)
    training = 0
    test = 0
    validation = 0
    for k, v in dst_data['database'].items():
        if v['annotations'] is not None:
            label = v['annotations']['label']
        else:
            label = 'test'
        if v['subset'] == 'training':
            training += 1
        elif v['subset'] == 'test':
            test += 1
        else:
            validation += 1
        video_path = video_dir_path / label / k
        n_frames = get_n_frames(video_path)
        v['annotations']['segment'] = (1, n_frames + 1)

    print("generating {} ...".format(dst_json_path))
    print("training/test/validation {} {} {} ...".format(
        training, test, validation))
    with dst_json_path.open('w') as dst_file:
        json.dump(dst_data, dst_file)

def convert_hmdb51_csv_to_validation_json(csv_dir_path, video_dir_path,
                                          dst_json_path):
    labels = get_labels(csv_dir_path)
    # print("labels:", labels)
    database = convert_csv_to_dict(csv_dir_path)
    # print('database:', database)
    dst_data = {}
    dst_data['labels'] = labels
    dst_data['database'] = {}
    dst_data['database'].update(database)

    for k, v in list(dst_data['database'].items()):
        if v['annotations'] is not None:
            label = v['annotations']['label']
        else:
            label = 'test'

        if v['subset'] != 'validation':
           del dst_data['database'][k]
           # print('delete {} {}'.format(k, v['subset']))
           continue
        # else:
        #    print('adding {} {}'.format(k, v['subset']))
       
        video_path = video_dir_path / label / k
        n_frames = get_n_frames(video_path)
        v['annotations']['segment'] = (1, n_frames + 1)

    print(len(dst_data['database']))
    print("generating {} ...".format(dst_json_path))
    with dst_json_path.open('w') as dst_file:
        json.dump(dst_data, dst_file)

def convert_hmdb51_csv_to_test_json(csv_dir_path, video_dir_path,
                                          dst_json_path):
    labels = get_labels(csv_dir_path)
    # print("labels:", labels)
    database = convert_csv_to_dict(csv_dir_path)
    # print('database:', database)
    dst_data = {}
    dst_data['labels'] = labels
    dst_data['database'] = {}
    dst_data['database'].update(database)

    for k, v in list(dst_data['database'].items()):
        if v['annotations'] is not None:
            label = v['annotations']['label']
        else:
            label = 'test'

        if v['subset'] != 'test':
           del dst_data['database'][k]
           # print('delete {} {}'.format(k, v['subset']))
           continue
        # else:
        #    print('adding {} {}'.format(k, v['subset']))
       
        video_path = video_dir_path / label / k
        n_frames = get_n_frames(video_path)
        v['annotations']['segment'] = (1, n_frames + 1)

    print(len(dst_data['database']))
    print("generating {} ...".format(dst_json_path))
    with dst_json_path.open('w') as dst_file:
        json.dump(dst_data, dst_file)
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir_path',
                        default=None,
                        type=Path,
                        help='Directory path of HMDB51 annotation files.')
    parser.add_argument('video_path',
                        default=None,
                        type=Path,
                        help=('Path of video directory (jpg).'
                              'Using to get n_frames of each video.'))
    parser.add_argument('dst_dir_path',
                        default=None,
                        type=Path,
                        help='Directory path of dst json file.')

    args = parser.parse_args()


    dst_json_path = args.dst_dir_path / 'hmdb51_total.json'
    convert_hmdb51_csv_to_json(args.dir_path, args.video_path,
                               dst_json_path)
    validation_json_path = args.dst_dir_path / 'hmdb51_validation.json'
    convert_hmdb51_csv_to_validation_json(args.dir_path, args.video_path,
                                          validation_json_path)
        
    test_json_path = args.dst_dir_path / 'hmdb51_test.json'
    convert_hmdb51_csv_to_test_json(args.dir_path, args.video_path,
                                          validation_json_path)
