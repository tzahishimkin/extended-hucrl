import os
import pathlib
import json
import argparse
import importlib
import scipy.ndimage as ndi

from collections import defaultdict
from itertools import chain

import numpy as np
from matplotlib import pyplot as plt


def load_json(path):
    with open(path) as f:
        data = json.load(f)
    return data


def to_dict_list(list_dict):
    all_keys = set(chain(*[x.keys() for x in list_dict]))
    dict_list = {k: [d[k] for d in list_dict if k in d] for k in all_keys}
    return dict_list


def remove_invalid_sim(data_statistics, statistics, min_epochs=50):
    valid_sim = [len(d['rewards']) > min_epochs for d in data_statistics]
    valid_sim = np.where(valid_sim)[0]
    data_statistics = [data_statistics[v] for v in valid_sim]
    statistics = [statistics[v] for v in valid_sim]
    return data_statistics, statistics


def plot_key(data_statistics, statistics, key, out_dir):
    d_struct = [(d[key], '-'.join(s_names.split('/')[-3:-1]).split('-s')[0:-1][0])
                for (d, s_names) in zip(data_statistics, statistics) if key in d]


    # d_struct = sorted(d_struct, key=lambda x: x[1])
    uniques = np.unique([d[1] for d in d_struct])
    data = {unique: [d[0] for d in d_struct if d[1] == unique] for unique in uniques}

    if out_dir is None:
        out_dir = os.path.join(statistics[0].split('/')[0], 'figures')
    os.makedirs(out_dir, exist_ok=True)

    for key, value in data.items():
        fig = plt.figure()
        m_value = np.median(value, axis=0)
        m_value = ndi.gaussian_filter(m_value, sigma=10)
        # m_value = sklearn.
        plt.plot(m_value)
        plt.title(key)
        save_name = os.path.join(out_dir, key) + '.jpg'
        plt.savefig(save_name)
        plt.close()


def write_key_value(d, s_names, keys):
    p_str = '/'.join(s_names.split('/')[1:-1])
    p_str += ': '
    any_key_valid = False

    n_epochs = len(d['train_return'])

    for key in keys:
        if key not in d:
            continue
        values = d[key]
        values = values[-20:]
        mvalue = np.mean(values)
        p_str += f'{key}={mvalue:.1f},  '
        any_key_valid = True

    if 'epochs' in keys:
        p_str += f'epoch={n_epochs}'

    if any_key_valid:
        return p_str


def main(args):
    src_dir = args.src_dir
    out_dir = args.out_dir
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

    # fetch data
    lists = list(pathlib.Path(src_dir).glob('**/*.json'))
    statistics = [l.as_posix() for l in lists if 'statistics' in l.name]

    # arange data
    data_statistics = [load_json(path) for path in statistics]
    data_statistics = [to_dict_list(d) for d in data_statistics]
    data_statistics, statistics = remove_invalid_sim(data_statistics, statistics)

    # plot graphs
    [plot_key(data_statistics, statistics, save_key, out_dir) for save_key in args.save_keys]

    # write last results (averaged ovber 20 iterations)
    str_list = [write_key_value(d, s_names, args.save_keys) for (d, s_names) in zip(data_statistics, statistics)]
    print(np.sort(str_list))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameters for H-UCRL.")
    parser.add_argument("--src-dir", type=str, default="runs")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument('--save-keys', type=str, nargs='+', default=['train_return', 'sim_return', 'epochs'])
    main(parser.parse_args())
