import hashlib
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


import matplotlib.cm
def get_cmap_string(palette, domain):
    domain_unique = np.unique(domain)
    hash_table = {key: i_str for i_str, key in enumerate(domain_unique)}
    mpl_cmap = matplotlib.cm.get_cmap(palette, lut=len(domain_unique))
    def cmap_out(X, **kwargs):
        return mpl_cmap(hash_table[X], **kwargs)
    return cmap_out

def plot_key(data_statistics, statistics, attribute, out_dir):
    def get_sim_name(s_name):
        try:
            name = '-'.join(s_name.split('/')[-3:-1]).split('-s')[0:-1][0]
        except IndexError:
            name = '-'.join(s_name.split('/')[-3:-1]).split('-all')[0:-1][0]
        return name

    d_struct = [(d[attribute], get_sim_name(s_names))
            for (d, s_names) in zip(data_statistics, statistics) if attribute in d]


    # d_struct = sorted(d_struct, key=lambda x: x[1])
    uniques = np.unique([d[1] for d in d_struct])
    data = {unique: [d[0] for d in d_struct if d[1] == unique] for unique in uniques}

    if out_dir is None:
        out_dir = os.path.join(statistics[0].split('/')[0], 'figures')
    out_dir = os.path.join(out_dir, attribute)
    os.makedirs(out_dir, exist_ok=True)

    def get_hash_color(st, alpha):
        col = (
            int(hashlib.sha1(st.encode("utf-8")).hexdigest(), 16) % (200) / 200,
            int(hashlib.sha1(st.encode("utf-8")).hexdigest(), 16) % (545) / 545,
            int(hashlib.sha1(st.encode("utf-8")).hexdigest(), 16) % (1010) / 1010,
            alpha
        )
        return col
    figs_list = []
    for key, value in data.items():
        agent, env, exploration, ac, b, ep = key.split('-')
        fig_num = int(hashlib.sha1(env.encode("utf-8")).hexdigest(), 16) % (10 ** 3)
        exp = f'{agent}-{env}-{ac}-{b}'
        algo = f'{agent}-{exploration}' #key

        color = get_hash_color(algo, 1)
        fig = plt.figure(exp)
        figs_list.append(exp)
        val_median = np.median(value, axis=0)
        val_median = ndi.gaussian_filter(val_median, sigma=10)
        plt.plot(val_median, label=algo, color=color)

        val_std = np.std(value, axis=0)
        std = np.minimum(val_std, 1e2)  # for visualization
        std = ndi.gaussian_filter(std, sigma=10)

        n_stds = 0.3
        for k in np.linspace(0, n_stds, 4):
            plt.fill_between(
                range(len(std)),(val_median - k * std), (val_median + k * std),
                alpha=0.3,
                edgecolor=None,
                facecolor=get_hash_color(algo, 0.1),
                linewidth=0,
                zorder=1,
            )

    for fig_name in figs_list:
        fig = plt.figure(fig_name)
        # algo = key #f'{agent}-{exploration}'
        plt.legend()
        save_name = os.path.join(out_dir, fig_name+'.jpg')
        plt.title(fig_name)
        plt.savefig(save_name)

    for fig_name in figs_list:
        fig = plt.figure(fig_name)
        plt.close()

    # for key, value in data.items():
    #     save_name = os.path.join(out_dir, key+'.jpg')
    #     plt.savefig(save_name)


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

def get_new_name(old):
    # AA = old.replace('-v0', '_v0_exps')\
    #     .replace('-Probabilistic Ensemble', '')\
    #     .replace('Optimistic','optimistic')\
    #     .replace('-0.','-ac0.') \
    #     .replace('all', 'b1.0-ep250-s')
    #
    # date = '_'.join(AA.split('_')[-2:]).replace('-', '_')
    # B = '_'.join(AA.split('_')[:-2])
    # B = '-'.join([B,date])
    B = old.replace('-v0-', '_v0_')
    return B

def dirs_renaming(src_dir, dst_dir):
    lists = list(pathlib.Path(src_dir).glob('**/*.json'))
    statistics = [l.as_posix() for l in lists if 'statistics' in l.name]
    dirs = ['/'.join(s.split('/')[0:-1]) for s in statistics]
    new_paths = [os.path.join(dst_dir, '/'.join(dir.split('/')[1:-1]), get_new_name(dir.split('/')[-1])) for dir in dirs]
    import shutil
    for source, dest in zip(dirs, new_paths):
        # print(f'{source} -> {dest}')
        shutil.move(source, dest)


def main(args):
    src_dir = args.src_dir
    dst_dir = args.dst_dir
    if dst_dir is not None:
        os.makedirs(dst_dir, exist_ok=True)

    # dirs_renaming(src_dir, dst_dir)

    # fetch data
    lists = list(pathlib.Path(src_dir).glob('**/*.json'))
    statistics = [l.as_posix() for l in lists if 'statistics' in l.name]

    # arange data
    data_statistics = [load_json(path) for path in statistics]
    data_statistics = [to_dict_list(d) for d in data_statistics]
    data_statistics, statistics = remove_invalid_sim(data_statistics, statistics)

    # plot graphs
    [plot_key(data_statistics, statistics, save_key, dst_dir) for save_key in args.save_keys]

    # write last results (averaged ovber 20 iterations)
    str_list = [write_key_value(d, s_names, args.save_keys) for (d, s_names) in zip(data_statistics, statistics)]
    print(np.sort(str_list))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameters for H-UCRL.")
    parser.add_argument("--src-dir", type=str, default="runs")
    parser.add_argument("--dst-dir", type=str, default=None)
    parser.add_argument('--save-keys', type=str, nargs='+', default=['train_return', 'sim_return', 'epochs'])
    main(parser.parse_args())
