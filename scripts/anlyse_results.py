import hashlib
import itertools
import os
import pathlib
import json
import argparse
import importlib
import shutil

import scipy.ndimage as ndi

from collections import defaultdict
from itertools import chain

import numpy as np
from matplotlib import pyplot as plt


def get_hash_color(st, alpha):
    col = (
        int(hashlib.sha1(st.encode("utf-8")).hexdigest(), 16) % (200) / 200,
        int(hashlib.sha1(st.encode("utf-8")).hexdigest(), 16) % (545) / 545,
        int(hashlib.sha1(st.encode("utf-8")).hexdigest(), 16) % (1010) / 1010,
        alpha
    )
    return col


def load_json(path):
    with open(path) as f:
        data = json.load(f)
    return data


def to_dict_list(list_dict):
    all_keys = set(chain(*[x.keys() for x in list_dict]))
    dict_list = {k: [d[k] for d in list_dict if k in d] for k in all_keys}
    return dict_list


def get_valid_simulations_indxes(data_statistics, min_epochs=250):
    # simulations under min_epochs
    valid_idxs = [len(d['rewards']) >= min_epochs for d in data_statistics]
    valid_idxs = np.where(valid_idxs)[0]
    return valid_idxs


import matplotlib.cm


def get_cmap_string(palette, domain):
    domain_unique = np.unique(domain)
    hash_table = {key: i_str for i_str, key in enumerate(domain_unique)}
    mpl_cmap = matplotlib.cm.get_cmap(palette, lut=len(domain_unique))

    def cmap_out(X, **kwargs):
        return mpl_cmap(hash_table[X], **kwargs)

    return cmap_out


def split_dict_indx_by_unique_key_list(data_dict, keys_list):
    dict_split_ixds = {}
    for ii, h_param in enumerate(data_dict):
        key = '-'.join([str(h_param[fs]) for fs in keys_list])
        dict_split_ixds.setdefault(key, []).append(ii)
    return dict_split_ixds


def plot_key(data_statistics, statistics_files, hparams, attribute, dst_dir, **kwargs):
    # d_struct = [(d[attribute], hparam)
    #             for (d, hparam) in zip(data_statistics, hparams) if attribute in d]

    # d_struct = sorted(d_struct, key=lambda x: x[1])
    # uniques = np.unique([d[1]['env'] for d in d_struct])
    # data = {unique: [d for d in d_struct if d[1]['env'] == unique] for unique in uniques}

    if dst_dir is None:
        dst_dir = os.path.join(statistics_files[0].split('/')[0], 'figures')
    out_dir = os.path.join(dst_dir, attribute)
    os.makedirs(out_dir, exist_ok=True)

    figs_list = []
    datas = {}
    for (data_statistic, statistics_file, hparam) in zip(data_statistics, statistics_files, hparams):
        # statistics_file
        exp = f"{hparam['env']}-ac{hparam['action_cost']}-b{hparam['beta']}"
        algo = f"{hparam['agent']}-{hparam['exploration']}"  # key
        try:
            datas.setdefault(f"{exp}--{algo}", []).append(data_statistic[attribute])
        except BaseException:
            continue

    for expalgo, data in datas.items():
        if data == []:
            continue

        exp, algo, = expalgo.split('--')
        # fig_num = int(hashlib.sha1(env.encode("utf-8")).hexdigest(), 16) % (10 ** 3)
        # exp = f"{hparam['env']}-ac{hparam['action_cost']}-b{hparam['beta']}"
        # algo = f"{hparam['agent']}-{hparam['exploration']}"  # key
        min_length = min([len(d) for d in data])
        data = [d[:min_length] for d in data]

        value = np.array(data)
        color = get_hash_color(algo, 1)
        fig = plt.figure(exp)
        figs_list.append(exp)
        val_median = np.median(value, axis=0)
        val_median = ndi.gaussian_filter(val_median, sigma=10)
        plt.plot(range(len(val_median)), val_median, label=algo, color=color)

        val_std = np.std(value, axis=0)
        std = np.minimum(val_std, 1e2)  # for visualization
        std = ndi.gaussian_filter(std, sigma=10)

        n_stds = 0.3
        for k in np.linspace(0, n_stds, 4):
            plt.fill_between(
                range(len(std)), (val_median - k * std), (val_median + k * std),
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
        save_name = os.path.join(out_dir, fig_name + '.jpg')
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
    new_paths = [os.path.join(dst_dir, '/'.join(dir.split('/')[1:-1]), get_new_name(dir.split('/')[-1])) for dir in
                 dirs]
    import shutil
    for source, dest in zip(dirs, new_paths):
        # print(f'{source} -> {dest}')
        shutil.move(source, dest)


def copy_jsons(src_dir, dst_dir):
    assert src_dir != dst_dir, 'src and dest cannot be the same'
    assert dst_dir is not None, 'dest cannot be None'

    lists = list(pathlib.Path(src_dir).glob('**/*.json'))
    statistics_src = [l.as_posix() for l in lists if 'statistics' in l.name]
    alls_src = [s.replace('statistics', 'all') for s in statistics_src]

    statistics_dst = [s.replace(src_dir, dst_dir, 1) for s in statistics_src]
    alls_dst = [s.replace('statistics', 'all') for s in statistics_dst]

    for src, dst in zip(statistics_src, statistics_dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)

    for src, dst in zip(alls_src, alls_dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)


def get_unique_keys(json_data):
    ll = [list(stat.keys()) for stat in json_data]
    ll = [item for sublist in ll for item in sublist]
    uniq = np.unique(ll)
    return uniq


def create_hparam_file(src_dir):
    lists = list(pathlib.Path(src_dir).glob('**/*.json'))
    statistics = [l.as_posix() for l in lists if 'statistics' in l.name]
    hparams_paths = [s.replace('statistics', 'hparams') for s in statistics]
    for hparams_path in hparams_paths:
        if not os.path.exists(hparams_path):
            _, agent, model, _ = hparams_path.split('/')[-4:]
            try:
                env, exploration, action_cost, beta, train_episode, seed, date = model.split('-')
            except ValueError:
                env, exploration, action_cost, beta, train_episode, seed, date, _ = model.split('-')

            action_cost = float(action_cost.replace('ac', ''))
            beta = float(beta.replace('b', ''))
            train_episode = float(train_episode.replace('ep', ''))
            seed = float(seed.replace('s', ''))
            params = {'agent': agent, 'env': env, 'exploration': exploration, 'action_cost': action_cost,
                      'beta': beta, 'train_episode': train_episode, 'seed': seed}

            with open(hparams_path, "w") as f:
                json.dump(params, f)


def add_fields_in_hparam_file(src_dir):
    hparams_paths = list(pathlib.Path(src_dir).glob('**/hparams.json'))
    envs, envs_not = [], []
    for hparams_path in hparams_paths:
        data = load_json(hparams_path)
        # if 'agent' in data:
        #     envs.append(data['agent'])

        if 'agent' not in data:
            print(hparams_path)
            agent = [fl for fl in hparams_path.as_posix().split('/') if 'Agent' in fl][0]
            data['agent'] = agent
            with open(hparams_path, "w") as f:
                json.dump(data, f)

        if 'env' not in data:
            env = hparams_path.as_posix().split('/')[-2].split('-')[0]
            envs_not.append(env)
            data['env'] = env
            with open(hparams_path, "w") as f:
                json.dump(data, f)

        envs = np.unique(envs)
        envs_not = np.unique(envs_not)


def main(args):
    src_dir = args.src_dir
    dst_dir = args.dst_dir
    if dst_dir is not None:
        os.makedirs(dst_dir, exist_ok=True)

    # dirs_renaming(src_dir, dst_dir)
    # copy_jsons(src_dir, dst_dir)
    create_hparam_file(src_dir)
    # add_fields_in_hparam_file(src_dir)

    # fetch data
    lists = list(pathlib.Path(src_dir).glob('**/*.json'))
    statistics_files = [l.as_posix() for l in lists if 'statistics' in l.name]

    # arange data
    data_statistics = [load_json(path) for path in statistics_files]
    data_statistics = [to_dict_list(d) for d in data_statistics]
    statistics_files = [l.as_posix() for l in lists if 'statistics' in l.name]
    hparams_files = [s.replace('statistics', 'hparams') for s in statistics_files]
    alls = [s.replace('statistics', 'all') for s in statistics_files]

    # # check unique features of statistics and  all
    # uniq_key_stat = get_unique_keys(data_statistics)
    # data_alls = [load_json(path) for path in alls]
    # data_alls = [to_dict_list(d) for d in data_alls]
    # uniq_key_all = get_unique_keys(data_alls)

    # remove invalid simulations
    valid_sim_inds = get_valid_simulations_indxes(data_statistics)
    data_statistics = [data_statistics[v] for v in valid_sim_inds]
    statistics_files = [statistics_files[v] for v in valid_sim_inds]
    hparams_files = [hparams_files[v] for v in valid_sim_inds]

    for h_file in hparams_files:
        assert os.path.isfile(h_file), f'hfile does not exist for {h_file}'
    hparams = [load_json(h_file) for h_file in hparams_files]

    # plot graphs

    [plot_key(data_statistics, statistics_files, hparams, save_key, dst_dir=dst_dir)  # , **args.__dict__)
     for save_key in args.keys_to_plot]

    # write last results (averaged over 20 iterations)
    str_list = [write_key_value(d, s_names, args.keys_to_plot)
                for (d, s_names) in zip(data_statistics, statistics_files)]
    print(np.sort(str_list))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameters for H-UCRL.")
    parser.add_argument("--src-dir", type=str, default="runs")
    parser.add_argument("--dst-dir", type=str, default=None)
    parser.add_argument('--keys-to-plot', type=str, nargs='+',
                        default=['train_return', 'sim_return', 'rewards', 'eval_return', 'epochs'])
    parser.add_argument("--figures-split", type=str, nargs='+', default=['env'])
    parser.add_argument("--subfigure-split", type=str, nargs='+', default=['action_cost'])

    main(parser.parse_args())
