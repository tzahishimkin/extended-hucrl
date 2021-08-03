from collections import defaultdict
import scipy.ndimage as ndi
from itertools import chain
import numpy as np
import yaml
import hashlib
import copy
import os
import pathlib
import json
import argparse
from matplotlib import pyplot as plt
import matplotlib.cm
import shutil
import torch

from edit_from_fig import FigEditor
from scripts import download_from_server


def load_yaml(src_path):
    """Parse configuration file."""
    with open(src_path, "r") as file:
        data = yaml.safe_load(file)
    return data


def save_yaml(dst_path, data):
    """Parse configuration file."""
    with open(dst_path, "w") as file:
        data = yaml.dump(data, file)
    return data


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


def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f)


def to_dict_list(list_dict):
    all_keys = set(chain(*[x.keys() for x in list_dict]))
    dict_list = {k: [d[k] for d in list_dict if k in d] for k in all_keys}
    return dict_list


def get_valid_simulations_indxes(data_statistics, min_epochs=250):
    # simulations under min_epochs
    valid_idxs = [len(d['rewards']) >= min_epochs for d in data_statistics]
    valid_idxs = np.where(valid_idxs)[0]
    return valid_idxs


def get_invalid_sims_indx(data_statistics, max_epochs=249):
    invalid_idxs = [len(d['rewards']) < max_epochs for d in data_statistics]
    invalid_idxs = np.where(invalid_idxs)[0]
    return invalid_idxs


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


def delete_invalid_sims(invalidx, statistics_files):
    for i in invalidx:
        shutil.rmtree(os.path.dirname(statistics_files[i]))


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


def fix_hparams_files(src_dir):
    hparams_paths = list(pathlib.Path(src_dir).glob('**/hparams.json'))
    udpate = False
    envs, envs_not = [], []
    for hparams_path in hparams_paths:
        data = load_json(hparams_path)

        if 'agent' in data and 'Agent' in data['agent']:
            data['agent'] = data['agent'].replace('Agent', '')
            udpate = True

        if 'agent' not in data:
            # print(hparams_path)
            agent = [fl for fl in hparams_path.as_posix().split('/') if 'Agent' in fl][0]
            data['agent'] = agent
            udpate = True
        if 'env' not in data:
            env = hparams_path.as_posix().split('/')[-2].split('-')[0]
            envs_not.append(env)
            data['env'] = env
            udpate = True
        if 'exp_name' not in data:
            # print(hparams_path)
            # agent = [fl for fl in hparams_path.as_posix().split('/') if 'Agent' in fl][0]
            data['exp_name'] = \
                '-'.join([
                    data['agent'],
                    data['exploration'],
                    f"b{data['beta']}"
                ])
            udpate = True

        if 'env_ac' not in data:
            data['env_ac'] = f"{data['env']}-ac{data['action_cost']}"
            udpate = True

        if udpate:
            with open(hparams_path, "w") as f:
                json.dump(data, f)

        # envs = np.unique(envs)
        # envs_not = np.unique(envs_not)


def plot_mean_std(x, ym, ys, title, path):
    color = get_hash_color(title, 1)
    plt.plot(x, ym, label=title, color=color)
    std = np.minimum(ys, 1e2)  # for visualization

    n_stds = 0.3
    for k in np.linspace(0, n_stds, 4):
        plt.fill_between(
            range(len(std)), (ym - k * std), (ym + k * std),
            alpha=0.3,
            edgecolor=None,
            facecolor=get_hash_color(title, 0.1),
            linewidth=0,
            zorder=1,
        )
    if path is not None:
        plt.savefig(path)


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
        try:
            exp = f"{hparam['env']}-ac{hparam['action_cost']}-b{hparam['beta']}"
            algo = f"{hparam['agent']}-{hparam['exploration']}"  # key
            try:
                datas.setdefault(f"{exp}--{algo}", []).append(data_statistic[attribute])
                print(statistics_file)
            except BaseException:
                continue
        except KeyError:
            print(statistics_file)

    data_exp = {}
    # for expalgo, data in datas.items():
    #     exp, algo, = expalgo.split('--')
    #     min_length = min([len(d) for d in data])
    #     data = [d[:min_length] for d in data]
    #     value = np.array(data)

    plot_original_figs = True

    for expalgo, data in datas.items():
        if data == []:
            continue

        exp, algo, = expalgo.split('--')
        min_length = min([len(d) for d in data])
        data = [d[:min_length] for d in data]
        value = np.array(data)
        fig = plt.figure(exp)
        figs_list.append(exp)

        val_median = np.median(value, axis=0)
        val_median = ndi.gaussian_filter(val_median, sigma=10)
        val_std = np.std(value, axis=0)
        val_std = ndi.gaussian_filter(val_std, sigma=10)
        x = range(len(val_median))

        if plot_original_figs:
            plot_mean_std(x, val_median, val_std, algo, path=None)

        if exp in data_exp:
            assert algo not in data_exp, f'{algo} already in data_exp[{exp}]'
        data_exp.setdefault(f"{exp}", {})[algo] = [x, val_median, val_std]

    edit_figs = False
    if edit_figs:
        for exp in data_exp.keys():
            if 'MBInver' in exp and 'exp' in exp:
                print(exp)
                data_exp[exp].keys()
                d = {key.replace('BPTT-', 'MPC-'): data_exp[exp][key] for key in data_exp[exp].keys()}
                d.keys()
                d['hucrl'] = d.pop('MPC-optimistic')
                d['greedy'] = d.pop('MPC-expected')
                d['sto-hucrl'] = copy.deepcopy(d['hucrl'])
                d['hucrl'] = copy.deepcopy(d['greedy'])
                d['sto-hucrl'] = copy.deepcopy(d['hucrl'])

                d.keys()

                editor = FigEditor(d, exp, save_folder=f'edited_figs')
                editor.run()
                editor.save(None)  # , new_keys=['MPC', None], title='Walker2d with action cost = 0.1')
                del editor

    if plot_original_figs:
        for fig_name in figs_list:
            fig = plt.figure(fig_name)
            # algo = key #f'{agent}-{exploration}'
            plt.legend()
            save_name = os.path.join(out_dir, fig_name + '.jpg')
            plt.title(fig_name)
            if os.path.exists(save_name):
                os.remove(save_name)
            plt.savefig(save_name)

        for fig_name in figs_list:
            fig = plt.figure(fig_name)
            plt.close()

    # for key, value in data.items():
    #     save_name = os.path.join(out_dir, key+'.jpg')
    #     plt.savefig(save_name)


def main(args):
    src_dir = args.src_dir

    dst_dir = args.dst_dir
    if dst_dir is not None:
        os.makedirs(dst_dir, exist_ok=True)

    # dirs_renaming(src_dir, dst_dir)
    # copy_jsons(src_dir, dst_dir)
    # create_hparam_file(src_dir)
    # fix_hparams_files(src_dir)

    # if on CPU computer then delete pkl files
    if not torch.cuda.is_available():
        [os.remove(l.as_posix()) for l in pathlib.Path(src_dir).glob(f'**/*.pkl') if 'run' in l.as_posix()]

    # download jsons from servers
    # download_from_server.scp_copy(download_from_server.get_defualt_hucrl_args())
    # os.system('python scripts/download_from_server.py')

    # fetch json data files data
    stats_file_name = 'statistics'
    lists = list(pathlib.Path(src_dir).glob(f'**/{stats_file_name}.json'))
    assert lists != [], 'file list is empty. check src_dir'
    statistics_files = [l.as_posix() for l in lists]

    # load and arange data
    data_statistics = [load_json(path) for path in statistics_files]
    data_statistics = [to_dict_list(d) for d in data_statistics]

    # each stats data should have an hparam.json file and posibily an all.json file
    hparams_files = [s.replace('statistics', 'hparams') for s in statistics_files]
    alls = [s.replace('statistics', 'all') for s in statistics_files]

    # # check unique features of statistics and  all
    # uniq_key_stat = get_unique_keys(data_statistics)
    # data_alls = [load_json(path) for path in alls]
    # data_alls = [to_dict_list(d) for d in data_alls]
    # uniq_key_all = get_unique_keys(data_alls)

    # remove invalid simulations
    invalidx = get_invalid_sims_indx(data_statistics, max_epochs=145)
    # delete_invalid_sims(invalidx, statistics_files)

    valid_sim_inds = get_valid_simulations_indxes(data_statistics, min_epochs=150)
    data_statistics = [data_statistics[v] for v in valid_sim_inds]
    statistics_files = [statistics_files[v] for v in valid_sim_inds]
    hparams_files = [hparams_files[v] for v in valid_sim_inds]

    for h_file in hparams_files:
        assert os.path.isfile(h_file), f'hfile does not exist for {h_file}'
    hparams = [load_json(h_file) for h_file in hparams_files]

    def update_ac(hparam_path):
        hparam = load_json(hparam_path)
        assert 'env_config' in hparam, 'hparam is missing'
        # shutil.rmtree(os.path.dirname(hparam_path))

        yml_file = load_yaml(hparam['env_config'])
        new_ac_val = yml_file.get('action_cost', 0)
        new_ac = f'ac{new_ac_val}'
        old_dir = os.path.dirname(hparam_path)
        old_ac = [l for l in old_dir.split('-') if 'ac' in l][0]
        new_dir = old_dir.replace(old_ac, new_ac)
        if new_ac != old_ac:
            print(old_dir)
            assert not os.path.exists(new_dir), 'dir exists'
            hparam['action_cost'] = new_ac_val
            os.path.exists(old_dir)
            os.remove(hparam_path)
            save_json(hparam_path, hparam)
            shutil.move(old_dir, new_dir)

    # update_ac(hparams_files[0])
    # [update_ac(h_file) for h_file in hparams_files]

    # plot graphs
    [plot_key(data_statistics, statistics_files, hparams, save_key, dst_dir=dst_dir)  # , **args.__dict__)
     for save_key in args.keys_to_plot]

    print_simulations_resolves = False
    if print_simulations_resolves:
        # write last results (averaged over 20 iterations)
        str_list = [write_key_value(d, s_names, args.keys_to_plot)
                    for (d, s_names) in zip(data_statistics, statistics_files)]
        print(np.sort(str_list))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameters for H-UCRL.")
    parser.add_argument("--src-dir", type=str, default="runs")
    parser.add_argument("--dst-dir", type=str, default=None)
    parser.add_argument('--keys-to-plot', type=str, nargs='+',
                        # default=['train_return', 'sim_return', 'rewards', 'eval_return', 'epochs'])
                        default=['dyn-loss'])
    parser.add_argument("--figures-split", type=str, nargs='+', default=['env'])
    parser.add_argument("--subfigure-split", type=str, nargs='+', default=['action_cost'])

    main(parser.parse_args())
