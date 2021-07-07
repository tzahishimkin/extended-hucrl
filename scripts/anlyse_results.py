import os
import pathlib
import json
from collections import defaultdict
from itertools import chain

import numpy as np
from matplotlib import pyplot as plt

src_dir = 'runs-gpu-before-naming'
# src_dir = 'runs'
out_dir = None

if out_dir is not None:
    os.makedirs(out_dir, exist_ok=True)


def load_json(path):
    with open(path) as f:
        data = json.load(f)
    return data


lists = list(pathlib.Path(src_dir).glob('**/*.json'))
statistics = [l.as_posix() for l in lists if 'statistics' in l.name]

data_statistics = [load_json(path) for path in statistics]


# res = defaultdict(list)
# A = {res[key].append(sub[key]) for sub in data_statistics[0] for key in sub}


def to_dict_list(list_dict):
    all_keys = set(chain(*[x.keys() for x in list_dict]))
    dict_list = {k: [d[k] for d in list_dict if k in d] for k in all_keys}
    return dict_list


data_statistics = [to_dict_list(d) for d in data_statistics]


def remove_invalid_sim(data_statistics, statistics, min_epochs=50):
    valid_sim = [len(d['rewards']) > min_epochs for d in data_statistics]
    valid_sim = np.where(valid_sim)[0]
    data_statistics = [data_statistics[v] for v in valid_sim]
    statistics = [statistics[v] for v in valid_sim]
    return data_statistics, statistics


data_statistics, statistics = remove_invalid_sim(data_statistics, statistics)


def plot_key(d, s_names, key='train_return'):
    if key not in d:
        return
    rewards = d[key]
    plt.plot(rewards)
    plt.title(key)
    if out_dir is not None:
        l_names = s_names.split('/')[-3:-1]
        l_names.insert(0, out_dir)
        l_names = [l.replace('.', '_') for l in l_names]
    else:
        l_names = s_names.split('/')[:-1]

    l_names.insert(-1, key)
    save_name = os.path.join(*l_names)
    os.makedirs(os.path.dirname(save_name), exist_ok=True)
    plt.savefig(save_name)
    plt.close()


save_keys = ['train_return', 'sim_return']
# [[plot_key(d, s_names, save_key) for (d, s_names) in zip(data_statistics, statistics)] for save_key in save_keys]

def write_key_value(d, s_names, keys):
    p_str = l_names = '/'.join(s_names.split('/')[1:-1])
    p_str += ': '
    any_key_ok = False

    n_epochs = len(d['train_return'])

    for key in keys:
        if key not in d:
            continue
        values = d[key]
        values = values[-20:]
        mvalue = np.mean(values)
        p_str += f'{key}={mvalue:.1f},  '
        any_key_ok = True
    if any_key_ok:
        # p_str += f'epoch={n_epochs}'
        return p_str

save_keys = ['train_return', 'eval_return']
str_list = [write_key_value(d, s_names, save_keys) for (d, s_names) in zip(data_statistics, statistics)]
print(np.sort(str_list))
