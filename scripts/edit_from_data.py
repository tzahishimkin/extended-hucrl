import argparse
import os
import numpy as np
# from scripts.anlyse_results import edit_figs
from scripts.edit_from_fig import FigEditor
from scripts.data_change import *


def get_key_swaps(env_name):
    del_keys = [
        # 'BPTT-thompson-b1.0-helrTrue',
        # 'MPC-thompson-b1.0-helrTrue',
    ]
    rename_keys = [
        # new, old
        # 'SAC-expected-b1.0-helrTrue', 'BPTT-expected-b1.0-helrTrue'
        # '', '', ''
        ['expected', 'BPTT-expected-b1.0-helrTrue'],
        ['hucrl', 'SAC-expected-b1.0-helrTrue'],
        # ['sto-hucrl', 'SAC-expected-b1.0-helrTrue'],
        # ['hucrl', 'BPTT-optimistic-b1.0-helrTrue'],
    ]
    duplicate_keys = [
        ['shucrl', 'hucrl'],
    ]
    return del_keys, rename_keys, duplicate_keys

def load_data(data_path, env=None):
    data = np.load(data_path, allow_pickle=True)
    data = data.item()
    if env is not None:
        data = data[env]
    return data


def change_keys(data, del_keys=[], rename_keys=[], duplicate_keys=[]):
    data = keys_delete(data, del_keys)
    data = keys_rename(data, rename_keys)
    data = keys_duplicate(data, duplicate_keys)
    return data


def main(args):
    route_path = 'edited_figs_with_strong_smooth'

    import glob
    lls = []

    update_many = False
    if update_many:
        for dir in os.listdir(route_path):
            list_dirs = glob.glob(route_path + '/*/' + dir+'/data')
            ll = [os.path.join(list_dir, np.sort(os.listdir(list_dir))[-1]) for list_dir in list_dirs]
            lls.extend(ll)

        editor = FigEditor()
        for ll in lls:
            title = ll.split('\\')[1]
            data = load_data(ll)

            # editor = FigEditor(ll, exp_name=title, save_folder=os.path.join(route_path, title)) #), add_noise=[1.5,1])
            editor.set_all_data(ll, exp_name=title, save_folder=os.path.join(route_path, title)) #), add_noise=[1.5,1])
            editor.run()
        return
            # x_scale = 3 if 'Ant' in title else 1
            # editor.save(None, save_folder=None,  # new_keys=['gridy', 'hucrl', 'st-hucrl'],
            #             title=title, x_scale=x_scale, y_scale=1)  # , y_bounds=(0, 200))
            # del editor

    from_already_edited = True
    if from_already_edited:
        route_path = 'edited_figs'

        data = 'edited_figs/Ant-v2 action-noise=0 actu=0/data/2021_08_23_03_42_42.npy'
        env = 'Ant'
        editor = FigEditor()
        editor.set_all_data(data, exp_name=env, save_folder=route_path)
        editor.run()
        return
        # args.data_src = 'edited_figs/Uncertainty - Ant with stochastic dynamics error/data/2021_08_10_02_41_12.npy'
    else:
        route_path = 'edited_figs'
        data = 'np_data/train_return.npy'
        titles = [
            'MBAnt_v0_rllib-ac0.1-UncNone0-Una1',
            # 'MBInvertedPendulum_v0_rllib-ac0.01-UncNone0.5-Una1.0',
            # 'MBCartPole_v0_rllib-ac0.01-UncNone0.5-Una1.0',
            # 'MBPusher_v0_exps-ac0.5-UncNone0-Una1',
            # 'MBHopper_v0_exps-ac0.0-UncNone0-Una1',
        ]
        env = titles[0]
        data = load_data(data, env)

        editor = FigEditor()
        editor.set_all_data(data, exp_name=env, save_folder=route_path, filter_sigma=6)
        editor.run()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameters for data edit.")
    parser.add_argument("--data-src", type=str)
    parser.add_argument("--dst-dir", type=str)
    main(parser.parse_args())
