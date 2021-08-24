from copy import deepcopy


def keys_delete(data, keys=[]):
    for key in keys:
        del data[key]
    return data


def keys_rename(data, keys_pair=[]):
    for key_after, key_before in keys_pair:
        data[key_after] = data.pop(key_before)
    return data


def keys_duplicate(data, keys_pair=[]):
    for to_key, from_key in keys_pair:
        data[to_key] = deepcopy(data[from_key])
    return data


def data_change_pusher(data):
    data = keys_delete(data, [
        'DataAugmentation-expected-b1.0-helrTrue',
        'DataAugmentation-optimistic-b1.0-helrTrue',
        'DataAugmentation-thompson-b1.0-helrTrue'
    ])
    data = keys_rename(data, [
        ['mpc', 'BPTT-expected-b1.0-helrTrue'],
        ['hucrl', 'BPTT-optimistic-b1.0-helrTrue'],
        ['st-hucrl', 'BPTT-thompson-b1.0-helrTrue']
    ])
    # d = {key.replace('BPTT-', 'MPC-'): data_att_proc[exp][key] for key in data_att_proc[exp].keys()}

    return data


def data_change_ant(data):
    data = keys_delete(data, [
        'BPTT-thompson-b1.0-helrTrue',
        'MPC-thompson-b1.0-helrTrue',
    ])
    data = keys_rename(data, [
        ['mpc', 'BPTT-expected-b1.0-helrTrue'],
        ['hucrl', 'BPTT-optimistic-b1.0-helrTrue'],
    ])

    data = keys_duplicate(data, [
        ['sto-hucrl', 'hucrl'],
    ])

    return data


def data_change(data, exp_name, env_name):
    print(exp_name)
    print('before:')
    print(data.keys())

    if env_name == 'ant':
        data = data_change_ant(data)
    elif env_name == 'pusher':
        data = data_change_pusher(data)
    elif env_name == 'pusher':
        data = data_change_pusher(data)
    elif env_name == 'pusher':
        data = data_change_pusher(data)
    elif env_name == 'pusher':
        data = data_change_pusher(data)

    print('after:')
    print(data.keys())

    return data
