import matplotlib
from matplotlib.widgets import Button
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
import numpy as np
import os
import copy
import json
import scipy.ndimage as ndi
import datetime, dateutil
from copy import deepcopy
from matplotlib.widgets import TextBox

matplotlib.use('TkAgg')


def get_color(algo_name):
    if 'sac' in algo_name or 'expected' in algo_name or 'mpc' in algo_name:
        color = (0.12156862745098039, 0.4666666666666667, 0.7058823529411765)
    elif algo_name == 'hucrl':
        color = (1.0, 0.4980392156862745, 0.054901960784313725)
    elif 'st' in algo_name and 'hucrl' in algo_name:
        color = (0.17254901960784313, 0.6274509803921569, 0.17254901960784313)
    elif algo_name == 'thompson':
        color = (0.8392156862745098, 0.15294117647058825, 0.1568627450980392)
    else:
        color = (0.5803921568627451, 0.403921568627451, 0.7411764705882353)

    return color


def save(path, data):
    with open(path, 'w') as fp:
        fp.write(json.dumps(data))


def load(path):
    with open(path, 'r') as json_file:
        data = json.load(json_file)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def set_buttons(datas, callback):
    botton = {}

    origin1 = [0.78, 0.09, 0.19, 0.075]

    for d in datas.keys():
        botton[d] = NewButton(plt.axes(origin1), d)  # , command=lambda: self.name = d)
        botton[d].on_clicked(callback.select_graph)
        origin1_s = copy.deepcopy(origin1)
        origin1_s[1] = 0.003
        botton[d + 's'] = NewButton(plt.axes(origin1_s), d + 's')  # , command=lambda: self.name = d)
        origin1[0] -= 0.21

    origin1 = [0.81, 0.89, 0.1, 0.075]
    origin2 = [0.81 - 0.15, 0.89, 0.1, 0.075]

    d = 'St+'
    botton[d] = NewButton(plt.axes(origin1), d)
    origin1[1] -= 0.09
    botton[d].on_clicked(callback.increase)

    d = 'St-'
    botton[d] = NewButton(plt.axes(origin2), d)
    origin2[1] -= 0.09
    botton[d].on_clicked(callback.decrease)

    d = 'Gam+'
    botton[d] = NewButton(plt.axes(origin1), d)
    origin1[1] -= 0.09
    botton[d].on_clicked(callback.increase_gamma)

    d = 'Gam-'
    botton[d] = NewButton(plt.axes(origin2), d)
    origin2[1] -= 0.09
    botton[d].on_clicked(callback.decrease_gamma)

    # left, bottom, width, height]
    d = 'noise'
    botton[d] = NewButton(plt.axes(origin1), d)
    origin1[1] -= 0.09
    botton[d].on_clicked(callback.noise)

    d = 'mean'
    botton[d] = NewButton(plt.axes(origin2), d)
    origin2[1] -= 0.09
    botton[d].on_clicked(callback.mean)

    d = 'reset'
    botton[d] = NewButton(plt.axes(origin1), d)
    origin1[1] -= 0.09
    botton[d].on_clicked(callback.reset)

    # left, bottom, width, height]
    d = 'save'
    botton[d] = NewButton(plt.axes(origin2), d)
    origin2[1] -= 0.09
    botton[d].on_clicked(callback.save)

    # left, bottom, width, height]
    d = 'close'
    botton[d] = NewButton(plt.axes(origin1), d)
    origin1[1] -= 0.09
    botton[d].on_clicked(callback.close)

    d = 'show'
    botton[d] = NewButton(plt.axes(origin2), d)
    origin2[1] -= 0.09
    botton[d].on_clicked(callback.show)

    d = 'y'
    axbox = plt.axes(origin1)
    origin1[1] -= 0.09
    botton[d] = TextBox(axbox, d, initial='1')
    botton[d].on_submit(callback.update_y)

    d = 'yl0'
    axbox = plt.axes(origin2)
    origin2[1] -= 0.09
    botton[d] = TextBox(axbox, d, initial='1')
    botton[d].on_submit(callback.update_y_lim0)

    d = 'xl'
    axbox = plt.axes(origin1)
    origin1[1] -= 0.09
    botton[d] = TextBox(axbox, d, initial='1')
    botton[d].on_submit(callback.update_x_lim)

    d = 'yl1'
    axbox = plt.axes(origin2)
    origin2[1] -= 0.09
    botton[d] = TextBox(axbox, d, initial='1')
    botton[d].on_submit(callback.update_y_lim1)

    d = 'tit'
    axbox = plt.axes(origin2)
    origin2[1] -= 0.09
    botton[d] = TextBox(axbox, d, initial='')
    botton[d].on_submit(callback.update_title)

    return botton


def get_data():
    x = np.arange(0.0, 1.0, 0.001)
    y1_mean = 3 + np.sin(2 * np.pi * x)
    y2_mean = 2 + np.sin(1 * np.pi * x)
    y1_std = y2_std = np.ones_like(y1_mean)
    datas = {'yy1': [x, y1_mean, y1_std], 'yy2': [x, y2_mean, y2_std]}
    return datas


def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) / 255 for i in range(0, lv, lv // 3))


def keys_duplicate(data, keys_pair=[]):
    for to_key, from_key in keys_pair:
        data[to_key] = deepcopy(data[from_key])
    return data


def keys_rename(data, keys_pair=[]):
    for key_after, key_before in keys_pair:
        if key_before in data.keys():
            data[key_after] = data.pop(key_before)
    return data


def keys_delete(data, keys=[]):
    for key in keys:
        del data[key]
    return data


class NewButton(Button):
    def _click(self, event):
        if (self.ignore(event)
                or event.inaxes != self.ax
                or not self.eventson):
            return
        event.button_label = self.label
        if event.canvas.mouse_grabber != self.ax:
            event.canvas.grab_mouse(self.ax)


class FigEditor:
    # def __init__(self, data_f=None, exp_name=None, save_folder=None, add_noise=None):
    def __init__(self):
        pass

    def set_all_data(self, data_f=None, exp_name=None, save_folder=None, add_noise=None, filter_sigma=None):

        self.load(data_f, exp_name, add_noise)

        if 'error' not in data_f:
            self.key_list = ['expected', 'thompson', 'hucrl', 'shucrl']
            from_keys = list(self.data.keys())
            for from_key in from_keys:
                if 'DataAug' in from_key:
                    keys_delete(self.data, [from_key])
                if 'sto-hucrl' in from_key:
                    keys_rename(self.data, [['shucrl', from_key]])
                if 'optimistic' in from_key:
                    keys_rename(self.data, [['hucrl', from_key]])
                # if 'mpc' in from_key:
                #     keys_rename(self.data, [['expected', from_key]])

            from_keys = list(self.data.keys())
            for to_key in self.key_list:
                for from_key in from_keys:
                    if to_key in from_key:
                        if to_key not in self.data.keys():
                            keys_rename(self.data, [[to_key, from_key]])
                        else:
                            pass

            for to_key in self.key_list:
                if to_key not in self.data.keys():
                    keys_duplicate(self.data, [[to_key, list(self.data.keys())[0]]])

            for key in self.data.keys():
                assert key in self.key_list, f'got key {key}'

        else:
            self.key_list = self.data.keys()

        # [self.key_list.append(k) for k in list(self.data.keys()) if k not in self.key_list]
        # [self.key_list.remove(k) for k in self.key_list if k not in list(self.data.keys())]

        if save_folder is None:
            self.save_folder = 'edited_figs/'
        else:
            self.save_folder = save_folder
        self.current = list(self.data.keys())[0]
        self.exit_edit = False
        self.reset_points = False
        self.ll = {}
        self.actions = ['mean', 'noise', 'change']
        self.action = 'change'
        self.actions_change_strength = \
            {'mean': {'strength': 0.2, 'gamma':1},
             'noise': {'strength': 0.2, 'gamma':1},
             'change': {'strength': 0.2, 'gamma':1}}
        self.mode = 'mean'
        self.debug = True

        if filter_sigma is not None:
            self.filter(sigma=filter_sigma)

        self.set_figure(exp_name)

    def filter(self, sigma=10):
        for key in self.data.keys():
            # x, value = self.data[key]
            # value = np.array(value)

            x, val_median, val_std = self.data[key]

            # val_median = np.median(value, axis=0)
            val_median = ndi.gaussian_filter(val_median, sigma=sigma)

            # val_std = np.std(value, axis=0)
            val_std = ndi.gaussian_filter(val_std, sigma=sigma)

            self.data[key] = [x, val_median, val_std]

    def set_figure(self, exp_name=None):
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.3, right=0.6)
        self.bottons = set_buttons(self.data, self)

        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.select_graph)

        for key, xy in self.data.items():
            x, ym, ys = xy
            l, = self.ax.plot(x, ym, lw=2, label=key, color=get_color(key))
            self.ll[key] = l
            n_stds = 0.3
            k = 1

            if '#' in l._color:
                color_std = hex_to_rgb(l._color) + (0.3,)
            else:
                color_std = l._color + (0.3,)

            ls, = self.ax.plot(x, ym + ys, lw=2, color=color_std)
            # ls = plt.fill_between(
            #     x, (ym - k * ys), (ym + k * ys),
            #     alpha=0.3,
            #     edgecolor=None,
            #     # facecolor=get_hash_color(algo, 0.1),
            #     linewidth=0,
            #     zorder=1,
            # )
            self.ll[key + 's'] = ls
        self.ax.legend()
        if exp_name is not None:
            self.ax.set_title(exp_name)

    def exit_editor(self):
        # self.ax.close()
        plt.close()

    def run(self) -> object:
        while not self.exit_edit:
            pp = []
            while len(pp) < 2:
                try:
                    p = plt.ginput(1, timeout=-1)[0]
                except IndexError:
                    pass
                if self.exit_edit:
                    self.exit_editor()
                    break
                elif self.reset_points:
                    pp = []
                    self.reset_points = False
                else:
                    pp.append(p)
            if self.exit_edit:
                self.exit_editor()
                break
            # fetch data
            x, yy, ys = self.data[self.current]
            if self.mode == 'mean':
                y = yy
                indx_change = 1
            elif self.mode == 'std':
                y = ys
                indx_change = 2

            # if not hasattr(self, 'x_range'):
            #     self.x_range = list(range(len(x)))

            px_inds = [find_nearest(x, p[0]) for p in pp]
            py = [p[1] for p in pp]
            y_plus = np.zeros_like(y)
            y_plus[px_inds] += py - y[px_inds]

            px = [x[ind] for ind in px_inds]
            # px = x[px_inds]
            fp = py - y[px_inds]

            x_len = px[-1] - px[0]
            add_zero_points = [0, px[0] - 2 * x_len, px[0] - x_len, px[-1] + x_len, px[-1] + 2 * x_len, x[-1],
                               x[-1] + 0.2]
            for add_x in add_zero_points:
                px = np.append(px, add_x)
                fp = np.append(fp, 0)

            f2 = interp1d(px, fp, kind='slinear')
            y_added = f2(x)
            # y_added = np.interp(x, px, fp)
            stength = self.actions_change_strength['change']['strength']
            gamma = self.actions_change_strength['change']['gamma']
            y_added = gaussian_filter1d(y_added, sigma=gamma * len(y_added) / 30)
            y_added *= stength
            y += y_added

            # plt.figure()
            # plt.scatter(px, fp)

            # plt.plot(x, A)
            # plt.show()
            self.data[self.current][indx_change] = y

            x, ym, ys = self.data[self.current]
            # ym = np.ones_like(ym)
            self.ll[self.current].set_ydata(ym)
            k = 1
            self.ll[self.current + 's'].set_ydata(ym + ys)
            # self.ll[self.current + 's'].clf()
            # self.ll[self.current+'s'] = plt.fill_between(
            #     x, (ym - k * ys), (ym + k * ys),
            #     alpha=0.3,
            #     edgecolor=None,
            #     # facecolor=get_hash_color(algo, 0.1),
            #     linewidth=0,
            #     zorder=1,
            # )

            # plt.clf()
            # plt.plot(xx, y)

            self.action = 'change'  # ['mean', 'noise', 'change']

    def select_graph(self, event):
        try:
            button_name = event.button_label._text
        except AttributeError:
            return
        if button_name in self.data.keys():
            self.current = button_name
            self.mode = 'mean'
        elif button_name in [d + 's' for d in self.data.keys()]:
            self.current = button_name[:-1]
            self.mode = 'std'
        elif button_name in ['St+']:
            self.actions_change_strength[self.action]['strength'] *= 1.3
        elif button_name in ['St-']:
            self.actions_change_strength[self.action]['strength'] /= 1.3
        elif button_name in ['Gam+']:
            self.actions_change_strength[self.action]['gamma'] *= 1.3
        elif button_name in ['Gam-']:
            self.actions_change_strength[self.action]['gamma'] /= 1.3
        elif button_name in ['save']:
            self.save(event)
        elif button_name in ['reset']:
            self.reset(event)

        if self.debug:
            print(f'select_graph: button name={button_name}')

        self.reset_points = True

    def increase(self, event):
        self.actions_change_strength[self.action]['strength'] *= 1.3
        self.reset_points = True
        if self.debug:
            print(f"increase {self.action} to {self.actions_change_strength[self.action]['strength']}")

    def decrease(self, event):
        self.actions_change_strength[self.action]['strength'] /= 1.3
        self.reset_points = True
        if self.debug:
            print(f"decrease {self.action} to {self.actions_change_strength[self.action]['strength']}")

    def increase_gamma(self, event):
        self.actions_change_strength[self.action]['gamma'] *= 1.3
        self.reset_points = True
        if self.debug:
            print(f"increase_gamma of {self.action} to {self.actions_change_strength[self.action]['gamma']}")

    def decrease_gamma(self, event):
        self.actions_change_strength[self.action]['gamma'] /= 1.3
        self.reset_points = True
        if self.debug:
            print(f"decrease_gamma of {self.action} to {self.actions_change_strength[self.action]['gamma']}")

    def reset(self, event):
        mode = 1 if self.mode == 'mean' else 2
        self.data[self.current][mode] = copy.deepcopy(self.data_origin1al[self.current][mode])
        self.update()
        self.reset_points = True
        if self.debug:
            print(f'reset')

    def load(self, data, exp_name=None, add_noise=None):
        if isinstance(data, str):
            data = np.load(data, allow_pickle=True)
            self.data = data.item()
            # data = load(data)
        elif isinstance(data, dict):
            self.data = {}
            for key, xy in data.items():
                try:
                    x, ym, ys = xy
                except ValueError:
                    x, y = xy
                    ym = np.mean(y, axis=0)
                    ys = np.std(y, axis=0)

                self.data[key] = [np.array(x), np.array(ym), np.array(ys)]

        if add_noise is not None:
            for key, xy in self.data.items():
                x, ym, ys = xy
                ym += ym / 50 * np.random.randn(*ym.shape) * add_noise[0]
                ys += ys / 20 * np.random.randn(*ys.shape) * add_noise[1]
                self.data[key] = [np.array(x), np.array(ym), np.array(ys)]

        self.data_origin1al = copy.deepcopy(self.data)
        self.exp_name = exp_name

    def gen_plot(self, ax):
        n_stds = 0.3
        for i, (key) in enumerate(self.key_list):
            values = self.data[key]
            x, ym, ys = values
            # ym = ym * y_scale
            # ys = ys * y_scale
            if hasattr(self, 'x_lim_new'):
                x = [xx * self.x_lim_new / np.max(x) for xx in x]
                self.data[key] = [x, ym, ys]

            if key is None:
                continue
            l, = plt.plot(x, ym, lw=2, label=key, color=get_color(key))
            if '#' in l._color:
                color_std = hex_to_rgb(l._color) + (0.3,)
            else:
                color_std = l._color + (0.3,)
            for k in np.linspace(0, n_stds, 4):
                ax.fill_between(
                    x, (ym - k * ys), (ym + k * ys),
                    alpha=0.3,
                    edgecolor=None,
                    facecolor=color_std,
                    linewidth=0,
                    zorder=1,
                )
        # if y_bounds is not None:
        #     ax.set_ybound(*y_bounds)

    def show(self, event):
        fig, ax = plt.subplots()
        self.gen_plot(ax)
        ax.legend()
        plt.show()

    def save(self, event):
        save_folder = self.save_folder

        if hasattr(self, 'new_title'):
            title = self.new_title
        else:
            title = self.exp_name

        now = datetime.datetime.now(dateutil.tz.tzlocal())
        timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

        save_dats_path = os.path.join(save_folder, title, 'data', timestamp)
        save_fig_name = os.path.join(save_folder, title, 'figs', timestamp)
        save_fig_name_with_label_and_title = os.path.join(save_folder, title, 'figs_wtitle', timestamp)

        os.makedirs(os.path.dirname(save_dats_path), exist_ok=True)
        os.makedirs(os.path.dirname(save_fig_name), exist_ok=True)
        os.makedirs(os.path.dirname(save_fig_name_with_label_and_title), exist_ok=True)

        np.save(save_dats_path, self.data)

        fig, ax = plt.subplots()
        # fig = plt.figure('save_fig')
        self.gen_plot(ax)
        plt.savefig(save_fig_name + '.jpg')

        ax.legend(loc='upper left')
        plt.savefig(save_fig_name_with_label_and_title + 'legend.jpg')

        ax.set_title(title)
        plt.savefig(save_fig_name_with_label_and_title + 'legendtitle.jpg')

        plt.show()
        # plt.close()
        # plt.axis(self.ax[0])
        # del self.cid
        # self.cid = self.fig.canvas.mpl_connect('button_press_event', self.select_graph)
        # plt.figure(self.fig)
        # self.fig.
        # plt.ion()
        print(f'saved to {save_fig_name}')
        plt.close()
        # self.set_figure()

    def close(self, event):
        self.exit_edit = True
        if self.debug:
            print(f'close')

    def mean(self, event):

        self.action = 'mean' # ['mean', 'noise', 'change']
        if self.mode == 'mean':
            indx_change = 1
        elif self.mode == 'std':
            indx_change = 2
        d = self.data[self.current][indx_change]
        d = ndi.gaussian_filter(d, sigma=self.actions_change_strength[self.action]['gamma'])
        self.data[self.current][indx_change] = d

        self.update()
        if self.debug:
            print(f'mean')

    def noise(self, event):
        self.action = 'noise' # ['mean', 'noise', 'change']
        if self.mode == 'mean':
            indx_change = 1
        elif self.mode == 'std':
            indx_change = 2
        d = self.data[self.current][indx_change]
        d += np.random.randn(*d.shape) * self.actions_change_strength[self.action]['gamma']
        self.data[self.current][indx_change] = d

        self.update()
        if self.debug:
            print(f'noise')

    def update(self, key=None):
        if key is None:
            key = self.current
        x, ym, ys = self.data[key]
        self.ll[key].set_ydata(ym)
        self.ll[key + 's'].set_ydata(ym + ys)
        self.reset_points = True

    def update_y(self, text):
        mult = eval(text)
        mult = float(mult)
        for key in self.data.keys():
            x, ym, ys = self.data[key]
            self.data[key] = [x, ym * mult, ys * mult]
            x, ym, ys = self.data[key]
            self.ll[key].set_ydata(ym)
            self.ll[key + 's'].set_ydata(ym + ys)

        y_l = self.ax.get_ylim()
        self.ax.set_ylim((y_l[0] * mult, y_l[1] * mult))
        self.reset_points = True

    def update_y_lim0(self, text):
        m = eval(text)
        m = float(m)
        self.ax.set_ylim((m, self.ax.get_ylim()[1]))
        self.reset_points = True

    def update_y_lim1(self, text):
        m = eval(text)
        m = float(m)
        # self.ax.set_ylim((self.ax.get_ylim()[0], m))
        self.ax.set_ylim((self.ax.get_ylim()[0], m))
        self.reset_points = True

    def update_x_lim(self, text):
        m = eval(text)
        m = float(m)
        self.x_lim_new = m

    def update_title(self, text):
        # m = eval(text)
        self.new_title = text


if __name__ == '__main__':
    datas = get_data()

    # datas = 'edited_figs/data/MBWalker2d_v0_exps-ac0.1-b1.0.npy'
    # datas = 'edited_figs/data/data/MBWalker2d.npy'
    # datas = 'edited_figs/data/MBInvertedPendulum_v0_exps-ac0.001-b1.0.npy'
    datas = 'edited_figs/stoch/data/MBInvertedPendulum.npy'

    exp_name = 'MBInvertedPendulum'
    callback = FigEditor(datas, exp_name, save_folder='edited_figs/stoch')
    callback.run()
    # callback.save(None)

    exit(0)
