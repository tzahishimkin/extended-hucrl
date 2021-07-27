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

# matplotlib.use('TkAgg')


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

    origin = [0.78, 0.09, 0.19, 0.075]

    for d in datas.keys():
        botton[d] = NewButton(plt.axes(origin), d)  # , command=lambda: self.name = d)
        botton[d].on_clicked(callback.select_graph)
        origin_s = copy.deepcopy(origin)
        origin_s[1] = 0.003
        botton[d + 's'] = NewButton(plt.axes(origin_s), d + 's')  # , command=lambda: self.name = d)
        origin[0] -= 0.21

    origin = [0.81, 0.89, 0.1, 0.075]

    d = 'St+'
    botton[d] = NewButton(plt.axes(origin), d)
    origin[1] -= 0.09
    botton[d].on_clicked(callback.increase)

    d = 'St-'
    botton[d] = NewButton(plt.axes(origin), d)
    origin[1] -= 0.09
    botton[d].on_clicked(callback.decrease)

    d = 'Gam+'
    botton[d] = NewButton(plt.axes(origin), d)
    origin[1] -= 0.09
    botton[d].on_clicked(callback.increase_gamma)

    d = 'Gam-'
    botton[d] = NewButton(plt.axes(origin), d)
    origin[1] -= 0.09
    botton[d].on_clicked(callback.decrease_gamma)

    # left, bottom, width, height]
    d = 'mean'
    botton[d] = NewButton(plt.axes(origin), d)
    origin[1] -= 0.09
    botton[d].on_clicked(callback.mean)

    d = 'reset'
    botton[d] = NewButton(plt.axes(origin), d)
    origin[1] -= 0.09
    botton[d].on_clicked(callback.reset)

    # left, bottom, width, height]
    d = 'save'
    botton[d] = NewButton(plt.axes(origin), d)
    origin[1] -= 0.09
    botton[d].on_clicked(callback.save)

    # left, bottom, width, height]
    d = 'close'
    botton[d] = NewButton(plt.axes(origin), d)
    origin[1] -= 0.09
    botton[d].on_clicked(callback.save)

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
    def __init__(self, data, exp_name, save_folder):

        if isinstance(data, str):
            data = np.load(data, allow_pickle=True)
            data = data.item()
            # data = load(data)
        elif isinstance(data, dict):
            for key, xy in data.items():
                x, ym, ys = xy
                data[key] = [np.array(x), np.array(ym), np.array(ys)]

        self.data_original = copy.deepcopy(data)
        self.data = data
        self.save_folder = save_folder
        self.exp_name = exp_name
        self.current = list(data.keys())[0]
        self.exit_edit = False
        self.reset_points = False
        self.ll = {}
        self.strength = 0.2
        self.strength_gamma = 1
        self.mode = 'mean'
        self.debug = True
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.3, right=0.75)
        self.bottons = set_buttons(self.data, self)

        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.select_graph)

        for key, xy in data.items():
            x, ym, ys = xy
            l, = self.ax.plot(x, ym, lw=2, label=key)
            self.ll[key] = l
            n_stds = 0.3
            k = 1
            ls, = self.ax.plot(x, ym + ys, lw=2, label=key + 's', color=hex_to_rgb(l._color) + (0.3,))
            # ls = plt.fill_between(
            #     x, (ym - k * ys), (ym + k * ys),
            #     alpha=0.3,
            #     edgecolor=None,
            #     # facecolor=get_hash_color(algo, 0.1),
            #     linewidth=0,
            #     zorder=1,
            # )
            self.ll[key + 's'] = ls

    def run(self):
        while not self.exit_edit:
            pp = []
            while len(pp) < 2:
                p = plt.ginput(1, timeout=-1)[0]
                if self.exit_edit:
                    break
                elif self.reset_points:
                    pp = []
                    self.reset_points = False
                else:
                    pp.append(p)
            if self.exit_edit:
                break
            # fetch data
            x, yy, ys = self.data[self.current]
            if self.mode == 'mean':
                y = yy
                indx_change = 1
            elif self.mode == 'std':
                y = ys
                indx_change = 2

            px_inds = [find_nearest(x, p[0]) for p in pp]
            py = [p[1] for p in pp]
            y_plus = np.zeros_like(y)
            y_plus[px_inds] += py - y[px_inds]

            px = x[px_inds]
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
            y_added = gaussian_filter1d(y_added, sigma=self.strength_gamma * len(y_added) / 30)
            y_added *= self.strength
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
            self.strength *= 1.3
        elif button_name in ['St-']:
            self.strength /= 1.3
        elif button_name in ['Gam+']:
            self.strength_gamma *= 1.3
        elif button_name in ['Gam-']:
            self.strength_gamma /= 1.3
        elif button_name in ['save']:
            self.save(event)
        elif button_name in ['reset']:
            self.reset(event)

        if self.debug:
            print(f'select_graph: button name={button_name}')

        self.reset_points = True

    def increase(self, event):
        self.strength *= 1.3
        self.reset_points = True
        if self.debug:
            print(f'increase')

    def decrease(self, event):
        self.strength /= 1.3
        self.reset_points = True
        if self.debug:
            print(f'decrease')

    def increase_gamma(self, event):
        self.strength_gamma *= 1.3
        self.reset_points = True
        if self.debug:
            print(f'increase_gamma')

    def decrease_gamma(self, event):
        self.strength_gamma /= 1.3
        self.reset_points = True
        if self.debug:
            print(f'decrease_gamma')

    def reset(self, event):
        self.data[self.current][1] = copy.deepcopy(self.data_original[self.current][1])
        self.update()
        self.reset_points = True
        if self.debug:
            print(f'reset')

    def save(self, event, new_keys=None, title=None):
        save_folder = self.save_folder
        os.makedirs(save_folder, exist_ok=True)
        data_dir = os.path.join(save_folder, 'data')
        os.makedirs(data_dir, exist_ok=True)
        np.save(os.path.join(data_dir, self.exp_name), self.data)

        n_stds = 0.3
        fig, ax = plt.subplots()
        for i, (key, values) in enumerate(self.data.items()):
            x, ym, ys = values
            if not new_keys is None:
                key = new_keys[i]
            if key is None:
                continue
            l, = ax.plot(x, ym, lw=2, label=key)
            for k in np.linspace(0, n_stds, 4):
                plt.fill_between(
                    x, (ym - k * ys), (ym + k * ys),
                    alpha=0.3,
                    edgecolor=None,
                    facecolor=hex_to_rgb(l._color) + (0.5,),
                    linewidth=0,
                    zorder=1,
                )
        # plt.legend()
        # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.legend(loc='upper left')

        save_fig_name = os.path.join(save_folder, self.exp_name + '.jpg')
        plt.savefig(save_fig_name)
        # plt.close()
        print(f'saved to {save_folder}')
        self.exit_edit = True
        if self.debug:
            print(f'saved to {self.save_folder}')

    def close(self, event):
        self.exit_edit = True
        if self.debug:
            print(f'close')

    def mean(self, event):
        if self.mode == 'mean':
            indx_change = 1
        elif self.mode == 'std':
            indx_change = 2
        d = self.data[self.current][indx_change]
        d = ndi.gaussian_filter(d, sigma=10)
        self.data[self.current][indx_change] = d

        self.update()
        if self.debug:
            print(f'mean')

    def update(self):
        x, ym, ys = self.data[self.current]
        self.ll[self.current].set_ydata(ym)
        self.ll[self.current + 's'].set_ydata(ym + ys)
        self.reset_points = True


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
