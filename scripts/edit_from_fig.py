import matplotlib
from matplotlib.widgets import Button
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
import numpy as np
import os
import copy

matplotlib.use('TkAgg')


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


class NewButton(Button):
    def _click(self, event):
        if (self.ignore(event)
                or event.inaxes != self.ax
                or not self.eventson):
            return
        event.button_label = self.label
        if event.canvas.mouse_grabber != self.ax:
            event.canvas.grab_mouse(self.ax)


class Handler:
    def __init__(self, data, save_dir):
        self.data_original = copy.deepcopy(data)
        self.data = data
        self.save_dir = save_dir
        self.current = list(data.keys())[0]
        self.exit_edit = False
        self.reset_points = False
        self.ll = {}
        self.strength = 1
        self.strength_gamma = 1
        for key, xy in data.items():
            x, y = xy
            l, = plt.plot(x, y, lw=2)
            self.ll[key] = l

    def runner(self):
        while not self.exit_edit:
            pp = []
            while len(pp) < 3:
                p = plt.ginput(1, timeout=-1)[0]
                if self.exit_edit:
                    break
                if self.reset_points:
                    pp = []
                    self.reset_points = False
                pp.append(p)
            if self.exit_edit:
                break

            xx, yy = self.data[self.current]
            # plt.plot(xx, yy)
            print(pp)

            px = [find_nearest(xx, p[0]) for p in pp]
            py = [p[1] for p in pp]
            y_plus = np.zeros_like(yy)
            y_plus[px] += py - yy[px]

            x = range(len(yy))
            xp = px
            fp = py - yy[px]

            x_len = px[-1] - px[0]
            add_zero_points = [0, px[0] - 2 * x_len, px[0] - x_len, px[-1] + x_len, px[-1] + 2 * x_len, len(yy)]
            for add_x in add_zero_points:
                xp = np.append(xp, add_x)
                fp = np.append(fp, 0)

            f2 = interp1d(xp, fp, kind='slinear')
            yy_added = f2(x)
            # yy_added = np.interp(x, xp, fp)
            yy_added = gaussian_filter1d(yy_added, sigma=self.strength_gamma * len(yy_added) / 30)
            yy_added *= self.strength
            yy += yy_added

            # plt.figure()
            # plt.scatter(xp, fp)

            # plt.plot(x, A)
            # plt.show()
            np.any(yy - self.data_original[self.current][1])
            self.data[self.current] == yy
            self.ll[self.current].set_ydata(yy)
            # plt.clf()
            # plt.plot(xx, yy)

    def select_graph(self, event):
        button_name = event.button_label._text
        if button_name in self.data.keys():
            self.current = button_name
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

        self.reset_points = True

    def increase(self, event):
        self.strength *= 1.3
        self.reset_points = True

    def decrease(self, event):
        self.strength /= 1.3
        self.reset_points = True

    def increase_gamma(self, event):
        self.strength_gamma *= 1.3
        self.reset_points = True

    def decrease_gamma(self, event):
        self.strength_gamma /= 1.3
        self.reset_points = True

    def reset(self, event):
        self.data[self.current][1] = copy.deepcopy(self.data_original[self.current][1])
        self.ll[self.current].set_ydata(self.data[self.current][1])
        self.reset_points = True

    def save(self, event):
        save_path = os.path.join(self.save_dir, 'data')
        np.save(save_path, self.data)
        print(f'saved to {save_path}')
        self.exit_edit = True


def set_buttons(datas, callback):
    botton = {}
    origin = [0.81, 0.05, 0.1, 0.075]
    for d in datas.keys():
        # Button(root, text=files[i], command=lambda c=i: print(btn[c].cget("text"))))

        botton[d] = NewButton(plt.axes(origin), d)  # , command=lambda: self.name = d)
        origin[0] -= 0.11
        botton[d].on_clicked(callback.select_graph)

    d = 'save'
    botton[d] = NewButton(plt.axes(origin), d)
    origin[0] -= 0.11
    botton[d].on_clicked(callback.save)

    d = 'St+'
    botton[d] = NewButton(plt.axes(origin), d)
    origin[0] -= 0.11
    botton[d].on_clicked(callback.increase)

    d = 'St-'
    botton[d] = NewButton(plt.axes(origin), d)
    origin[0] -= 0.11
    botton[d].on_clicked(callback.decrease)

    d = 'Gam+'
    botton[d] = NewButton(plt.axes(origin), d)
    origin[0] -= 0.11
    botton[d].on_clicked(callback.increase_gamma)

    d = 'Gam-'
    botton[d] = NewButton(plt.axes(origin), d)
    origin[0] -= 0.11
    botton[d].on_clicked(callback.decrease_gamma)

    d = 'reset'
    botton[d] = NewButton(plt.axes(origin), d)
    origin[0] -= 0.11
    botton[d].on_clicked(callback.reset)

    return botton


if __name__ == '__main__':
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    x = np.arange(0.0, 1.0, 0.001)
    yy1 = 3 + np.sin(2 * np.pi * x)
    yy2 = 2 + np.sin(1 * np.pi * x)
    datas = {'yy1': [x, yy1], 'yy2': [x, yy2]}

    callback = Handler(datas, save_dir='.')

    bottons = set_buttons(datas, callback)
    cid = fig.canvas.mpl_connect('button_press_event', callback.select_graph)
    callback.runner()
    exit(0)
