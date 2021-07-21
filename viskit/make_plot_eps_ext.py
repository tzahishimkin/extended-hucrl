import sys

sys.path.append('.')
import matplotlib

matplotlib.use('Agg')
# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42
# matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

from viskit import core
import numpy as np
# import threading, webbrowser
import plotly.offline as po
import plotly.graph_objs as go


def make_plot_ext(plot_list, use_median=False, plot_width=None, plot_height=None, title=None):
    data = []
    p25, p50, p75 = [], [], []

    Y = []
    for idx, plt in enumerate(plot_list):
        color = core.color_defaults[idx % len(core.color_defaults)]
        if use_median:
            x = list(range(len(plt.percentile50)))
            y = list(plt.percentile50)

        else:
            x = list(range(len(plt.means)))
            y = list(plt.means)

        y_sum = np.sum(y)
        Y.append(y_sum)

        if plt.legend == "TD3":
            td3ind = idx
        else:
            td3ind = 100001
    if len(Y):
        best_n = np.min([3, len(Y)])
        threshold = np.sort(Y)[-best_n]  #:][::-1]
        # threshold = Y[td3ind]*1.15

    for idx, plt in enumerate(plot_list):
        color = core.color_defaults[idx % len(core.color_defaults)]
        if use_median:
            p25.append(np.mean(plt.percentile25))
            p50.append(np.mean(plt.percentile50))
            p75.append(np.mean(plt.percentile75))
            x = list(range(len(plt.percentile50)))
            y = list(plt.percentile50)
            y_upper = list(plt.percentile75)
            y_lower = list(plt.percentile25)
        else:
            x = list(range(len(plt.means)))
            y = list(plt.means)
            y_upper = list(plt.means + plt.stds)
            y_lower = list(plt.means - plt.stds)

        y_sum = np.sum(y)
        visible = True
        if y_sum < threshold and not idx == td3ind:
            visible = 'legendonly'

        data.append(go.Scatter(
            x=x + x[::-1],
            y=y_upper + y_lower[::-1],
            fill='tozerox',
            fillcolor=core.hex_to_rgb(color, 0.25),
            # line=go.Line(color='transparent'),
            line=go.Line(color='white', width=0.),
            showlegend=False,
            legendgroup=plt.legend,
            visible=visible,
            hoverinfo='none'
        ))
        data.append(go.Scatter(
            x=x,
            y=y,
            name=plt.legend,
            legendgroup=plt.legend,
            visible=visible,
            line=dict(color=core.hex_to_rgb(color)),
        ))
    p25str = '['
    p50str = '['
    p75str = '['
    for p25e, p50e, p75e in zip(p25, p50, p75):
        p25str += (str(p25e) + ',')
        p50str += (str(p50e) + ',')
        p75str += (str(p75e) + ',')
    p25str += ']'
    p50str += ']'
    p75str += ']'
    print(p25str)
    print(p50str)
    print(p75str)

    layout = go.Layout(
        legend=dict(
            x=1,
            y=1,
            # xanchor="left",
            # yanchor="bottom",
        ),
        width=plot_width,
        height=plot_height,
        # title=title,
    )
    fig = go.Figure(data=data, layout=layout)
    fig_div = po.plot(fig, output_type='div', include_plotlyjs=False)
    if "footnote" in plot_list[0]:
        footnote = "<br />".join([
            r"<span><b>%s</b></span>: <span>%s</span>" % (plt.legend, plt.footnote)
            for plt in plot_list
        ])
        return r"%s<div>%s</div>" % (fig_div, footnote)
    else:
        return fig_div


def make_plot_eps_ext(plot_list, use_median=False, counter=0, best_n=3):
    import matplotlib.pyplot as _plt
    f, ax = _plt.subplots(figsize=(8, 5))

    Y = []
    for idx, plt in enumerate(plot_list):
        color = core.color_defaults[idx % len(core.color_defaults)]
        if use_median:
            x = list(range(len(plt.percentile50)))
            y = list(plt.percentile50)
            # y_upper = list(plt.percentile75)
            # y_lower = list(plt.percentile25)
        else:
            x = list(range(len(plt.means)))
            y = list(plt.means)
            # y_upper = list(plt.means + plt.stds)
            # y_lower = list(plt.means - plt.stds)

        y_sum = np.sum(y)
        Y.append(y_sum)

        if plt.legend == "TD3":
            td3ind = idx

    if len(Y):
        # threshold = np.sort(Y)[-best_n]#:][::-1]
        threshold = Y[td3ind] * 1.05

    for idx, plt in enumerate(plot_list):
        color = core.color_defaults[idx % len(core.color_defaults)]
        if use_median:
            x = list(range(len(plt.percentile50)))
            y = list(plt.percentile50)
            y_upper = list(plt.percentile75)
            y_lower = list(plt.percentile25)
        else:
            x = list(range(len(plt.means)))
            y = list(plt.means)
            y_upper = list(plt.means + plt.stds)
            y_lower = list(plt.means - plt.stds)

        y_sum = np.sum(y)
        if y_sum < threshold and not idx == td3ind:
            continue

        plt.legend = plt.legend.replace('td3sinp', 'INCA-STATE,')
        plt.legend = plt.legend.replace('td3tdinp', 'INCA-TD,')
        plt.legend = plt.legend.replace('TD3dag', '[Pathak\'19],')

        # ax.set_xlabel('Iteration', fontsize = 32)
        # ax.set_ylabel('Cum. Reward', fontsize = 32)

        # %n CR
        # ax.set_xlabel('Iteration', fontsize = 38)
        # ax.set_ylabel('Cum. Reward', fontsize = 38)

        ax.set_xlabel('Iteration', fontsize=34)
        ax.set_ylabel('Cum. Reward', fontsize=34)

        # in presentation
        # ax.set_xlabel('Iteration', fontsize = 32)
        # ax.set_ylabel('Cum. Reward', fontsize = 32)

        # plt.xticks(fontsize=14)
        # plt.yticks(fontsize=14)

        # in CR
        ax.tick_params(axis='both', which='major', labelsize=22)
        ax.tick_params(axis='both', which='minor', labelsize=20)

        # in presentation
        # ax.tick_params(axis='both', which='major', labelsize=20)
        # ax.tick_params(axis='both', which='minor', labelsize=18)

        ax.fill_between(
            x, y_lower, y_upper, interpolate=True, facecolor=color, linewidth=0.0, alpha=0.3)
        if idx == 2:
            ax.plot(x, y, color=color, label=plt.legend, linewidth=1.5)
        else:
            ax.plot(x, y, color=color, label=plt.legend, linewidth=1.5)
        ax.grid(True)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        loc = 'lower right'

        # # SwimmerGather
        # ax.set_xlim([0,1500])
        # ax.set_ylim([0,0.35])

        # # LQR1
        # ax.set_xlim([100,500])
        # ax.set_ylim([-3000,-500])

        # LQR2
        # ax.set_xlim([0,500])

        # #   # CPSwingupX
        # ax.set_xlim([0,500])
        # ax.set_ylim([0,400])
        # 
        # # Walker2d
        # ax.set_xlim([0,15000])
        # ax.set_ylim([0,2400])

        # ax.grid(False)
        leg = ax.legend(loc=loc, prop={'size': 14}, ncol=1)
        for legobj in leg.legendHandles:
            legobj.set_linewidth(5.0)

        def y_fmt(x, y):
            return str(int(np.round(x / 1000.0))) + 'K'

        #         ax.xaxis.set_major_formatter(tick.FuncFormatter(y_fmt))
        _plt.savefig('tmp' + str(counter) + '.pdf', bbox_inches='tight')
