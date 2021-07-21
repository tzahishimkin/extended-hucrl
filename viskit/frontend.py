import sys

sys.path.append('.')
import matplotlib
import os
import importlib

matplotlib.use('Agg')
# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42
# matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import viskit.make_plot_eps_ext as ext_plot

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

import flask  # import Flask, render_template, send_from_directory
from misc.ext import flatten
from viskit import core
from misc import ext
import sys
import argparse
import json
import numpy as np


# import threading, webbrowser


def sliding_mean(data_array, window=5):
    data_array = np.array(data_array)
    new_list = []
    for i in range(len(data_array)):
        indices = list(range(max(i - window + 1, 0),
                             min(i + window + 1, len(data_array))))
        avg = 0
        for j in indices:
            avg += data_array[j]
        avg /= float(len(indices))
        new_list.append(avg)

    return np.array(new_list)


import itertools

app = flask.Flask(__name__, static_url_path='/static')

exps_data = None
plottable_keys = None
distinct_params = None


@app.route('/js/<path:path>')
def send_js(path):
    return flask.send_from_directory('js', path)


@app.route('/css/<path:path>')
def send_css(path):
    return flask.send_from_directory('css', path)


def make_plot(plot_list, use_median=False, plot_width=None, plot_height=None, title=None):
    importlib.reload(ext_plot)
    return ext_plot.make_plot_ext(plot_list, use_median, plot_width, plot_height, title)


def make_plot_eps(plot_list, use_median=False, counter=0):
    importlib.reload(ext_plot)
    ext_plot.make_plot_eps_ext(plot_list, use_median, counter)


def summary_name(exp, selector=None):
    return exp.params["exp_name"]


def check_nan(exp):
    return all(not np.any(np.isnan(vals)) for vals in list(exp.progress.values()))


def get_plot_instruction(plot_key, split_key=None, group_key=None, filters=None, use_median=False,
                         only_show_best=False, only_show_best_final=False, gen_eps=False,
                         only_show_best_sofar=False, clip_plot_value=None, plot_width=None,
                         plot_height=None, filter_nan=False, smooth_curve=False, window_size=None, custom_filter=None,
                         legend_post_processor=None, normalize_error=False, custom_series_splitter=None):
    print(plot_key, split_key, group_key, filters)
    if filter_nan:
        nonnan_exps_data = list(filter(check_nan, exps_data))
        selector = core.Selector(nonnan_exps_data)
    else:
        selector = core.Selector(exps_data)
    if legend_post_processor is None:
        legend_post_processor = lambda x: x
    if filters is None:
        filters = dict()
    for k, v in filters.items():
        selector = selector.where(k, str(v))
    if custom_filter is not None:
        selector = selector.custom_filter(custom_filter)
    # print selector._filters

    if split_key is not None:
        vs = [vs for k, vs in distinct_params if k == split_key][0]
        split_selectors = [selector.where(split_key, v) for v in vs]
        split_legends = list(map(str, vs))
    else:
        split_selectors = [selector]
        split_legends = ["Plot"]
    plots = []
    counter = 1
    for split_selector, split_legend in zip(split_selectors, split_legends):
        if custom_series_splitter is not None:
            exps = split_selector.extract()
            splitted_dict = dict()
            for exp in exps:
                key = custom_series_splitter(exp)
                if key not in splitted_dict:
                    splitted_dict[key] = list()
                splitted_dict[key].append(exp)
            splitted = list(splitted_dict.items())
            group_selectors = [core.Selector(list(x[1])) for x in splitted]
            group_legends = [x[0] for x in splitted]
        else:
            if group_key and group_key is not "exp_name":
                vs = [vs for k, vs in distinct_params if k == group_key][0]
                group_selectors = [split_selector.where(group_key, v) for v in vs]
                group_legends = [str(x) for x in vs]
            else:
                group_key = "exp_name"
                vs = sorted([x.params["exp_name"] for x in split_selector.extract()])
                group_selectors = [split_selector.where(group_key, v) for v in vs]
                group_legends = [summary_name(x.extract()[0], split_selector) for x in group_selectors]
        # group_selectors = [split_selector]
        # group_legends = [split_legend]
        to_plot = []
        for group_selector, group_legend in zip(group_selectors, group_legends):
            filtered_data = group_selector.extract()
            if len(filtered_data) > 0:

                if only_show_best or only_show_best_final or only_show_best_sofar:
                    # Group by seed and sort.
                    # -----------------------
                    filtered_params = core.extract_distinct_params(filtered_data, l=0)
                    filtered_params2 = [p[1] for p in filtered_params]
                    filtered_params_k = [p[0] for p in filtered_params]
                    product_space = list(itertools.product(
                        *filtered_params2
                    ))
                    data_best_regret = None
                    best_regret = -np.inf
                    kv_string_best_regret = None
                    for idx, params in enumerate(product_space):
                        selector = core.Selector(exps_data)
                        for k, v in zip(filtered_params_k, params):
                            selector = selector.where(k, str(v))
                        data = selector.extract()
                        if len(data) > 0:
                            progresses = [
                                exp.progress.get(plot_key, np.array([np.nan])) for exp in data
                            ]
                            #                             progresses = [progress[:500] for progress in progresses ]
                            sizes = list(map(len, progresses))
                            max_size = max(sizes)
                            progresses = [
                                np.concatenate([ps, np.ones(max_size - len(ps)) * np.nan]) for ps in progresses]

                            if only_show_best_final:
                                progresses = np.asarray(progresses)[:, -1]
                            if only_show_best_sofar:
                                progresses = np.max(np.asarray(progresses), axis=1)
                            if use_median:
                                medians = np.nanmedian(progresses, axis=0)
                                regret = np.mean(medians)
                            else:
                                means = np.nanmean(progresses, axis=0)
                                regret = np.mean(means)
                            distinct_params_k = [p[0] for p in distinct_params]
                            distinct_params_v = [
                                v for k, v in zip(filtered_params_k, params) if k in distinct_params_k]
                            distinct_params_kv = [
                                (k, v) for k, v in zip(distinct_params_k, distinct_params_v)]
                            distinct_params_kv_string = str(
                                distinct_params_kv).replace('), ', ')\t')
                            print(
                                '{}\t{}\t{}'.format(regret, len(progresses), distinct_params_kv_string))
                            if regret > best_regret:
                                best_regret = regret
                                best_progress = progresses
                                data_best_regret = data
                                kv_string_best_regret = distinct_params_kv_string

                    print(group_selector._filters)
                    print('best regret: {}'.format(best_regret))
                    # -----------------------
                    if best_regret != -np.inf:
                        progresses = [
                            exp.progress.get(plot_key, np.array([np.nan])) for exp in data_best_regret]
                        #                         progresses = [progress[:500] for progress in progresses ]
                        sizes = list(map(len, progresses))
                        # more intelligent:
                        max_size = max(sizes)
                        progresses = [
                            np.concatenate([ps, np.ones(max_size - len(ps)) * np.nan]) for ps in progresses]
                        legend = '{} (mu: {:.3f}, std: {:.5f})'.format(
                            group_legend, best_regret, np.std(best_progress))
                        if window_size is None:
                            window_size = np.maximum(
                                int(np.round(max_size / float(1000))), 1)
                        if use_median:
                            percentile25 = np.nanpercentile(
                                progresses, q=25, axis=0)
                            percentile50 = np.nanpercentile(
                                progresses, q=50, axis=0)
                            percentile75 = np.nanpercentile(
                                progresses, q=75, axis=0)
                            if smooth_curve:
                                percentile25 = sliding_mean(percentile25,
                                                            window=window_size)
                                percentile50 = sliding_mean(percentile50,
                                                            window=window_size)
                                percentile75 = sliding_mean(percentile75,
                                                            window=window_size)
                            if clip_plot_value is not None:
                                percentile25 = np.clip(percentile25, -clip_plot_value, clip_plot_value)
                                percentile50 = np.clip(percentile50, -clip_plot_value, clip_plot_value)
                                percentile75 = np.clip(percentile75, -clip_plot_value, clip_plot_value)
                            to_plot.append(
                                ext.AttrDict(percentile25=percentile25, percentile50=percentile50,
                                             percentile75=percentile75, legend=legend_post_processor(legend)))
                        else:
                            means = np.nanmean(progresses, axis=0)
                            stds = np.nanstd(progresses, axis=0)
                            if normalize_error:  # and len(progresses) > 0:
                                stds /= np.sqrt(np.sum((1. - np.isnan(progresses)), axis=0))
                            if smooth_curve:
                                means = sliding_mean(means,
                                                     window=window_size)
                                stds = sliding_mean(stds,
                                                    window=window_size)
                            if clip_plot_value is not None:
                                means = np.clip(means, -clip_plot_value, clip_plot_value)
                                stds = np.clip(stds, -clip_plot_value, clip_plot_value)
                            to_plot.append(
                                ext.AttrDict(means=means, stds=stds, legend=legend_post_processor(legend)))
                        if len(to_plot) > 0 and len(data) > 0:
                            to_plot[-1]["footnote"] = "%s; e.g. %s" % (
                                kv_string_best_regret, data[0].params.get("exp_name", "NA"))
                        else:
                            to_plot[-1]["footnote"] = ""
                else:
                    progresses = [
                        exp.progress.get(plot_key, np.array([np.nan])) for exp in filtered_data]
                    sizes = list(map(len, progresses))
                    # more intelligent:
                    max_size = max(sizes)
                    progresses = [
                        np.concatenate([ps, np.ones(max_size - len(ps)) * np.nan]) for ps in progresses]
                    if window_size is None:
                        window_size = np.maximum(int(np.round(max_size / float(1000))), 1)

                    if use_median:
                        percentile25 = np.nanpercentile(
                            progresses, q=25, axis=0)
                        percentile50 = np.nanpercentile(
                            progresses, q=50, axis=0)
                        percentile75 = np.nanpercentile(
                            progresses, q=75, axis=0)
                        if smooth_curve:
                            percentile25 = sliding_mean(percentile25,
                                                        window=window_size)
                            percentile50 = sliding_mean(percentile50,
                                                        window=window_size)
                            percentile75 = sliding_mean(percentile75,
                                                        window=window_size)
                        if clip_plot_value is not None:
                            percentile25 = np.clip(percentile25, -clip_plot_value, clip_plot_value)
                            percentile50 = np.clip(percentile50, -clip_plot_value, clip_plot_value)
                            percentile75 = np.clip(percentile75, -clip_plot_value, clip_plot_value)
                        to_plot.append(
                            ext.AttrDict(percentile25=percentile25, percentile50=percentile50,
                                         percentile75=percentile75, legend=legend_post_processor(group_legend)))
                    else:
                        means = np.nanmean(progresses, axis=0)
                        stds = np.nanstd(progresses, axis=0)
                        if smooth_curve:
                            means = sliding_mean(means,
                                                 window=window_size)
                            stds = sliding_mean(stds,
                                                window=window_size)
                        if clip_plot_value is not None:
                            means = np.clip(means, -clip_plot_value, clip_plot_value)
                            stds = np.clip(stds, -clip_plot_value, clip_plot_value)
                        to_plot.append(
                            ext.AttrDict(means=means, stds=stds, legend=legend_post_processor(group_legend)))

        if len(to_plot) > 0 and not gen_eps:
            fig_title = "%s: %s" % (split_key, split_legend)
            plots.append("<h3>%s</h3>" % fig_title)
            plots.append(make_plot(
                to_plot,
                use_median=use_median, title=fig_title,
                plot_width=plot_width, plot_height=plot_height
            ))

        if gen_eps:
            make_plot_eps(to_plot, use_median=use_median, counter=counter)
        counter += 1
    return "\n".join(plots)


def parse_float_arg(args, key):
    x = args.get(key, "")
    try:
        return float(x)
    except Exception:
        return None


@app.route("/plot_div")
def plot_div():
    #     reload_data()
    args = flask.request.args
    plot_key = args.get("plot_key")
    split_key = args.get("split_key", "")
    group_key = args.get("group_key", "")
    filters_json = args.get("filters", "{}")
    filters = json.loads(filters_json)
    if len(split_key) == 0:
        split_key = None
    if len(group_key) == 0:
        group_key = None
    # group_key = distinct_params[0][0]
    # print split_key
    # exp_filter = distinct_params[0]
    use_median = args.get("use_median", "") == 'True'
    gen_eps = args.get("eps", "") == 'True'
    only_show_best = args.get("only_show_best", "") == 'True'
    only_show_best_final = args.get("only_show_best_final", "") == 'True'
    only_show_best_sofar = args.get("only_show_best_sofar", "") == 'True'
    normalize_error = args.get("normalize_error", "") == 'True'
    filter_nan = args.get("filter_nan", "") == 'True'
    smooth_curve = args.get("smooth_curve", "") == 'True'
    wind_size = parse_float_arg(args, "wsize")
    wind_size = int(wind_size) if wind_size is not None else None
    clip_plot_value = parse_float_arg(args, "clip_plot_value")
    plot_width = parse_float_arg(args, "plot_width")
    plot_height = parse_float_arg(args, "plot_height")
    custom_filter = args.get("custom_filter", None)
    custom_series_splitter = args.get("custom_series_splitter", None)
    if custom_filter is not None and len(custom_filter.strip()) > 0:
        custom_filter = safer_eval(custom_filter)

    else:
        custom_filter = None
    legend_post_processor = args.get("legend_post_processor", None)
    if legend_post_processor is not None and len(legend_post_processor.strip()) > 0:
        legend_post_processor = safer_eval(legend_post_processor)
    else:
        legend_post_processor = None
    if custom_series_splitter is not None and len(custom_series_splitter.strip()) > 0:
        custom_series_splitter = safer_eval(custom_series_splitter)
    else:
        custom_series_splitter = None
    plot_div = get_plot_instruction(plot_key=plot_key, split_key=split_key, filter_nan=filter_nan,
                                    group_key=group_key, filters=filters, use_median=use_median, gen_eps=gen_eps,
                                    only_show_best=only_show_best, only_show_best_final=only_show_best_final,
                                    only_show_best_sofar=only_show_best_sofar,
                                    clip_plot_value=clip_plot_value, plot_width=plot_width, plot_height=plot_height,
                                    smooth_curve=smooth_curve, window_size=wind_size, custom_filter=custom_filter,
                                    legend_post_processor=legend_post_processor, normalize_error=normalize_error,
                                    custom_series_splitter=custom_series_splitter)
    # print plot_div
    return plot_div


def safer_eval(some_string):
    """
    Not full-proof, but taking advice from:

    https://nedbatchelder.com/blog/201206/eval_really_is_dangerous.html
    """
    if "__" in some_string or "import" in some_string:
        raise Exception("string to eval looks suspicious")
    return eval(some_string, {'__builtins__': {}})


@app.route("/")
def index():
    # exp_folder_path = "data/s3/experiments/ppo-atari-3"
    # _load_data(exp_folder_path)
    # exp_json = json.dumps(exp_data)
    if "AverageReturn" in plottable_keys:
        plot_key = "AverageReturn"
    elif "eval_reward" in plottable_keys:
        plot_key = "eval_reward"
    elif len(plottable_keys) > 0:
        plot_key = plottable_keys[0]
    else:
        plot_key = None

    def does_key_exist(distinct_params, key):
        return np.any([key == param[0] for param in distinct_params])

    group_key = None
    split_key = None
    if len(distinct_params) > 0:
        if does_key_exist(distinct_params, key='policy'):
            group_key = 'policy'
        else:
            group_key = distinct_params[0][0]
        if does_key_exist(distinct_params, key='env'):
            split_key = 'env'

    plot_div = get_plot_instruction(
        plot_key=plot_key, split_key=split_key, group_key=group_key)
    return flask.render_template(
        "main.html",
        plot_div=plot_div,
        plot_key=plot_key,
        group_key=group_key,
        plottable_keys=plottable_keys,
        distinct_param_keys=[str(k) for k, v in distinct_params],
        distinct_params=dict([(str(k), list(map(str, v)))
                              for k, v in distinct_params]),
    )


@app.route("/reload_data_on")
def reload_data_on():
    global exps_data
    global plottable_keys
    global distinct_params

    global data_paths
    global disable_variant
    global g_prefix

    # refresh folder list following a prefix
    if g_prefix != "???":
        data_paths = []
        dirname = os.path.dirname(g_prefix)
        subdirprefix = os.path.basename(g_prefix)
        for subdirname in os.listdir(dirname):
            path = os.path.join(dirname, subdirname)
            if os.path.isdir(path) and (subdirprefix in subdirname):
                data_paths.append(path)

    exps_data = core.load_exps_data(data_paths, disable_variant)
    plottable_keys = list(
        set(flatten(list(exp.progress.keys()) for exp in exps_data)))
    plottable_keys = sorted([k for k in plottable_keys if k is not None])
    distinct_params = sorted(core.extract_distinct_params(exps_data))

    print("finished reImporting data from {path}...".format(path=data_paths))
    return "Reloading..."


def reload_data():
    global exps_data
    global plottable_keys
    global distinct_params

    global data_paths
    global disable_variant

    data_paths = args.data_paths
    disable_variant = args.disable_variant

    exps_data = core.load_exps_data(args.data_paths, args.disable_variant)
    plottable_keys = list(
        set(flatten(list(exp.progress.keys()) for exp in exps_data)))
    plottable_keys = sorted([k for k in plottable_keys if k is not None])
    distinct_params = sorted(core.extract_distinct_params(exps_data))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_paths", type=str, nargs='*')
    parser.add_argument("--prefix", type=str, nargs='?', default="???")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--disable-variant", default=False, action='store_true')
    args = parser.parse_args(sys.argv[1:])

    global g_prefix;
    g_prefix = args.prefix

    # load all folders following a prefix
    if args.prefix != "???":
        args.data_paths = []
        dirname = os.path.dirname(args.prefix)
        subdirprefix = os.path.basename(args.prefix)
        for subdirname in os.listdir(dirname):
            path = os.path.join(dirname, subdirname)
            if os.path.isdir(path) and (subdirprefix in subdirname):
                args.data_paths.append(path)
    print("Importing data from {path}...".format(path=args.data_paths))
    reload_data()
    # port = 5000
    # url = "http://0.0.0.0:{0}".format(port)
    print("Done! View http://localhost:%d in your browser" % args.port)
    app.run(host='0.0.0.0', port=args.port, debug=args.debug)
