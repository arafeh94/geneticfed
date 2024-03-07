import json

import numpy as np
from matplotlib import pyplot as plt

from src.apis import utils
from src.apis.db_graph_tools import Graphs
from src.apis.fed_sqlite import FedDB

graphs = Graphs(FedDB('splitlearn_v2.sqlite'))

ax_configs = {
    'split': {'color': 'r', 'label': 'SL', 'linestyle': "-.", 'linewidth': 3.2},
    'splitfed': {'color': 'b', 'label': 'SFL', 'linestyle': ":", 'linewidth': 3.2},
    '1layer': {'color': 'g', 'label': '1L', 'linestyle': ":", 'linewidth': 3.2},
    '2layers_selection': {'color': 'y', 'label': '2Lo', 'linestyle': "-", 'linewidth': 3.2},
    '2layers_selection_v1': {'color': 'm', 'label': '2Lo1', 'linestyle': "-", 'linewidth': 3.2},
    '2layers_standard': {'color': 'c', 'label': '2Ls', 'linestyle': "--", 'linewidth': 3.2},
    '2layers_standard_v1': {'color': 'k', 'label': '2Ls1', 'linestyle': "--", 'linewidth': 3.2},
}


def log_transform(dt):
    dt = np.array(dt)
    non_zero_mask = (dt != 0)
    result = np.where(non_zero_mask, np.log10(dt), 0)
    return result


def plt_config(plt):
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=8, fontsize='large')
    plt.rcParams.update({'font.size': 12})
    plt.gca().tick_params(axis='both', which='major', labelsize='large')
    for spine in plt.gca().spines.values():
        spine.set_edgecolor('gray')


def acc(sessions, trans=None):
    configurations = []
    for exp_code, session in sessions.items():
        configurations.append({
            'session_id': session,
            'transform': trans,
            'field': 'round_time',
            'query': f'select max(acc) as acc from {session} group by round_num',
            'config': ax_configs[exp_code]
        })
    graphs.plot(configurations, 'Plot', xlabel='Rounds', ylabel='Accuracy', plt_func=plt_config)


def time(sessions, trans=None):
    configurations = []
    for exp_code, session in sessions.items():
        configurations.append({
            'session_id': session,
            'transform': trans,
            'field': 'round_time',
            'query': f'select max(round_time) as round_time from {session} group by round_num',
            'config': ax_configs[exp_code]
        })
    graphs.plot(configurations, 'Plot', xlabel='Rounds', ylabel='Exec Time', plt_func=plt_config)


def time2acc(sessions, trans=None):
    sessions_val = {}
    for ex_code, table in sessions.items():
        data = graphs.db().query(f"select max(acc) as acc, round_time from {table} group by round_num")
        acc_dt = [0] + [d[0] for d in data]
        acc_dt = trans(acc_dt) if trans else acc_dt
        time_dt = [0] + [d[1] / 1000 for d in data]
        time_cumulative = np.cumsum(time_dt)
        sessions_val[table] = {'x': time_cumulative, 'y': acc_dt, 'config': ax_configs[ex_code]}
    graphs.plot2(sessions_val, 'time2acc', xlabel='Cumulative Time', ylabel='Accuracy',
                 plt_func=plt_config)


def generate_exps(session_ids, only=None):
    table_names = ', '.join(map(lambda x: str(f"'{x}'"), session_ids))
    query = f"SELECT * FROM session WHERE session_id IN ({table_names})"
    sess = graphs.db().query(query)
    res = {}
    for item in sess:
        name = json.loads(item[1].replace("'", '"'))['name']
        if only is None or name in only:
            res[name] = item[0]
    return res


def v0v1(ss):
    selection = ['2layers_selection', '2layers_selection_v1', '2layers_standard', '2layers_standard_v1']
    acc(generate_exps(ss, selection), utils.smooth)
    time2acc(generate_exps(ss, selection), utils.smooth)
    time(generate_exps(ss, selection), utils.smooth)


def standard(ss):
    acc(generate_exps(ss, ['split', 'splitfed', '1layer', '2layers_selection', '2layers_standard']), utils.smooth)
    time2acc(generate_exps(ss, ['split', 'splitfed', '1layer', '2layers_selection', '2layers_standard']), utils.smooth)
    time2acc(generate_exps(ss, ['splitfed', '1layer', '2layers_selection', '2layers_standard']), utils.smooth)
    time(generate_exps(ss, ['split', 'splitfed', '1layer', '2layers_selection', '2layers_standard']), utils.smooth)
    time(generate_exps(ss, ['splitfed', '1layer', '2layers_selection', '2layers_standard']), utils.smooth)


if __name__ == '__main__':
    code_exp = ['split', 'splitfed', '1layer', '2layers_selection', '2layers_selection_v1',
                '2layers_standard', '2layers_standard_v1']
    ss = ['t1707931858', 't1707931886', 't1707931907', 't1707931929', 't1707932011', 't1707932093', 't1707932170']
    ss2 = ['t1707890747', 't1707890979', 't1707891155', 't1707891331', 't1707892081', 't1707892819', 't1707893534']
    # v0v1(ss)
    standard(ss)
