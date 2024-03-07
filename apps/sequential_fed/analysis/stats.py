import json

import numpy as np
from matplotlib import pyplot as plt
from easydict import EasyDict as edict
from src.apis import utils
from src.apis.db_graph_tools import Graphs
from src.apis.fed_sqlite import FedDB
from src.apis.utils import str_all_in

graphs = Graphs(FedDB('../seqfed.sqlite'))

ax_configs = {
    'seqop_rn': {'color': '#ff7f0e', 'label': 'Seq_RN', 'linestyle': "--", 'linewidth': 2.5},
    'seqop_ga': {'color': '#1f77b4', 'label': 'Seq_GA', 'linestyle': "--", 'linewidth': 2.5},
    'seqop_all': {'color': '#7f7f7f', 'label': 'Seq_All', 'linestyle': "--", 'linewidth': 2.5},
    'ewc_rn': {'color': '#ff7f0e', 'label': 'EWC_RN', 'linestyle': "-", 'linewidth': 2.5},
    'ewc_ga': {'color': '#1f77b4', 'label': 'EWC_GA', 'linestyle': "-", 'linewidth': 2.5},
    'ewc_all': {'color': '#7f7f7f', 'label': 'EWC_All', 'linestyle': "-", 'linewidth': 2.5},
    'warmup': {'color': 'k', 'label': 'Shared', 'linestyle': ":", 'linewidth': 2.5},
    'default': {'color': 'k', 'label': 'Any', 'linestyle': "--", 'linewidth': 2.5},
}


class C:
    @staticmethod
    def mnist(id):
        return 'mnist' in id

    @staticmethod
    def kdd(id):
        return 'kdd' in id

    @staticmethod
    def shard(id):
        return 'shard' in id


def log_transform(dt):
    dt = np.array(dt)
    non_zero_mask = (dt != 0)
    result = np.where(non_zero_mask, np.log10(dt), 0)
    return result


def plt_config(plt):
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.06), ncol=8, fontsize='large')
    plt.rcParams.update({'font.size': 12})
    plt.gca().tick_params(axis='both', which='major', labelsize='large')
    plt.xlim(0)
    for spine in plt.gca().spines.values():
        spine.set_edgecolor('gray')


def acc(sessions, trans=None):
    configurations = []
    for exp_code, session in sessions.items():
        configurations.append({
            'session_id': session,
            'transform': trans,
            'field': 'acc',
            'config': ax_configs[exp_code]
        })
    graphs.plot(configurations, 'Plot', xlabel='Rounds', ylabel='Accuracy', plt_func=plt_config)


def time2acc(sessions, trans=None):
    sessions_val = {}
    for ex_code, table in sessions.items():
        data = graphs.db().query(f"select acc, time from {table}")
        acc_dt = [0] + [d[0] for d in data]
        acc_dt = trans(acc_dt) if trans else acc_dt
        time_dt = [0] + [d[1] / 1000 for d in data]
        time_cumulative = np.cumsum(time_dt)
        sessions_val[table] = {'x': time_cumulative, 'y': acc_dt, 'config': ax_configs[ex_code]}
    graphs.plot2(sessions_val, 'time2acc', xlabel='Cumulative Time', ylabel='Accuracy',
                 plt_func=plt_config)


def generate_exps(session_ids, filter=None):
    table_names = ', '.join(map(lambda x: str(f"'{x}'"), session_ids))
    query = f"SELECT * FROM session WHERE session_id IN ({table_names})"
    sess = graphs.db().query(query)
    res = {}
    for item in sess:
        sess_item = edict(json.loads(item[1].replace("'", '"')))
        print(item[0] + ": ", item[1])
        name = sess_item['method']
        if (filter and filter(name, sess_item)) or not filter:
            res[name] = item[0]
    return res


def get_value(dictionary, key_path, default=None):
    keys = key_path.split('.')
    value = dictionary
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key, default)
        else:
            value = default
            break
    return value


def check(arg_value, value):
    if isinstance(arg_value, list):
        return all([check(a, value) for a in arg_value])
    if callable(arg_value):
        if not arg_value(value):
            return False
    elif value != arg_value:
        return False
    return True


def collect(conditions):
    session_ids = []
    query = f"SELECT * FROM session"
    sess = graphs.db().query(query)
    for sess_id, conf in sess:
        conf = edict(json.loads(conf.replace("'", '"')))
        accepted = True
        for arg_name, arg_value in conditions.items():
            value = get_value(conf, arg_name)
            if not check(arg_value, value):
                accepted = False
        if accepted:
            session_ids.append(sess_id)
    return session_ids


def delete(session_ids):
    for sess_id in session_ids:
        graphs.db().query(f'drop table if exists {sess_id}')
        graphs.db().query(f'delete from session where session_id = "{sess_id}"')
        graphs.db().con.commit()


def standard(ss, filter=None):
    acc(generate_exps(ss, filter), utils.smooth)
    time2acc(generate_exps(ss, filter), utils.smooth)


if __name__ == '__main__':
    # conditions = {'id': C.mnist, 'wmp.rounds': 50, 'wmp.epochs': 20, 'wmp.lr': 0.01}
    conditions = {'id': C.kdd, 'wmp.rounds': 32, 'wmp.epochs': 5, 'wmp.lr': 0.01}

    # highlight: more epochs for each rounds
    # summary: good
    # sessions = collect({'wmp.rounds': 50, 'wmp.epochs': 20, 'wmp.lr': 0.01})

    # highlight: one epoch only
    # summary: bad, ewc is good when working on one epoch
    # sessions = collect({'wmp.rounds': 32, 'wmp.epochs': 1, 'wmp.lr': 0.01})

    # highlight: very low lr, one epoch
    # summary: bad, ewc is good when working with low lr
    # sessions = collect({'wmp.rounds': 32, 'wmp.epochs': 1, 'wmp.lr': 0.0001})

    # highlight: very low lr, more epochs
    # summary: bad, ewc is good when working with low lr even with high epochs
    # sessions = collect({'wmp.rounds': 32, 'wmp.epochs': 20, 'wmp.lr': 0.0001})

    # highlight: lots of rounds, low lr, one epoch
    # summary: number of rounds don't make that much difference
    # sessions = collect({'wmp.rounds': 500, 'wmp.epochs': 1, 'wmp.lr': 0.0001})

    # highlight: more clients/round, high epochs
    # sessions = collect({'wmp.cr': 30, 'wmp.epochs': 20})

    # highlight: more clients/round, one epoch
    # sessions = collect({'wmp.cr': 30, 'wmp.epochs': 1})

    # highlight: starting with kdd dataset, high lr, high epochs
    # summary:
    # sessions = collect({'wmp.rounds': 32, 'wmp.epochs': 20, 'wmp.lr': 0.1})

    # kdd
    # sessions = collect({'wmp.rounds': 32, 'wmp.epochs': 20, 'wmp.lr': 0.01})
    # delete(sessions)
    standard(collect(conditions))

# general notes:
# 1- ewc calculate fisher after every epoch, which means increasing the number of epochs exponentially affect the time
# 2- ewc works well when learn rate is low, need to check why it is bad when working in sequence
# 3- ewc works well on mnist, need to test different dataset and distributions
# 4- the number of rounds have no major effects on the accuracy as much as lr and epochs
