import numpy as np
import pandas as pd
import tqdm
from pandas import Series


def get_programs(*pths):
    prgs = {}

    for name, pth, st, end in zip(['train', 'valid', 'test'], pths,
                                  # start - 1 d
                                  ['2020-03-08', '2020-05-17', '2020-07-25'], ['2020-05-17', '2020-07-26', '2020-10-04']

                                  ):
        prgs[name] = pd.read_csv(pth)
        prgs[name] = prgs[name][prgs[name]['tv_show_id'] != 0]
        prgs[name]['start_time'] = pd.to_datetime(prgs[name]['start_time'])
        prgs[name]['end_time'] = prgs[name]['start_time'] + prgs[name]['duration'].astype(np.dtype('timedelta64[s]'))

        prgs[name] = prgs[name][
            (prgs[name]['start_time'] >= np.datetime64(st)) & (prgs[name]['start_time'] <= np.datetime64(end))]

        prgs[name]['w_start_time'], prgs[name]['w_end_time'] = create_wdts(prgs[name]['start_time'],
                                                                           prgs[name]['end_time'])

    return prgs


def create_weekdt(x):
    x_wd = '2020-01-0' + (x.dt.weekday + 1).astype(str)
    x = x_wd + ' ' + x.dt.time.astype(str)

    return x.astype(np.datetime64)


def create_wdts(start, end):
    start = create_weekdt(start)
    end = create_weekdt(end)

    end = np.where(start > end, end + np.timedelta64(7, 'D'), end)

    return start, end


def get_target(df, watch_rate_co=.8, top_k=5, add_empty=True, add_ids=()):
    ids = np.unique(df['user_id'])

    df = df[df['watch_rate'] >= watch_rate_co]
    grp = df.groupby(['user_id', 'tv_show_id']).size()
    grp.name = 'freq'
    grp = grp.reset_index()
    grp = grp.sort_values('freq', ascending=False)
    grp = grp.groupby('user_id')['tv_show_id'].agg(lambda x: x[:top_k].tolist())

    if add_empty:
        empty_index = [x for x in ids if x not in grp.index] + list(add_ids)
        add = Series([[]] * len(empty_index), index=empty_index)
        grp = pd.concat([grp, add])

    return grp


def get_pred_with_baseline(pred, *baselines):
    res = []

    baselines = [list(x) for x in baselines]

    for n, pr in enumerate(pred):

        while len(pr) < 5:
            for bs in baselines:
                if len(bs) == len(pred):
                    bs = bs[n]
                bs = bs[:(5 - len(pr))]

                pr = pr + bs
        res.append(pr)

    return res


def merge_and_calc_time(tr_short, prg_short):
    merged = pd.merge(tr_short, prg_short, on='channel_id', how='inner')
    merged = merged[(merged['s_start_time'] < merged['w_end_time']) &
                    (merged['w_start_time'] < merged['s_end_time'])]
    merged['time'] = merged[['w_end_time', 's_end_time']].min(axis=1) \
                     - merged[['s_start_time', 'w_start_time']].max(axis=1)
    merged['time'] = merged['time'] / np.timedelta64(1, 's')
    # merged = merged[['user_id', 'time', 'tv_show_id']]

    return merged


def merge_batch(tr_short, prg_short, batch_size=100):
    res = []

    un = np.unique(tr_short['user_id'])
    ids = np.array_split(un, un.shape[0] // batch_size)

    for _id in tqdm.tqdm_notebook(ids):
        tr = tr_short[tr_short['user_id'].isin(set(_id))]
        # print('start merge')
        merged = merge_and_calc_time(tr, prg_short)
        res.append(merged)

    res = pd.concat(res)

    return res


def _blend_predictions(*preds):
    n = 0
    # initial - intersection
    inter = set.intersection(*(set(x) for x in preds))
    res = [x for x in preds[0] if x in inter]

    for i in range(5):
        for arr in preds:
            try:
                val = arr[i]
            except IndexError:
                continue
            if val not in res and len(res) < 5:
                res.append(val)

    return res


def blend_predictions(*preds):
    return [_blend_predictions(*x) for x in zip(*preds)]


def multiblend(*preds):
    preds = preds[::-1]
    blend = preds[0]

    for pr in preds[1:]:
        blend = blend_predictions(pr, blend)

    return blend


def process_merged(merged):
    merged['watch_duration'] = merged[['w_end_time', 's_end_time']].min(axis=1) \
                               - merged[['s_start_time', 'w_start_time']].max(axis=1)

    merged['watch_duration'] = merged['watch_duration'] / np.timedelta64(1, 's')
    merged['show_duration'] = merged['w_end_time'] - merged['w_start_time']
    merged['show_duration'] = merged['show_duration'] / np.timedelta64(1, 's')

    merged['rate'] = merged['watch_duration'] / merged['show_duration']
    merged = merged.sort_values('time', ascending=False)

    return merged


def get_merged_pred(merged, ids):

    merged = merged.sort_values('time', ascending=False)

    grp = merged.groupby('user_id')['tv_show_id'].agg(lambda x: x.value_counts()[:5].index.tolist())
    new_pred = grp.loc[[x for x in ids if x in grp.index]]
    new_ids = list(set(ids) - set(new_pred.index))
    add = Series([[]] * len(new_ids), index=new_ids)
    new_pred = pd.concat([new_pred, add], axis=0).loc[ids]

    return new_pred
