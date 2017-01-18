from __future__ import division
import pandas as pd
import os
import numpy as np

__author__ = 'Vladimir Iglovikov'


train = pd.read_csv('../data/trainingData.csv', sep=';')
test = pd.read_csv('../data/testData.csv', sep=';')

values_mapper_dict = {'Buy': 1, 'Hold': 0, 'Sell': -1}


def helper(a):
    keys, values = zip(*map(lambda x: x.split(',')[:2], a[1:-1].split('}{')))
    keys = np.array(keys).astype(int)

    values = [values_mapper_dict[x] for x  in values]
    return dict(zip(keys, values))

joined = pd.concat([train, test])

joined['dict'] = joined['Recommendations'].apply(helper)

tx = pd.DataFrame(joined['dict'].values)

joined['Decision'] = joined['Decision'].map(values_mapper_dict)

joined = joined.reset_index(drop=True)

tx = pd.DataFrame(list(joined['dict'].values))

tx_mean = np.mean(tx, axis=1)
tx_std = np.std(tx, axis=1)

tx_0 = [tx == 0]

joined['mean_pred'] = tx_mean
joined['std_pred'] = tx_std

joined = joined.drop('dict', 1)
joined = pd.concat([joined, tx], axis=1)
joined.to_csv('../data/joined.csv', index=False)