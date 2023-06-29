# -*- coding: utf-8 -*-
# @Time    : 2023 03
# @Author  : yicao
# %%
import pandas as pd
from fedlab.utils.functional import partition_report
from matplotlib import pyplot as plt

from utils import public_utils

plt.rcParams['font.sans-serif'] = ['Times New Roman']
DataName = 'Cifar10'
NumClasses = 10

train_set, _ = public_utils.get_data_set(DataName)

# %%
distribution = "iid"
ClientNum = 16
Seed = 50
part = public_utils.get_data_part(DataName, distribution, train_set.targets, Seed, ClientNum)

csv_file = f"./painting/csv/data-part-{DataName}-{distribution}.csv"
partition_report(train_set.targets, part.client_dict,
                 class_num=NumClasses,
                 verbose=False, file=csv_file)

hetero_dir_part_df = pd.read_csv(csv_file, header=1)
hetero_dir_part_df = hetero_dir_part_df.set_index('client')
col_names = [f"label {i}" for i in range(NumClasses)]
col_names.append('Amount')
hetero_dir_part_df.columns = col_names
for col in col_names:
    hetero_dir_part_df[col] = (hetero_dir_part_df[col] * hetero_dir_part_df['Amount']).astype(int)

hetero_dir_part_df[col_names[:-1]].iloc[:].plot.barh(stacked=True, fontsize=20)
plt.tight_layout()
plt.xticks([1000, 2000, 3000], size=20)
plt.yticks(size=20)
plt.legend(fontsize=10)
plt.gcf().subplots_adjust(bottom=0.1)
plt.xlabel('sample num', fontsize=25)
plt.ylabel('')
plt.savefig(f"./painting/pic/data-part-{DataName}-{distribution}.svg", bbox_inches='tight')
plt.show()
