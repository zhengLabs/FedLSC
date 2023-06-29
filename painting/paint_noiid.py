# -*- coding: utf-8 -*-
# @Time    : 2023 01
# @Author  : yicao
# %%
import pandas as pd
from matplotlib import pyplot as plt

csv_file = "./painting/csv/cifar10_no_iid_part3.csv"
NumClasses = 10

hetero_dir_part_df = pd.read_csv(csv_file, header=1)
hetero_dir_part_df = hetero_dir_part_df.set_index('client')
col_names = [f"class{i}" for i in range(NumClasses)]
for col in col_names:
    hetero_dir_part_df[col] = (hetero_dir_part_df[col] * hetero_dir_part_df['Amount']).astype(int)

hetero_dir_part_df[col_names].iloc[:].plot.barh(stacked=True)
plt.tight_layout()
plt.xlabel('sample num')
plt.savefig(f"./painting/pic/cifar10_no_iid_part2.png", dpi=400)
plt.show()
