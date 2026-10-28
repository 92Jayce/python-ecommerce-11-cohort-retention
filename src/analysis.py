"""第11讲 Cohort同期群留存
业务问题：新客后续还回来买了吗？
全部使用模拟教学数据。"""
from pathlib import Path
import sys
if "__file__" in globals():
    COURSE_ROOT = Path(__file__).resolve().parents[1]
else:
    COURSE_ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "course_common.py").exists())
if str(COURSE_ROOT) not in sys.path:
    sys.path.insert(0, str(COURSE_ROOT))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from course_common import (ROOT, DATA, AS_OF, BLUE, TEAL, ORANGE, load_tables,
    build_facts, positive_orders, rfm_table, safe_divide, outdir, save_table, save_figure, show)

# %% 以观察期首次正净额订单所在月作为cohort
o=positive_orders();o['activity_month']=o.order_date.dt.to_period('M')
first=o.groupby('customer_id').activity_month.min().rename('cohort')
o=o.merge(first,on='customer_id',validate='many_to_one')
o['age']=(o.activity_month.dt.year-o.cohort.dt.year)*12+o.activity_month.dt.month-o.cohort.dt.month
count=o.groupby(['cohort','age']).customer_id.nunique().unstack(fill_value=0).reindex(columns=range(12),fill_value=0)
retention=count.div(count[0],axis=0)
last_month=(AS_OF-pd.Timedelta(days=1)).to_period('M')
for cohort in retention.index:
    observed=(last_month.year-cohort.year)*12+last_month.month-cohort.month
    retention.loc[cohort,retention.columns>observed]=np.nan
assert retention[0].eq(1).all()
print(retention.round(3).to_string());save_table(retention.reset_index(),11,'cohort_retention')
fig,ax=plt.subplots(figsize=(9,5));im=ax.imshow(retention.to_numpy()*100,cmap='Blues',vmin=0,vmax=100,aspect='auto')
ax.set_yticks(range(len(retention)),retention.index.astype(str));ax.set_xticks(range(12),range(12))
ax.set(title='Purchase retention by first-purchase cohort',xlabel='Months since cohort entry',ylabel='First positive-net purchase month')
fig.colorbar(im,ax=ax,label='Customers retained (%)');save_figure(fig,11,'cohort_heatmap')
print('空白代表尚未完成观察，不是0留存；同月多笔订单对留存只算一个客户。')
