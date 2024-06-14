import matplotlib.pyplot as plt

x1 = [2985, 4975, 3823, 5469, 5348, 4864, 5512, 2767]
y1 = [10567, 14197, 12590, 13769, 13779, 12459, 15278, 10008]

x2 = [1518, 3478, 3541, 5533, 4852, 3309, 1859, 4264]
y2 = [14324, 13429, 17944, 18664, 17266, 14430, 13781, 17686]

x3 = [1949, 1584, 2634, 2503, 2378, 2893, 2160, 1699]
y3 = [11247, 21664, 13036, 12035, 12617, 12992, 15963, 19385]

x4 = [1014, 1175, 1121, 1099, 1300, 1229, 1134, 1445]
y4 = [10554, 10961, 11818, 12121, 10850, 11403, 13024, 12824]

figSize = 7, 5
fig, ax = plt.subplots(figsize=figSize)   # axes

# 设置散点图1的属性
plt.scatter(x1, y1, c='red', label='HSR', alpha=0.5)

# 设置散点图2的属性
plt.scatter(x2, y2, c='blue', label='Composited scheduling rule', alpha=0.5)

# 设置散点图3的属性
plt.scatter(x3, y3, c='teal', label='D3QN with PER', alpha=0.5)

# 设置散点图3的属性
plt.scatter(x4, y4, c='black', label='DDS-Rainbow DQN', alpha=0.5)

def format_number(number):
    if number >= 10000:
        return "{:,.0f}".format(number)
    else:
        return "{:.0f}".format(number)


# 图像设置
plt.rcParams['xtick.direction'] = 'in'  # 坐标轴刻度朝内
plt.rcParams['ytick.direction'] = 'in'  # 坐标轴刻度朝内
ax.tick_params(which='major',axis='x',length =8,width=1)  # top=True
ax.tick_params(which='major',axis='y',length =8,width=1)  # right=True
ax.tick_params(which='minor',axis='x',length =4,width=1)  # top=True
ax.tick_params(which='minor',axis='y',length =4,width=1)  # right=True
ax.spines['bottom'].set_linewidth(1)  # 设置底部坐标轴的粗细
ax.spines['left'].set_linewidth(1)  # 设置左边坐标轴的粗细
ax.spines['right'].set_linewidth(1)  # 设置右边坐标轴的粗细
ax.spines['top'].set_linewidth(1)  # 设置上部坐标轴的粗细
ax.yaxis.grid(True, which='major', linestyle='--', c='grey')
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: format_number(x)))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: format_number(x)))


headerFont = {'family': 'Times New Roman', 'weight': 'normal', 'size': 16}
contentFont = {'family': 'Times New Roman', 'weight': 'normal', 'size': 13}


plt.title('Scatter plot of MkPro 4', headerFont)
plt.xlabel("Total Weighted Tardiness", contentFont)
plt.ylabel("Total Energy Consumption", contentFont)
plt.legend(loc=0, prop=contentFont)

plt.savefig('F:\Yee\毕业论文\初稿\数据\Scatter_MkPro 4.svg', dpi=600, format='svg')

plt.show()
