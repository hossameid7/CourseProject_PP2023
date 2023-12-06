import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

data = []
with open('result.txt', 'r') as f:
    lines = f.readlines()
    cnt = 0
    limit = 28
    d = []
    for line in lines:
        line = line.strip().split(' ')
        line = [float(x) for x in line]
        d.append(line)
        cnt += 1
        if cnt == limit:
            cnt = 0
            data.append(d)
            d = []

x = 5
y = 6
f, ax = plt.subplots(x, y)

for i in range(x):
    for j in range(y):
        # set upper and lower bounds to 300 and 1200 and remove the x and y labels / ticks
        sns.heatmap(data[i * y + j],
                    cbar=False,
                    linewidths=0,
                    ax=ax[i, j],
                    vmin=300,
                    vmax=1200,
                    xticklabels=False,
                    yticklabels=False)
# norm = plt.Normalize(300, 1200)
# sm = plt.cm.ScalarMappable(norm=norm)
# sm.set_array([])
# # change color palette from viridis to inferno
# cmap = mpl.cm.inferno
# plt.colorbar(sm, cax=f.add_axes([0.95, 0.15, 0.03, 0.7]), cmap=cmap)
plt.tight_layout()
plt.show()
