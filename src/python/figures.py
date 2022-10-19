import pandas as pd
import joypy
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Mock-ups

data = pd.read_csv('./data/intermediate/union_example/betas.csv', header=None)


data.median()

data['mean'] = data.median(axis=1)

data2 = data.sort_values(by=['mean'])

data2.reset_index(drop=True, inplace=True)

data2.drop(columns=['mean'], inplace=True)

toplot = data2.sample(300).sort_index()

joypy.joyplot(toplot.T,
              overlap=1,
              colormap=cm.OrRd_r,
              linecolor='w',
              linewidth=.5,
              figsize=[8, 19],
              ylabels=False,)
plt.show()


def main_figure(results):
    pass

