import joypy
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
matplotlib.use('TkAgg')

# Mock-ups

def plot_joyplot(beta, fig_path):
    beta['median'] = beta.median(axis=1)
    beta2 = beta.sort_values(by=['median'])
    beta2.reset_index(drop=True, inplace=True)
    beta2.drop(columns=['median'], inplace=True)
    toplot = beta2.sample(100).sort_index()
    joypy.joyplot(toplot.T,
                  overlap=3,
                  colormap=cm.OrRd_r,
                  linecolor='w',
                  linewidth=.5,
                  figsize=[8, 19],
                  ylabels=False,)
    plt.savefig(os.path.join(fig_path, 'joyplot.png'))


def plot_curve(summary_df, fig_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    summary_df = summary_df.sort_values(by='beta_med')
    summary_df = summary_df.reset_index(drop=True)
    summary_df['beta_med'].plot(ax=ax)
    summary_df['beta_std_plus'].plot(ax=ax)
    summary_df['beta_std_minus'].plot(ax=ax)
    summary_df['beta_min'].plot(ax=ax)
    summary_df['beta_max'].plot(ax=ax)
    plt.savefig(os.path.join(fig_path, 'curve.png'))

def main_plotter(beta, summary_df, fig_path):
    #plot_joyplot(beta, fig_path)
    plot_curve(summary_df, fig_path)
    pass

