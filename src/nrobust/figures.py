import joypy
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
from nrobust.utils import get_selection_key
from nrobust.utils import get_colors
matplotlib.use('TkAgg')


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


def plot_curve(results_object,
               specs=None,
               ax=None,
               colormap=None,
               colorset=None):
    if ax is None:
        ax = plt.gca()
    if colormap is None:
        colormap = 'Set1'
    df = results_object.summary_df.copy()
    if specs:
        key = get_selection_key(specs)
        df['idx'] = df.spec_name.isin(key)
    df = df.sort_values(by='median')
    df = df.reset_index(drop=True)
    df['median'].plot(ax=ax, color='blue')
    df['std_one_up'].plot(ax=ax, color='red', alpha=.9)
    df['std_one_down'].plot(ax=ax, color='red', alpha=.9)
    df['min'].plot(ax=ax, color='grey', alpha=.5)
    df['max'].plot(ax=ax, color='grey', alpha=.5)
    ax.axhline(y=0, color='black')
    if specs:
        idxs = df.index[df['idx']].tolist()
        lines = []
        if colorset is None:
            colors = get_colors(specs=specs, color_set_name=colormap)
        for idx, i in zip(idxs, range(len(specs))):
            control_names = list(df.spec_name.iloc[idx])
            label = 'Controls: ' + ', '.join(control_names)
            lines.append(ax.axvline(x=idx,
                                    color=colors[i],
                                    label=label))
            ax.legend(handles=lines)
    return ax


def plot_ic(results_object,
            ic,
            specs=None,
            ax=None,
            colormap='Set1',
            colorset=None):
    if ax is None:
        ax = plt.gca()
    df = results_object.summary_df.copy()
    if specs:
        key = get_selection_key(specs)
        df['idx'] = df.spec_name.isin(key)
        df = df.sort_values(by=ic).reset_index(drop=True)
        ic, = ax.plot(df[ic])
        idxs = df.index[df['idx']].tolist()
        if colorset is None:
            colors = get_colors(specs=specs, color_set_name=colormap)
        for idx, i in zip(idxs, range(len(specs))):
            ax.axvline(x=idx,
                       color=colors[i])
        return ic
    df = df.sort_values(by=ic).reset_index(drop=True)
    ic, = ax.plot(df[ic])
    return ic


def plot_results(results_object,
                 specs=None,
                 colormap=None,
                 colorset=None,
                 figsize=(12, 8)):
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_axes([0.05, 0.05, 0.9, 0.9])
    ax2 = fig.add_axes([0.2, 0.65, 0.2, 0.2])
    ax3 = fig.add_axes([0.45, 0.65, 0.2, 0.2])
    plot_curve(results_object=results_object,
               specs=specs,
               ax=ax1,
               colormap=colormap,
               colorset=colorset)
    plot_ic(results_object=results_object,
            ic='bic',
            specs=specs,
            ax=ax2,
            colormap=colormap,
            colorset=colorset)
    plot_ic(results_object=results_object,
            ic='aic',
            specs=specs,
            ax=ax3,
            colormap=colormap,
            colorset=colorset)
    ax1.set_title('Estimates curve')
    ax2.set_title('BIC curve')
    ax3.set_title('AIC curve')
    return fig, ax1, ax2, ax3
