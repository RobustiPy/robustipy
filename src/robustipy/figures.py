import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from robustipy.utils import get_selection_key
from robustipy.utils import get_default_colormap
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch


def plot_curve(results_object,
               specs=None,
               ax=None,
               colormap=None,
               colorset=None):
    """
    Plots the curve of median, confidence intervals, minimum, and maximum
    coefficient estimates for a given results object.

    Parameters:
        results_object (object): Object containing the results data.
        specs (list, optional): List of specification names to be highlighted.
        ax (matplotlib.axes._subplots.AxesSubplot, optional): Axes to plot on.
        colormap (str, optional): Colormap to use for highlighting specifications.
        colorset (list, optional): List of colors for specification highlighting.

    Returns:
        matplotlib.axes._subplots.AxesSubplot: Axes containing the plotted curve.
    """
    colors_curve = ['#001c54', 'goldenrod']
    if ax is None:
        ax = plt.gca()
    if colormap is None:
        colormap = 'Set1'
    df = results_object.summary_df.copy()
    full_spec = list(results_object.specs_names.iloc[-1])
    full_spec_key = get_selection_key([full_spec])
    df['full_spec_idx'] = df.spec_name.isin(full_spec_key)
    if specs:
        key = get_selection_key(specs)
        df['idx'] = df.spec_name.isin(key)
    df = df.sort_values(by='median')
    df = df.reset_index(drop=True)
    df['median'].plot(ax=ax,
                      color=colors_curve[0],
                      linestyle='--')
    df['ci_up'].plot(ax=ax,
                     color=colors_curve[1],
                     alpha=0.4,
                     linewidth=1)
    df['ci_down'].plot(ax=ax,
                       color=colors_curve[1],
                       alpha=0.4,
                       linewidth=1)
    df['min'].plot(ax=ax,
                   color=colors_curve[1],
                   alpha=0.5,
                   linewidth=1)
    df['max'].plot(ax=ax,
                   color=colors_curve[1],
                   alpha=0.5,
                   linewidth=1)
    ci_up = df['ci_up']
    ci_down = df['ci_down']
    min = df['min']
    max = df['max']
    ax.fill_between(df.index,
                    ci_up,
                    ci_down,
                    facecolor=colors_curve[1],
                    alpha=0.25)
    ax.fill_between(df.index,
                    min,
                    max,
                    facecolor=colors_curve[1],
                    alpha=0.2)
    ax.axhline(y=0,
               color='k',
               ls='--')
    lines = []
    if specs:
        idxs = df.index[df['idx']].tolist()
        if colorset is None:
            colors = get_default_colormap(specs)
        for idx, i in zip(idxs, range(len(specs))):
            control_names = list(df.spec_name.iloc[idx])
            label = 'Controls: ' + ', '.join(control_names).title()
            label = ', '.join(control_names).title()
            lines.append(ax.vlines(x=idx,
                                   ymin=df.at[idx, 'min'],
                                   ymax=df.at[idx, 'max'],
                                   color=colors[i],
                                   label=label))
            myArrow = FancyArrowPatch(posA=(idx, df.at[idx, 'min']),
                                      posB=(idx, df.at[idx, 'max']),
                                      arrowstyle='<|-|>',
                                      color=colors[i],
                                      mutation_scale=20,
                                      shrinkA=0,
                                      shrinkB=0)
            ax.add_artist(myArrow)
            ax.plot(idx,
                    df.at[idx, 'median'],
                    'o',
                    markeredgecolor=colors[i],
                    markerfacecolor='w',
                    markersize=15)
    full_spec_pos = df.index[df['full_spec_idx']].to_list()[0]
    lines.append(ax.vlines(x=full_spec_pos,
                           ymin=df['min'].iloc[full_spec_pos],
                           ymax=df['max'].iloc[full_spec_pos],
                           color='k',
                           label='Full Model'))

    myArrow = FancyArrowPatch(posA=(full_spec_pos,
                                    df['min'].iloc[full_spec_pos]),
                              posB=(full_spec_pos,
                                    df['max'].iloc[full_spec_pos]),
                              arrowstyle='<|-|>',
                              color='k',
                              mutation_scale=20,
                              shrinkA=0,
                              shrinkB=0)
    ax.add_artist(myArrow)
    ax.plot(full_spec_pos,
            df['median'].iloc[full_spec_pos],
            'o',
            markeredgecolor='k',
            markerfacecolor='w',
            markersize=15)
    ax.legend(handles=lines,
              frameon=True,
              edgecolor=(0, 0, 0, 1),
              fontsize=13,
              loc="lower center",
              ncols=4,
              framealpha=1,
              facecolor='w')
    ax.set_axisbelow(False)
    ax.grid(linestyle='--',
            color='k',
            alpha=0.15,
            zorder=100)
    return ax


def plot_ic(results_object,
            ic,
            specs=None,
            ax=None,
            colormap=None,
            colorset=None):
    """
    Plots the information criterion (IC) curve for the given results object.

    Parameters:
        results_object (object): Object containing the results data.
        ic (str): Information criterion to plot ('aic', 'bic', etc.).
        specs (list, optional): List of specification names to be highlighted.
        ax (matplotlib.axes._subplots.AxesSubplot, optional): Axes to plot on.
        colormap (str, optional): Colormap to use for highlighting specifications.
        colorset (list, optional): List of colors for specification highlighting.

    Returns:
        matplotlib.lines.Line2D: IC curve plot.
    """
    if ax is None:
        ax = plt.gca()
    df = results_object.summary_df.copy()
    if specs:
        key = get_selection_key(specs)
        full_spec = list(results_object.specs_names.iloc[-1])
        full_spec_key = get_selection_key([full_spec])
        df['idx'] = df.spec_name.isin(key)
        df['full_spec_idx'] = df.spec_name.isin(full_spec_key)
        df = df.sort_values(by=ic).reset_index(drop=True)
        ic_fig, = ax.plot(df[ic], color='#001c54')
        idxs = df.index[df['idx']].tolist()
        full_spec_pos = df.index[df['full_spec_idx']].to_list()[0]
        if colorset is None:
            colors = get_default_colormap(specs=specs)
        ymin = ax.get_ylim()[0]
        ymax = ax.get_ylim()[1]
        ax.set_ylim(ymin, ymax)
        for idx, i in zip(idxs, range(len(specs))):
            ax.vlines(x=idx,
                      ymin=ymin,
                      ymax=df.at[idx, ic],
                      color=colors[i])
            ax.plot(idx,
                    df.at[idx, ic],
                    'o',
                    markeredgecolor=colors[i],
                    markerfacecolor='w',
                    markersize=10)
        ax.vlines(x=full_spec_pos,
                  ymin=ymin,
                  ymax=df.at[full_spec_pos, ic],
                  color='k')
        ax.plot(full_spec_pos,
                df.at[full_spec_pos, ic],
                'o',
                markeredgecolor='k',
                markerfacecolor='w',
                markersize=10)
        return ic_fig
    else:
        df = df.sort_values(by=ic).reset_index(drop=True)
        ic_fig, = ax.plot(df[ic])
        return ic_fig


def plot_bdist(results_object,
               specs=None,
               ax=None,
               colormap=None,
               colorset=None):
    """
    Plots the distribution of coefficient estimates for the specified specifications.

    Parameters:
        results_object (object): Object containing the results data.
        specs (list, optional): List of specification names to be highlighted.
        ax (matplotlib.axes._subplots.AxesSubplot, optional): Axes to plot on.
        colormap (str, optional): Colormap to use for highlighting specifications.
        colorset (list, optional): List of colors for specification highlighting.

    Returns:
        matplotlib.axes._subplots.AxesSubplot: Axes containing the plotted distribution.
    """
    if ax is None:
        ax = plt.gca()
    df = results_object.estimates.T
    df.columns = results_object.specs_names
    idx = get_selection_key(specs)
    if colorset is None:
        colors = get_default_colormap(specs)
        plot = df[idx].plot(kind='density',
                            ax=ax,
                            color=colors,
                            legend=False)
        plot = df.iloc[:, -1:].plot(kind='density',
                                    legend=False,
                                    color='k', ax=ax)
        return plot
    else:
        return df[idx].plot(kind='density', ax=ax, legend=False)


def plot_results(results_object,
                 specs=None,
                 ic=None,
                 colormap=None,
                 colorset=None,
                 figsize=(26, 8)
                 ):
    """
    Plots the coefficient estimates, IC curve, and distribution plots for the given results object.

    Parameters:
        results_object (object): Object containing the results data.
        specs (list, optional): List of specification names to be highlighted.
        ic (str, optional): Information criterion to plot ('aic', 'bic', etc.).
        colormap (str, optional): Colormap to use for highlighting specifications.
        colorset (list, optional): List of colors for specification highlighting.
        figsize (tuple, optional): Figure size (width, height) in inches.

    Returns:
        matplotlib.figure.Figure: Figure containing the plotted results.
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 4, wspace=0.1, hspace=.3)
    ax1 = fig.add_subplot(gs[:, :-1])
    ax2 = fig.add_subplot(gs[:-1, -1])
    ax3 = fig.add_subplot(gs[-1, -1])

    ax2.axis('off')
    ax2.patch.set_alpha(0)
    ax3.axis('off')
    ax3.patch.set_alpha(0)
    plot_curve(results_object=results_object,
               specs=specs,
               ax=ax1,
               colormap=colormap,
               colorset=colorset)
    if ic is not None:
        plot_ic(results_object=results_object,
                ic=ic,
                specs=specs,
                ax=ax2,
                colormap=colormap,
                colorset=colorset)
        ax2.axis('on')
        ax2.patch.set_alpha(0.5)
    if specs is not None:
        plot_bdist(results_object=results_object,
                   specs=specs,
                   ax=ax3,
                   colormap=colormap,
                   colorset=colorset)
        ax3.axis('on')
        ax3.patch.set_alpha(0.5)
    ax1.set_title('A.',
                  loc='left',
                  fontsize=16,
                  y=1)
    ax2.set_title('B.',
                  loc='left',
                  fontsize=16,
                  y=1)
    ax3.set_title('C.',
                  loc='left',
                  fontsize=16,
                  y=1)
    ax1.tick_params(axis='both',
                    which='major',
                    labelsize=13)
    ax2.tick_params(axis='both',
                    which='major',
                    labelsize=13)
    ax3.tick_params(axis='both',
                    which='major',
                    labelsize=13)
    for ax in [ax2, ax3]:
        ax.yaxis.set_label_position("right")
    ax1.text(ax1.get_xlim()[1]*.05, ax1.get_ylim()[1]*.9,
             f'Number of specifications:{len(results_object.specs_names)}\nNumber of bootstraps done: {results_object.draws}',
             color='black',
             fontsize=13,
             bbox=dict(facecolor='none',
                       edgecolor='black',
                       boxstyle='round, pad=1'))
    sns.despine(ax=ax1)
    ax1.set_ylabel('Coefficient Estimates', fontsize=13)
    ax1.set_xlabel('Ordered Specifications', fontsize=13)
    ax2.set_ylabel(f'{ic.upper()} curve', fontsize=13)
    ax2.set_xlabel('Ordered Specifications', fontsize=13)
    ax3.set_ylabel('Density', fontsize=13)
    ax3.set_xlabel('Coefficient Estimate', fontsize=13)
    ax2.set_axisbelow(True)
    ax3.set_axisbelow(True)
    ax2.grid(linestyle='--', color='k', alpha=0.15, zorder=0)
    ax3.grid(linestyle='--', color='k', alpha=0.15, zorder=0)
    ax1.set_xlim(0, len(results_object.specs_names))
    sns.despine(ax=ax2, right=False, left=True)
    sns.despine(ax=ax3, right=False, left=True)
    return fig


def vars_scatter_plot(results_object,
                      var_name,
                      ax=None,
                      bin_size=1):
    """
    Plots the scatter plot of the specified covariate in the specifications.

    Parameters:
        results_object (object): Object containing the results data.
        var_name (str): Name of the covariate to be plotted.
        ax (matplotlib.axes._subplots.AxesSubplot, optional): Axes to plot on.
        bin_size (int, optional): Size of bins for scatter plot.

    Returns:
        matplotlib.axes._subplots.AxesSubplot: Axes containing the scatter plot.
    """
    if ax is None:
        ax = plt.gca()
    df = results_object.summary_df.sort_values(by='median').copy()
    count_bool = [var_name in ele for ele in df.spec_name]
    index = []
    for i, ele in enumerate(count_bool):
        if ele:
            new_index = np.floor(i/bin_size)*bin_size
            index.append(new_index)
    x = index
    y = np.zeros(len(x)) + np.random.normal(0, .01, size=len(x))
    ax.scatter(x, y, alpha=.2, s=50, linewidth=0, color='black')
    ax.axis(ymin=-1, ymax=1)
    ax.set_title(var_name)
    ax.yaxis.label.set(rotation='horizontal', ha='right')
    ax.tick_params(grid_alpha=0, colors='w')
    return ax


def vars_hist_plot(results_object,
                   var_name,
                   ax=None,
                   bin_size=50):
    """
    Plots the histogram of the specified covariate in the specifications.

    Parameters:
        results_object (object): Object containing the results data.
        var_name (str): Name of the covariate to be plotted.
        ax (matplotlib.axes._subplots.AxesSubplot, optional): Axes to plot on.
        bin_size (int, optional): Size of bins for histogram.

    Returns:
        matplotlib.axes._subplots.AxesSubplot: Axes containing the histogram plot.
    """
    if ax is None:
        ax = plt.gca()
    df = results_object.summary_df.sort_values(by='median').copy()
    count_bool = [var_name in ele for ele in df.spec_name]
    index = []
    for i, ele in enumerate(count_bool):
        if ele:
            new_index = np.floor(i/1)*1
            index.append(new_index)
    x = index
    ax.hist(x, bin_size, color='black')
    ax.set_title(var_name)
    ax.yaxis.label.set(rotation='horizontal', ha='right')
    ax.tick_params(grid_alpha=0, colors='w')
    return ax


def vars_line_plot(results_object,
                   var_name,
                   ax=None,
                   bin_size=None):
    """
    Plots the line plot of the specified covariate in the specifications.

    Parameters:
        results_object (object): Object containing the results data.
        var_name (str): Name of the covariate to be plotted.
        ax (matplotlib.axes._subplots.AxesSubplot, optional): Axes to plot on.
        bin_size (int, optional): Size of bins for the line plot.

    Returns:
        matplotlib.axes._subplots.AxesSubplot: Axes containing the line plot.
    """
    if ax is None:
        ax = plt.gca()
    df = results_object.summary_df.sort_values(by='median').copy()
    count_bool = [var_name in ele for ele in df.spec_name]
    count_list = [int(ele) for ele in count_bool]
    bin_sums = []
    index = []
    if bin_size is None:
        bin_size = 50
    for i, ele in enumerate(count_list):
        if i % bin_size == 0:
            bin_sum = np.sum(count_list[i-bin_size:i])
            bin_sums.append(bin_sum)
            index.append(i)
    ax.plot(index, bin_sums, color='black')
    ax.set_title(var_name)
    ax.yaxis.label.set(rotation='horizontal', ha='right')
    ax.tick_params(grid_alpha=0, colors='w')
    return ax
