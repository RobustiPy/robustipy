import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm
import pandas as pd
from robustipy.utils import get_selection_key
from robustipy.utils import get_default_colormap
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


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
    colors_curve = ['#001c54', '#E89818']
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
                      linestyle='-')
    loess_min = sm.nonparametric.lowess(df['min'],
                                        pd.to_numeric(df.index),
                                        frac=0.05
                                        )
    loess_max = sm.nonparametric.lowess(df['max'],
                                        pd.to_numeric(df.index),
                                        frac=0.05
                                        )
    ax.plot(loess_min[:, 0], loess_min[:, 1], color=colors_curve[0], linestyle='--')
    ax.plot(loess_max[:, 0], loess_max[:, 1], color=colors_curve[0], linestyle='--')
    ax.fill_between(df.index,
                    loess_min[:, 1],
                    loess_max[:, 1],
                    facecolor=colors_curve[1],
                    alpha=0.075)
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
                                   ymin=loess_min[idx, 1],
                                   ymax=loess_max[idx, 1],
                                   color=colors[i],
                                   label=label))
            myArrow = FancyArrowPatch(posA=(idx, loess_min[idx, 1]),
                                      posB=(idx, loess_max[idx, 1]),
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
                           ymin=loess_min[full_spec_pos, 1],
                           ymax=loess_max[full_spec_pos, 1],
                           color='k',
                           label='Full Model'))

    myArrow = FancyArrowPatch(posA=(full_spec_pos,
                                    loess_min[full_spec_pos, 1]),
                              posB=(full_spec_pos,
                                    loess_max[full_spec_pos, 1]),
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
              facecolor=((1, 1, 1, 0)
              )
              )
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

    df = df.sort_values(by='median')
    df = df.reset_index(drop=True)
    loess_max = sm.nonparametric.lowess(df['max'],
                                        pd.to_numeric(df.index),
                                        frac=0.05
                                        )

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

        lines = []
        markers = []
        for idx, i in zip(idxs, range(len(specs))):
            control_names = list(df.spec_name.iloc[idx])
            #            label = 'Controls: ' + ', '.join(control_names).title()
            label = ', '.join(control_names).title()
            lines.append(ax.vlines(x=idx,
                                   ymin=ymin,
                                   ymax=df.at[idx, ic],
                                   color=colors[i],
                                   label=label)
                         )
            markers.append(Line2D([0], [0],
                                  marker='o',
                                  color=colors[i],
                                  markerfacecolor='w',
                                  markersize=10,
                                  label=label)
                           )
            ax.plot(idx,
                    df.at[idx, ic],
                    'o',
                    markeredgecolor=colors[i],
                    markerfacecolor='w',
                    markersize=15,
                    label=label)
        full_spec_pos = df.index[df['full_spec_idx']].to_list()[0]
        lines.append(ax.vlines(x=full_spec_pos,
                               ymin=ymin,
                               ymax=loess_max[full_spec_pos, 1],
                               color='k',
                               label='Full Model'))
        markers.append(Line2D([0], [0], marker='o',
                              color='k',
                              markerfacecolor='w',
                              markersize=10,
                              label='Full Model')
                       )
        ax.plot(full_spec_pos,
                df.at[full_spec_pos, ic],
                'o',
                markeredgecolor='k',
                markerfacecolor='w',
                markersize=15)
        ax.legend(handles=markers,
                  frameon=True,
                  edgecolor=(0, 0, 0, 1),
                  fontsize=9,
                  loc="upper left",
                  ncols=1,
                  framealpha=1,
                  facecolor=((1, 1, 1, 0)
                  )
                  )

        return ic_fig, lines
    else:
        df = df.sort_values(by=ic).reset_index(drop=True)
        ic_fig, = ax.plot(df[ic])
        return ic_fig


def plot_bdist(results_object,
               lines,
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

        # @TODO this is _very_, _very_ hacky and needs fixing.
        # It's essentially taken from plot_ic()
        df = results_object.summary_df.copy()
        key = get_selection_key(specs)
        full_spec = list(results_object.specs_names.iloc[-1])
        full_spec_key = get_selection_key([full_spec])
        df['idx'] = df.spec_name.isin(key)
        df['full_spec_idx'] = df.spec_name.isin(full_spec_key)
        idxs = df.index[df['idx']].tolist()
        full_spec_pos = df.index[df['full_spec_idx']].to_list()[0]
        if colorset is None:
            colors = get_default_colormap(specs=specs)
        lines = []
        ymin = ax.get_ylim()[0]
        ymax = ax.get_ylim()[1]
        xmin = ax.get_xlim()[0]
        xmax = ax.get_xlim()[1]
        for idx, i in zip(idxs, range(len(specs))):
            control_names = list(df.spec_name.iloc[idx])
            label = ', '.join(control_names).title()
            lines.append(ax.vlines(x=-1000000000,
                                   linewidth=1,
                                   ymin=ymin,
                                   ymax=ymax,
                                   color=colors[i],
                                   label=label)
                         )
        full_spec_pos = df.index[df['full_spec_idx']].to_list()[0]
        lines.append(ax.vlines(x=-100000000,
                               linewidth=1,
                               ymin=ymin,
                               ymax=ymax,
                               color='k',
                               label='Full Model'))
        ax.set_xlim(xmin, xmax)
#        ax.legend(handles=lines,
#                  frameon=True,
#                  edgecolor=(0, 0, 0, 1),
#                  fontsize=9,
#                  loc="upper left",
#                  ncols=1,
#                  framealpha=1,
#                  facecolor=((1, 1, 1, 0)
#                  )
#                  )
        return plot
    else:
        return df[idx].plot(kind='density', ax=ax, legend=False)


def plot_kfolds(results_object,
                ax):
    # @TODO: hardcoding these is bad
    colors = ['#001c54', '#E89818', '#8b0000']
    sns.kdeplot(results_object.summary_df['av_k_metric'], ax=ax,
                color=colors[1])
    sns.histplot(results_object.summary_df['av_k_metric'], ax=ax,
                 color=colors[0],
                 bins=30, stat='density',
                 discrete=True)
    ax.yaxis.set_label_position("right")
    ax.grid(linestyle='--',
            color='k',
            alpha=0.15,
            zorder=100)
    legend_elements = [
        Line2D([0], [0], color=colors[1], lw=2, linestyle='-',
               label=r'Kernel Density', alpha=0.7),
        Patch(facecolor=colors[0], edgecolor=(0, 0, 0, 1),
              label=r'Histogram')]
    ax.legend(handles=legend_elements,
              loc='upper right',
              frameon=True,
              fontsize=10,
              title='Out-of-Sample',
              title_fontsize=12,
              framealpha=1,
              facecolor='w',
              edgecolor=(0, 0, 0, 1),
              ncols=1
              )
    sns.despine(ax=ax, left=True)


def plot_bma(results_object, ax_left, ax_right):
    colors = ['#001c54', '#E89818', '#8b0000']
    bma = results_object.compute_bma()
    bma = bma.set_index('control_var')
    bma = bma.sort_values(by='probs', ascending=False)
    bma['probs'].plot(kind='barh',
                      ax=ax_left,
                      color=(0 / 255, 28 / 255, 84 / 255, 0.75),
                      edgecolor='k',
                      )
    bma['average_coefs'].plot(kind='barh',
                              ax=ax_right,
                              color=(232 / 255, 152 / 255, 24 / 255, .75),
                              edgecolor='k',
                              )
    ax_right.set_yticklabels([])
    ax_right.set_ylabel('')
    ax_left.set_ylabel('')

    legend_elements = [
        Patch(facecolor=colors[0], edgecolor=(0, 0, 0, 1),
              label='     BMA      \nProbabilities', alpha=0.75)
    ]
    ax_left.legend(handles=legend_elements,
                   loc='upper right',
                   frameon=True,
                   fontsize=10,
                   framealpha=1,
                   facecolor='w',
                   edgecolor=(0, 0, 0, 1),
                   ncols=1
                   )

    legend_elements = [
        Patch(facecolor=colors[1], edgecolor=(0, 0, 0, 1),
              label='     BMA      \nCoefficients', alpha=0.75)
    ]
    ax_right.legend(handles=legend_elements,
                    loc='upper right',
                    frameon=True,
                    fontsize=10,
                    framealpha=1,
                    facecolor='w',
                    edgecolor=(0, 0, 0, 1),
                    ncols=1
                    )


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
    gs = GridSpec(6, 12, wspace=0.25, hspace=1)
    ax4 = fig.add_subplot(gs[0:2, 0:3])
    ax6 = fig.add_subplot(gs[0:2, 3:6])
    ax5 = fig.add_subplot(gs[0:2, 6:12])
    ax1 = fig.add_subplot(gs[2:6, 0:8])
    ax2 = fig.add_subplot(gs[2:4, 8:12])
    ax3 = fig.add_subplot(gs[4:6, 8:12])

    ax2.axis('off')
    ax2.patch.set_alpha(0)
    ax3.axis('off')
    ax3.patch.set_alpha(0)
    ax5.axis('off')
    ax5.patch.set_alpha(0)
    plot_curve(results_object=results_object,
               specs=specs,
               ax=ax1,
               colormap=colormap,
               colorset=colorset)

    plot_kfolds(results_object, ax5)
    plot_bma(results_object, ax4, ax6)
    ax5.axis('on')
    ax5.patch.set_alpha(0.5)
    if ic is not None:
        lines = plot_ic(results_object=results_object,
                        ic=ic,
                        specs=specs,
                        ax=ax2,
                        colormap=colormap,
                        colorset=colorset
                        )
        ax2.axis('on')
        ax2.patch.set_alpha(0.5)
    if specs is not None:
        plot_bdist(results_object=results_object,
                   lines=lines,
                   specs=specs,
                   ax=ax3,
                   colormap=colormap,
                   colorset=colorset)
        ax3.axis('on')
        ax3.patch.set_alpha(0.5)
    ax1.set_title('d.',
                  loc='left',
                  fontsize=16,
                  y=1)
    ax2.set_title('e.',
                  loc='left',
                  fontsize=16,
                  y=1)
    ax3.set_title('f.',
                  loc='left',
                  fontsize=16,
                  y=1)

    ax4.set_title('a.',
                  loc='left',
                  fontsize=16,
                  y=1)

    ax5.set_title('c.',
                  loc='left',
                  fontsize=16,
                  y=1)

    ax6.set_title('b.',
                  loc='left',
                  fontsize=16,
                  y=1)
    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax1.tick_params(axis='both',
                        which='major',
                        labelsize=13)
        ax2.tick_params(axis='both',
                        which='major',
                        labelsize=13)
        ax3.tick_params(axis='both',
                        which='major',
                        labelsize=13)
        ax4.tick_params(axis='both',
                        which='major',
                        labelsize=13)
        ax5.tick_params(axis='both',
                        which='major',
                        labelsize=13)
        ax6.tick_params(axis='both',
                        which='major',
                        labelsize=13)

    for ax in [ax2, ax3]:
        ax.yaxis.set_label_position("right")
    ax1.text(ax1.get_xlim()[1] * .05, ax1.get_ylim()[1] * .85,
             (f'Number of specifications:     {len(results_object.specs_names)}\n' +
              f'Number of bootstraps:          {results_object.draws}\n' +
              'Number of folds:                   100'
              ),
             color='black',
             fontsize=13,
             bbox=dict(facecolor='white',
                       edgecolor='black',
                       boxstyle='round, pad=1'))
    sns.despine(ax=ax1)
    ax1.set_ylabel('Coefficient Estimates', fontsize=13)
    ax1.set_xlabel('Ordered Specifications', fontsize=13)
    ax5.set_xlabel('Kullback-Leibler Divergence', fontsize=13)
    ax2.set_ylabel(f'{ic.upper()} curve', fontsize=13)
    ax2.set_xlabel('Ordered Specifications', fontsize=13)
    ax3.set_ylabel('Density', fontsize=13)
    ax5.set_ylabel('Density', fontsize=13)
    ax4.set_xlabel('Probabilities', fontsize=13)
    ax6.set_xlabel('Average Coefficients', fontsize=13)
    ax3.set_xlabel('Coefficient Estimate', fontsize=13)
    ax2.set_axisbelow(True)
    ax3.set_axisbelow(True)
    ax2.grid(linestyle='--', color='k', alpha=0.15, zorder=0)
    ax3.grid(linestyle='--', color='k', alpha=0.15, zorder=0)
    ax4.grid(linestyle='--', color='k', alpha=0.15, zorder=0)
    ax5.grid(linestyle='--', color='k', alpha=0.15, zorder=0)
    ax6.grid(linestyle='--', color='k', alpha=0.15, zorder=0)
    ax1.set_xlim(0, len(results_object.specs_names))
    ax1.set_ylim(ax1.get_ylim()[0] - (np.abs(ax1.get_ylim()[1]) - np.abs(ax1.get_ylim()[0])) / 20,
                 ax1.get_ylim()[1])
    sns.despine(ax=ax2, right=False, left=True)
    sns.despine(ax=ax3, right=False, left=True)
    sns.despine(ax=ax4)
    sns.despine(ax=ax6)
    sns.despine(ax=ax5, right=False, left=True)
    #    plt.tight_layout()
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
            new_index = np.floor(i / bin_size) * bin_size
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
            new_index = np.floor(i / 1) * 1
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
            bin_sum = np.sum(count_list[i - bin_size:i])
            bin_sums.append(bin_sum)
            index.append(i)
    ax.plot(index, bin_sums, color='black')
    ax.set_title(var_name)
    ax.yaxis.label.set(rotation='horizontal', ha='right')
    ax.tick_params(grid_alpha=0, colors='w')
    return ax