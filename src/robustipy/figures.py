import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm
import pandas as pd
import matplotlib as mpl
from robustipy.utils import get_selection_key
from robustipy.utils import get_colormap_colors
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
mpl.rcParams['font.family'] = 'Helvetica'


def plot_curve(results_object,
               specs=None,
               ax=None,
               colormap='Spectral_r'):
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
    colorset = get_colormap_colors(colormap, len(specs))
    if ax is None:
        ax = plt.gca()
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
                      color='#002d87',
                      linestyle='-')
    loess_min = sm.nonparametric.lowess(df['min'],
                                        pd.to_numeric(df.index),
                                        frac=0.05
                                        )
    loess_max = sm.nonparametric.lowess(df['max'],
                                        pd.to_numeric(df.index),
                                        frac=0.05
                                        )
    ax.plot(loess_min[:, 0], loess_min[:, 1], color='#002d87', linestyle='--')
    ax.plot(loess_max[:, 0], loess_max[:, 1], color='#002d87', linestyle='--')
    ax.fill_between(df.index,
                    loess_min[:, 1],
                    loess_max[:, 1],
                    facecolor='#FEE08B',
                    alpha=0.05)
    if ax.get_ylim()[0]<0 and ax.get_ylim()[1]>0:
        ax.axhline(y=0, color='k', ls='--')
    lines = []
    if specs:
        idxs = df.index[df['idx']].tolist()
        for idx, i in zip(idxs, range(len(specs))):
            control_names = list(df.spec_name.iloc[idx])
            label = ', '.join(control_names).title()
            lines.append(ax.vlines(x=idx,
                                   ymin=loess_min[idx, 1],
                                   ymax=loess_max[idx, 1],
                                   color=colorset[i],
                                   label=label))
            myArrow = FancyArrowPatch(posA=(idx, loess_min[idx, 1]),
                                      posB=(idx, loess_max[idx, 1]),
                                      arrowstyle='<|-|>',
                                      color=colorset[i],
                                      mutation_scale=20,
                                      shrinkA=0,
                                      shrinkB=0)
            ax.add_artist(myArrow)
            ax.plot(idx,
                    df.at[idx, 'median'],
                    'o',
                    markeredgecolor=colorset[i],
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
    return ax


def plot_ic(results_object,
            ic,
            specs=None,
            ax=None,
            colormap='Spectral_r'):
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
    colorset = get_colormap_colors(colormap, len(specs))
    df = results_object.summary_df.copy()

    df = df.sort_values(by='median')
    df = df.reset_index(drop=True)

    if specs:
        key = get_selection_key(specs)
        full_spec = list(results_object.specs_names.iloc[-1])
        full_spec_key = get_selection_key([full_spec])
        df['idx'] = df.spec_name.isin(key)
        df['full_spec_idx'] = df.spec_name.isin(full_spec_key)
        df = df.sort_values(by=ic).reset_index(drop=True)
        ic_fig, = ax.plot(df[ic], color='#002d87')
        idxs = df.index[df['idx']].tolist()
        ymin = ax.get_ylim()[0]
        ymax = ax.get_ylim()[1]
        ax.set_ylim(ymin, ymax)
        lines = []
        markers = []
        for idx, i in zip(idxs, range(len(specs))):
            control_names = list(df.spec_name.iloc[idx])
            label = ', '.join(control_names).title()
            lines.append(ax.vlines(x=idx,
                                   ymin=ymin,
                                   ymax=df.at[idx, ic],
                                   color=colorset[i],
                                   label=label)
                         )
            markers.append(Line2D([0], [0],
                                  marker='o',
                                  color=colorset[i],
                                  markerfacecolor='w',
                                  markersize=10,
                                  label=label)
                           )
            ax.plot(idx,
                    df.at[idx, ic],
                    'o',
                    markeredgecolor=colorset[i],
                    markerfacecolor='w',
                    markersize=15,
                    label=label)
        full_spec_pos = df.index[df['full_spec_idx']].to_list()[0]
        lines.append(ax.vlines(x=full_spec_pos,
                               ymin=ymin,
                               ymax=df.at[full_spec_pos, ic],
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
        ic_fig, = ax.plot(df[ic], color='#002d87')
        full_spec = list(results_object.specs_names.iloc[-1])
        full_spec_key = get_selection_key([full_spec])
        df['full_spec_idx'] = df.spec_name.isin(full_spec_key)
        ymin = ax.get_ylim()[0]
        ymax = ax.get_ylim()[1]
        ax.set_ylim(ymin, ymax)
        lines = []
        markers = []
        full_spec_pos = df.index[df['full_spec_idx']].to_list()[0]
        lines.append(ax.vlines(x=full_spec_pos,
                               ymin=ymin,
                               ymax=df.at[full_spec_pos, ic],
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
        return ic_fig


def plot_bdist(results_object,
               specs=None,
               ax=None,
               colormap='Spectral_r'):
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
    colorset = get_colormap_colors(colormap, len(specs))
    if ax is None:
        ax = plt.gca()
    df = results_object.estimates.copy().T
    df.columns = results_object.specs_names
    if specs:
        key = get_selection_key(specs)
        matching_cols = [col for col in df.columns if col in key]
        for i, col in enumerate(matching_cols):
            sns.kdeplot(df[col].squeeze(), common_norm=True, ax=ax, color=colorset[i], legend=False
                        )
    sns.kdeplot(df.iloc[:, -1:].squeeze(), common_norm=True, legend=False,
                color='black', ax=ax, label='Full Model')
    return ax


def plot_kfolds(results_object,
                colormap,
                ax=None,
                ):
    """

    Args:
        ax (object):
    """
    sns.kdeplot(results_object.summary_df['av_k_metric'], ax=ax, alpha=1,
                color=get_colormap_colors(colormap, 100)[99])
    sns.histplot(results_object.summary_df['av_k_metric'], ax=ax, alpha=1,
                 color=get_colormap_colors(colormap, 100)[0],
                 bins=30, stat='density')
    val_range = max(results_object.summary_df['av_k_metric']) - min(results_object.summary_df['av_k_metric'])
    min_lim = min(results_object.summary_df['av_k_metric']) - val_range *.1
    max_lim = max(results_object.summary_df['av_k_metric']) + val_range *.1
    ax.set_xlim(min_lim, max_lim)
    ax.yaxis.set_label_position("right")
    legend_elements = [
        Line2D([0], [0], color=get_colormap_colors(colormap, 100)[99], lw=2, linestyle='-',
               label=r'Kernel Density', alpha=1),
        Patch(facecolor=get_colormap_colors(colormap, 100)[0], edgecolor=(0, 0, 0, 1),
              label=r'Histogram')]
    ax.legend(handles=legend_elements,
              loc='upper left',
              frameon=True,
              fontsize=10,
              title='Out-of-Sample',
              title_fontsize=12,
              framealpha=1,
              facecolor='w',
              edgecolor=(0, 0, 0, 1),
              ncols=2
              )
    sns.despine(ax=ax, left=True)


def plot_bma(results_object, colormap, ax_left, ax_right):
    """
    Plots the Bayesian Model Averaging (BMA) probabilities and average coefficients.
    """
    bma = results_object.compute_bma()
    bma = bma.set_index('control_var')
    bma = bma.sort_values(by='probs', ascending=False)
    bma['probs'].plot(kind='barh',
                      ax=ax_left,
                      alpha=1,
                      color=get_colormap_colors(colormap, 100)[0],
                      edgecolor='k',
                      )
    bma['average_coefs'].plot(kind='barh',
                              ax=ax_right,
                              alpha=1,
                              color=get_colormap_colors(colormap, 100)[99],
                              edgecolor='k',
                              )
    ax_right.set_yticklabels([])
    ax_right.set_ylabel('')
    ax_left.set_ylabel('')
    legend_elements = [
        Patch(facecolor=get_colormap_colors(colormap, 100)[0], edgecolor=(0, 0, 0, 1),
              label='     BMA      \nProbabilities', alpha=1)
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
        Patch(facecolor=get_colormap_colors(colormap, 100)[99], edgecolor=(0, 0, 0, 1),
              label='     BMA      \nCoefficients', alpha=1)
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
                 colormap='Spectral_r',
                 figsize=(16, 12)
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
    gs = GridSpec(9, 24, wspace=0.5, hspace=1.5)
    ax7 = fig.add_subplot(gs[0:3, 0:12])
    ax8 = fig.add_subplot(gs[0:3, 13:24])
    ax4 = fig.add_subplot(gs[3:5, 0:6])
    ax6 = fig.add_subplot(gs[3:5, 6:12])
    ax5 = fig.add_subplot(gs[3:5, 12:23])
    ax1 = fig.add_subplot(gs[5:9, 0:15])
    ax2 = fig.add_subplot(gs[5:7, 16:23])
    ax3 = fig.add_subplot(gs[7:9, 16:23])

    ax2.axis('off')
    ax2.patch.set_alpha(0)
    ax3.axis('off')
    ax3.patch.set_alpha(0)
    ax5.axis('off')
    ax5.patch.set_alpha(0)
    plot_curve(results_object=results_object,
               specs=specs,
               ax=ax1,
               colormap=colormap)

    plot_kfolds(results_object, colormap, ax5)
    plot_bma(results_object, colormap, ax4, ax6)
    ax5.axis('on')
    ax5.patch.set_alpha(0.5)
    if ic is not None:
        lines = plot_ic(results_object=results_object,
                        ic=ic,
                        specs=specs,
                        ax=ax2,
                        colormap=colormap,
                        )
        ax2.axis('on')
        ax2.patch.set_alpha(0.5)
#    if specs is not None:
    plot_bdist(results_object=results_object,
               specs=specs,
               ax=ax3,
               colormap=colormap
               )
    ax3.axis('on')
    ax3.patch.set_alpha(0.5)
    ax1.set_title('f.',
                  loc='left',
                  fontsize=16,
                  y=1)
    ax2.set_title('g.',
                  loc='left',
                  fontsize=16,
                  y=1)
    ax3.set_title('h.',
                  loc='left',
                  fontsize=16,
                  y=1)
    ax4.set_title('c.',
                  loc='left',
                  fontsize=16,
                  y=1)
    ax5.set_title('e.',
                  loc='left',
                  fontsize=16,
                  y=1)
    ax6.set_title('d.',
                  loc='left',
                  fontsize=16,
                  y=1)
    ax7.set_title('a.',
                  loc='left',
                  fontsize=16,
                  y=1)
    ax8.set_title('b.',
                  loc='left',
                  fontsize=16,
                  y=1)
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:
        ax.tick_params(axis='both',
                        which='major',
                        labelsize=13)
        ax.grid(linestyle='--', color='k', alpha=0.15, zorder=-1)
        ax.set_axisbelow(True)
    for ax in [ax2, ax3]:
        ax.yaxis.set_label_position("right")
    sns.despine(ax=ax1)
    ax1.set_ylabel('Coefficient Estimates', fontsize=13)
    ax1.set_xlabel('Ordered Specifications', fontsize=13)
    if results_object.name_av_k_metric=='rmse':
        metric = results_object.name_av_k_metric.upper()
    elif results_object.name_av_k_metric=='R-Squared':
        metric = r'R$^2'
    else:
        metric = results_object.name_av_k_metric.title()
    ax5.set_xlabel('Out-of-Sample Evaluation Metric: ' + metric, fontsize=13)
    ax2.set_ylabel(f'{ic.upper()} curve', fontsize=13)
    ax2.set_xlabel('Ordered Specifications', fontsize=13)
    ax3.set_ylabel('Density', fontsize=13)
    ax5.set_ylabel('Density', fontsize=13)
    ax4.set_xlabel('Probabilities', fontsize=13)
    ax6.set_xlabel('Average Coefficients', fontsize=13)
    ax3.set_xlabel('Coefficient Estimate', fontsize=13)
    ax1.set_xlim(0, len(results_object.specs_names))
    ax1.set_ylim(ax1.get_ylim()[0] - (np.abs(ax1.get_ylim()[1]) - np.abs(ax1.get_ylim()[0])) / 20,
                 ax1.get_ylim()[1])
    ax1.text(
        0.05, 0.95,  # x, y coordinates
        (f'Number of specifications: {len(results_object.specs_names)}\n' +
         f'Number of bootstraps: {results_object.draws}\n' +
         f'Number of folds: {results_object.kfold}'
         ),  # The text string itself
        transform=ax1.transAxes,
        verticalalignment='top',
        horizontalalignment='left',
        color='black',
        fontsize=13,
        bbox=dict(facecolor='white',
                  edgecolor='black',
                  boxstyle='round,pad=1')
    )

    image = ax7.hexbin(results_object.estimates.stack(),
                       results_object.r2_values.stack(),
                       cmap=colormap, gridsize=20,  # extent=extent,
                       mincnt=1,
                       # bins='log',
                       edgecolor='k')
    cb = fig.colorbar(image, ax=ax7, spacing='uniform', pad=0.05, extend='max')
    data = image.get_array()
    ticks = np.linspace(data.min(), data.max(), num=6)  # Adjust this as needed
    cb.set_ticks(ticks)
    if (data.max() >= 1000) and (data.max() < 10000):
        cb.set_ticklabels([f'{tick / 1000:.1f}k' for tick in ticks])
    elif (data.max() >= 10000):
        cb.set_ticklabels([f'{tick / 1000:.0f}k' for tick in ticks])
    else:
        cb.set_ticklabels([f'{tick:.0f}' for tick in ticks])
    cb.ax.set_title('Count')
    ax7.set_ylabel(r'In-Sample R$^2$', fontsize=13)
    ax7.set_xlabel(r'Bootstrapped $\mathrm{\hat{\beta}}$ Coefficient Estimate', fontsize=13)
    image = ax8.hexbin([arr[0][0] for arr in results_object.all_b],
                       results_object.summary_df['ll'],
                       cmap=colormap,
                       gridsize=20,
                       # extent=extent,
                       mincnt=1,
                       #                   bins='log',
                       edgecolor='k')

    cb = fig.colorbar(image, ax=ax8, spacing='uniform', extend='max', pad=0.05)
    data = image.get_array()
    ticks = np.linspace(data.min(), data.max(), num=6)  # Adjust this as needed
    cb.set_ticks(ticks)
    if (data.max() >= 1000) and (data.max() < 10000):
        cb.set_ticklabels([f'{tick / 1000:.1f}k' for tick in ticks])
    elif (data.max() >= 10000):
        cb.set_ticklabels([f'{tick / 1000:.0f}k' for tick in ticks])
    else:
        cb.set_ticklabels([f'{tick:.0f}' for tick in ticks])
    cb.ax.set_title('Count')
    ax8.set_ylabel('')
    ax8.set_ylabel(r'Full Model Log Likelihood', fontsize=13)
    ax8.set_xlabel(r'Full-Sample $\mathrm{\hat{\beta}}$ Coefficient Estimates', fontsize=13)
    sns.despine()
    sns.despine(ax=ax2, right=False, left=True)
    sns.despine(ax=ax3, right=False, left=True)
    sns.despine(ax=ax4)
    sns.despine(ax=ax6)
    sns.despine(ax=ax5, right=False, left=True)
    #    plt.tight_layout()
#    return fig


#def vars_scatter_plot(results_object,
#                      var_name,
#                      ax=None,
#                      bin_size=1):
#    """
#    Plots the scatter plot of the specified covariate in the specifications.
#
#    Parameters:
#        results_object (object): Object containing the results data.
#        var_name (str): Name of the covariate to be plotted.
#        ax (matplotlib.axes._subplots.AxesSubplot, optional): Axes to plot on.
#        bin_size (int, optional): Size of bins for scatter plot.
#
#    Returns:
#        matplotlib.axes._subplots.AxesSubplot: Axes containing the scatter plot.
#    """
#    if ax is None:
#        ax = plt.gca()
#    df = results_object.summary_df.sort_values(by='median').copy()
#    count_bool = [var_name in ele for ele in df.spec_name]
#    index = []
#    for i, ele in enumerate(count_bool):
#        if ele:
#            new_index = np.floor(i / bin_size) * bin_size
#            index.append(new_index)
#    x = index
#    y = np.zeros(len(x)) + np.random.normal(0, .01, size=len(x))
#    ax.scatter(x, y, alpha=.2, s=50, linewidth=0, color='black')
#    ax.axis(ymin=-1, ymax=1)
#    ax.set_title(var_name)
#    ax.yaxis.label.set(rotation='horizontal', ha='right')
#    ax.tick_params(grid_alpha=0, colors='w')
#    return ax
#
#
#def vars_hist_plot(results_object,
#                   var_name,
#                   ax=None,
#                   bin_size=50):
#    """
#    Plots the histogram of the specified covariate in the specifications.
#
#    Parameters:
#        results_object (object): Object containing the results data.
#        var_name (str): Name of the covariate to be plotted.
#        ax (matplotlib.axes._subplots.AxesSubplot, optional): Axes to plot on.
#        bin_size (int, optional): Size of bins for histogram.
#
#    Returns:
#        matplotlib.axes._subplots.AxesSubplot: Axes containing the histogram plot.
#    """
#    if ax is None:
#        ax = plt.gca()
#    df = results_object.summary_df.sort_values(by='median').copy()
#    count_bool = [var_name in ele for ele in df.spec_name]
#    index = []
#    for i, ele in enumerate(count_bool):
#        if ele:
#            new_index = np.floor(i / 1) * 1
#            index.append(new_index)
#    x = index
#    ax.hist(x, bin_size, color='black')
#    ax.set_title(var_name)
#    ax.yaxis.label.set(rotation='horizontal', ha='right')
#    ax.tick_params(grid_alpha=0, colors='w')
#    return ax


#def vars_line_plot(results_object,
#                   var_name,
#                   ax=None,
#                   bin_size=None):
#    """
#    Plots the line plot of the specified covariate in the specifications.
#
#    Parameters:
#        results_object (object): Object containing the results data.
#        var_name (str): Name of the covariate to be plotted.
#        ax (matplotlib.axes._subplots.AxesSubplot, optional): Axes to plot on.
#        bin_size (int, optional): Size of bins for the line plot.
#
#    Returns:
#        matplotlib.axes._subplots.AxesSubplot: Axes containing the line plot.
#    """
#    if ax is None:
#        ax = plt.gca()
#    df = results_object.summary_df.sort_values(by='median').copy()
#    count_bool = [var_name in ele for ele in df.spec_name]
#    count_list = [int(ele) for ele in count_bool]
#    bin_sums = []
#    index = []
#    if bin_size is None:
#       bin_size = 50
#    for i, ele in enumerate(count_list):
#        if i % bin_size == 0:
#            bin_sum = np.sum(count_list[i - bin_size:i])
#            bin_sums.append(bin_sum)
#            index.append(i)
#    ax.plot(index, bin_sums, color='black')
#    ax.set_title(var_name)
#    ax.yaxis.label.set(rotation='horizontal', ha='right')
#    ax.tick_params(grid_alpha=0, colors='w')
#    return ax