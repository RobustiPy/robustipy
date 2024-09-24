from __future__ import annotations
import matplotlib.cm as cm
from scipy.stats import gaussian_kde
from shap.plots._labels import labels
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
plt.rcParams['axes.unicode_minus'] = False
mpl.rcParams['font.family'] = 'Helvetica'


def shap_violin(
        ax,
        shap_values,
        features=None,
        feature_names=None,
        max_display=10,
        color=None,
        alpha=1,
        cmap='Spectral_r',
        use_log_scale=False,
):
    """Create a SHAP beeswarm plot, colored by feature values when they are provided.

    Parameters
    ----------
    shap_values : numpy.array
        For single output explanations this is a matrix of SHAP values (# samples x # features).
        For multi-output explanations this is a list of such matrices of SHAP values.

    features : numpy.array or pandas.DataFrame or list
        Matrix of feature values (# samples x # features) or a feature_names list as shorthand

    feature_names : list
        Names of the features (length # features)

    max_display : int
        How many top features to include in the plot (default is 20, or 7 for interaction plots)
    """
    if str(type(shap_values)).endswith("Explanation'>"):
        shap_exp = shap_values
        shap_values = shap_exp.values
        if features is None:
            features = shap_exp.data
        if feature_names is None:
            feature_names = shap_exp.feature_names
        if len(shap_exp.base_values.shape) == 2 and shap_exp.base_values.shape[1] > 2:
            shap_values = [shap_values[:, :, i] for i in range(shap_exp.base_values.shape[1])]
    if isinstance(features, pd.DataFrame):
        if feature_names is None:
            feature_names = features.columns
        features = features.values
    elif isinstance(features, list):
        if feature_names is None:
            feature_names = features
        features = None
    elif (features is not None) and len(features.shape) == 1 and feature_names is None:
        feature_names = features
        features = None
    num_features = shap_values.shape[1]
    if features is not None:
        shape_msg = "The shape of the shap_values matrix does not match the shape of the provided data matrix."
        if num_features - 1 == features.shape[1]:
            raise ValueError(
                shape_msg + " Perhaps the extra column in the shap_values matrix is the "
                            "constant offset? Of so just pass shap_values[:,:-1]."
            )
        else:
            assert num_features == features.shape[1], shape_msg
    if feature_names is None:
        feature_names = np.array([labels["FEATURE"] % str(i) for i in range(num_features)])
    if use_log_scale:
        ax.xscale("symlog")
    if max_display is None:
        max_display = 20
    feature_order = np.argsort(np.sum(np.abs(shap_values), axis=0))
    feature_order = feature_order[-min(max_display, len(feature_order)):]
    for pos in range(len(feature_order)):
        ax.axhline(y=pos, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)

    if features is not None:
        global_low = np.nanpercentile(shap_values[:, : len(feature_names)].flatten(), 1)
        global_high = np.nanpercentile(shap_values[:, : len(feature_names)].flatten(), 99)
        for pos, i in enumerate(feature_order):
            shaps = shap_values[:, i]
            shap_min, shap_max = np.min(shaps), np.max(shaps)
            rng = shap_max - shap_min
            xs = np.linspace(np.min(shaps) - rng * 0.2, np.max(shaps) + rng * 0.2, 100)
            if np.std(shaps) < (global_high - global_low) / 100:
                ds = gaussian_kde(shaps + np.random.randn(len(shaps)) * (global_high - global_low) / 100)(xs)
            else:
                ds = gaussian_kde(shaps)(xs)
            ds /= np.max(ds) * 3
            values = features[:, i]
            smooth_values = np.zeros(len(xs) - 1)
            sort_inds = np.argsort(shaps)
            trailing_pos = 0
            leading_pos = 0
            running_sum = 0
            back_fill = 0
            for j in range(len(xs) - 1):
                while leading_pos < len(shaps) and xs[j] >= shaps[sort_inds[leading_pos]]:
                    running_sum += values[sort_inds[leading_pos]]
                    leading_pos += 1
                    if leading_pos - trailing_pos > 20:
                        running_sum -= values[sort_inds[trailing_pos]]
                        trailing_pos += 1
                if leading_pos - trailing_pos > 0:
                    smooth_values[j] = running_sum / (leading_pos - trailing_pos)
                    for k in range(back_fill):
                        smooth_values[j - k - 1] = smooth_values[j]
                else:
                    back_fill += 1
            vmin = np.nanpercentile(values, 5)
            vmax = np.nanpercentile(values, 95)
            if vmin == vmax:
                vmin = np.nanpercentile(values, 1)
                vmax = np.nanpercentile(values, 99)
                if vmin == vmax:
                    vmin = np.min(values)
                    vmax = np.max(values)
            nan_mask = np.isnan(values)
            ax.scatter(
                shaps[nan_mask],
                np.ones(shap_values[nan_mask].shape[0]) * pos,
                color="#777777",
                s=9,
                alpha=alpha,
                linewidth=0,
                zorder=1,
            )
            cvals = values[np.invert(nan_mask)].astype(np.float64)
            cvals_imp = cvals.copy()
            cvals_imp[np.isnan(cvals)] = (vmin + vmax) / 2.0
            cvals[cvals_imp > vmax] = vmax
            cvals[cvals_imp < vmin] = vmin
            ax.scatter(
                shaps[np.invert(nan_mask)],
                np.ones(shap_values[np.invert(nan_mask)].shape[0]) * pos,
                cmap='Spectral_r',
                vmin=vmin,
                vmax=vmax,
                s=9,
                c=cvals,
                alpha=alpha,
                linewidth=0,
                zorder=1,
            )
            smooth_values -= vmin
            if vmax - vmin > 0:
                smooth_values /= vmax - vmin
            from matplotlib.colors import LinearSegmentedColormap
            for i in range(len(xs) - 1):
                if ds[i] > 0.05 or ds[i + 1] > 0.05:
                    ax.fill_between(
                        [xs[i], xs[i + 1]],
                        [pos + ds[i], pos + ds[i + 1]],
                        [pos - ds[i], pos - ds[i + 1]],
                        color=plt.get_cmap(cmap)(smooth_values[i]),
                        zorder=2,
                    )
    else:
        parts = ax.violinplot(
            shap_values[:, feature_order],
            range(len(feature_order)),
            points=200,
            vert=False,
            widths=0.7,
            showmeans=False,
            showextrema=False,
            showmedians=False,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_edgecolor("none")
            pc.set_alpha(alpha)

    m = cm.ScalarMappable(cmap=cmap)
    m.set_array([0, 1])
    cb = plt.colorbar(m, ax=ax, ticks=[0, 1], aspect=20)
    cb.set_ticklabels(['Low', 'High'])
    cb.set_label('Feature Value', size=12, labelpad=-20)
    cb.outline.set_edgecolor('k')
    cb.ax.tick_params(labelsize=11, length=0)
    cb.set_alpha(1)
    cb.outline.set_visible(True)
    ax.xaxis.set_ticks_position("bottom")
    ax.spines["left"].set_visible(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    feature_name_order = [feature_names[i] for i in feature_order]
    ax.set_yticks(range(len(feature_order)), feature_name_order, fontsize=13)
    ax.set_ylim(-1, len(feature_order))
    ax.set_xlabel('SHAP Values', fontsize=13)
    return feature_name_order



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
              ncols=1
              )
    sns.despine(ax=ax, left=True)


def plot_bma(results_object, colormap, ax_left, feature_order):
    """
    Plots the Bayesian Model Averaging (BMA) probabilities and average coefficients.
    """
    bma = results_object.compute_bma()
    bma = bma.set_index('control_var')
    bma = bma.reindex(feature_order)
#    bma = bma.sort_values(by='probs', ascending=False)
    bma['probs'].plot(kind='barh',
                      ax=ax_left,
                      alpha=1,
                      color=get_colormap_colors(colormap, 100)[0],
                      edgecolor='k',
                      )
#    bma['average_coefs'].plot(kind='barh',
#                              ax=ax_right,
#                              alpha=1,
#                              color=get_colormap_colors(colormap, 100)[99],
#                              edgecolor='k',
#                              )
#    ax_right.set_yticklabels([])
#    ax_right.set_ylabel('')
    ax_left.set_ylabel('')
#    legend_elements = [
#        Patch(facecolor=get_colormap_colors(colormap, 100)[0], edgecolor=(0, 0, 0, 1),
#              label='     BMA      \nProbabilities', alpha=1)
#    ]
#    ax_left.legend(handles=legend_elements,
#                   loc='upper right',
#                   frameon=True,
#                   fontsize=10,
#                   framealpha=1,
#                   facecolor='w',
#                   edgecolor=(0, 0, 0, 1),
#                   ncols=1
#                   )

#    legend_elements = [
#        Patch(facecolor=get_colormap_colors(colormap, 100)[99], edgecolor=(0, 0, 0, 1),
#              label='     BMA      \nCoefficients', alpha=1)
#    ]
#    ax_right.legend(handles=legend_elements,
#                    loc='upper right',
#                    frameon=True,
#                    fontsize=10,
#                    framealpha=1,
#                    facecolor='w',
#                    edgecolor=(0, 0, 0, 1),
#                    ncols=1
#                    )

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
    ax4 = fig.add_subplot(gs[3:5, 0:7])
    ax6 = fig.add_subplot(gs[3:5, 7:14], sharey=ax4)  #
    ax5 = fig.add_subplot(gs[3:5, 14:23])
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

    feature_order = shap_violin(ax6,
                                np.delete(results_object.shap_return[0], 0, axis=1),
                                results_object.shap_return[1].drop(results_object.x_name, axis=1).to_numpy(),
                                results_object.shap_return[1].drop(results_object.x_name, axis=1).columns
                                )

    plot_bma(results_object, colormap, ax4, feature_order)
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
    ax4.set_xlabel('BMA Probabilities', fontsize=13)
    ax6.set_xlabel('SHAP Values', fontsize=13)
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