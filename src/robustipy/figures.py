from __future__ import annotations
import os
import matplotlib.cm as cm
from scipy.stats import gaussian_kde
from shap.plots._labels import labels
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm
import pandas as pd
import matplotlib.ticker as mticker
from robustipy.utils import get_selection_key
from robustipy.utils import get_colormap_colors
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
plt.rcParams['axes.unicode_minus'] = False


def axis_formatter(ax, ylabel, xlabel, title):
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.grid(linestyle='--', color='k', alpha=0.1, zorder=-1)
    ax.set_axisbelow(True)
    ax.set_ylabel(ylabel, fontsize=13)
    title_setter(ax, title)
    ax.set_xlabel(xlabel, fontsize=13)

def plot_hexbin_r2(results_object, ax, fig, colormap, title=''):
    image = ax.hexbin(results_object.estimates.stack(),
                      results_object.r2_values.stack(),
                      cmap=colormap, gridsize=20,
                      mincnt=1,
                      edgecolor='k')
    cb = fig.colorbar(image, ax=ax, spacing='uniform', pad=0.05, extend='max')
    data = image.get_array()
    ticks = np.linspace(data.min(), data.max(), num=6)
    cb.set_ticks(ticks)
    if (data.max() >= 1000) and (data.max() < 10000):
        cb.set_ticklabels([f'{tick / 1000:.1f}k' for tick in ticks])
    elif data.max() >= 10000:
        cb.set_ticklabels([f'{tick / 1000:.0f}k' for tick in ticks])
    else:
        cb.set_ticklabels([f'{tick:.0f}' for tick in ticks])
    cb.ax.set_title('Count')
    axis_formatter(ax, r'In-Sample R$^2$', r'Bootstrapped $\mathrm{\hat{\beta}}$ Coefficient Estimate', title)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(4))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(4))
    sns.despine(ax=ax)


def plot_hexbin_log(results_object, ax, fig, colormap, title=''):
    image = ax.hexbin([arr[0][0] for arr in results_object.all_b],
                      results_object.summary_df['ll'],
                      cmap=colormap,
                      gridsize=20,
                      mincnt=1,
                      edgecolor='k')
    cb = fig.colorbar(image, ax=ax, spacing='uniform', extend='max', pad=0.05)
    data = image.get_array()
    ticks = np.linspace(data.min(), data.max(), num=6)
    cb.set_ticks(ticks)
    if (data.max() >= 1000) and (data.max() < 10000):
        cb.set_ticklabels([f'{tick / 1000:.1f}k' for tick in ticks])
    elif data.max() >= 10000:
        cb.set_ticklabels([f'{tick / 1000:.0f}k' for tick in ticks])
    else:
        cb.set_ticklabels([f'{tick:.0f}' for tick in ticks])
    cb.ax.set_title('Count')
    axis_formatter(ax, r'Full Model Log Likelihood', r'Full-Sample $\mathrm{\hat{\beta}}$ Coefficient Estimates', title)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(4))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(4))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
    sns.despine(ax=ax)

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
        title='',
        clear_yticklabels=False
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
    axis_formatter(ax, r'', r'SHAP Values', title)
    title_setter(ax, title)
    if clear_yticklabels:
        ax.set_yticklabels([])
    return feature_name_order


def plot_curve(results_object,
               loess=True,
               specs=None,
               ax=None,
               colormap='Spectral_r',
               title=''):
    """
    Plots the curve of median, confidence intervals, minimum, and maximum
    coefficient estimates for a given results object.

    Parameters:
        results_object (object): Object containing the results data.
        loess (bool, optional): Whether to apply LOESS smoothing to the curve. Default is True.
        specs (list, optional): List of specification names to be highlighted.
        ax (matplotlib.axes._subplots.AxesSubplot, optional): Axes to plot on.
        colormap (str, optional): Colormap to use for highlighting specifications.

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
                      color=get_colormap_colors(colormap, 100)[0],
                      linestyle='-')

    if loess:  # Check if LOESS should be applied
        loess_min = sm.nonparametric.lowess(df['min'],
                                            pd.to_numeric(df.index),
                                            frac=0.05)
        loess_max = sm.nonparametric.lowess(df['max'],
                                            pd.to_numeric(df.index),
                                            frac=0.05)
        ax.plot(loess_min[:, 0], loess_min[:, 1], color=get_colormap_colors(colormap, 100)[99], linestyle='--')
        ax.plot(loess_max[:, 0], loess_max[:, 1], color=get_colormap_colors(colormap, 100)[99], linestyle='--')
        ax.fill_between(df.index,
                        loess_min[:, 1],
                        loess_max[:, 1],
                        facecolor='#fee08b',
                        alpha=0.15)
    else:
        # Plot raw min and max as dashed lines similar to loess
        ax.plot(df.index, df['min'], color=get_colormap_colors(colormap, 100)[99], linestyle='--', label='Min')
        ax.plot(df.index, df['max'], color=get_colormap_colors(colormap, 100)[99], linestyle='--', label='Max')
        ax.fill_between(df.index,
                        df['min'],
                        df['max'],
                        facecolor='#fee08b',
                        alpha=0.15)

    if ax.get_ylim()[0] < 0 and ax.get_ylim()[1] > 0:
        ax.axhline(y=0, color='k', ls='--')

    lines = []
    markers = []
    if specs:
        idxs = df.index[df['idx']].tolist()
        for idx, i in zip(idxs, range(len(specs))):
            control_names = list(df.spec_name.iloc[idx])
            label = ', '.join(control_names)
            lines.append(ax.vlines(x=idx,
                                   ymin=df['min'].iloc[idx] if not loess else loess_min[idx, 1],
                                   ymax=df['max'].iloc[idx] if not loess else loess_max[idx, 1],
                                   color=colorset[i],
                                   label=label))
            markers.append(Line2D([0], [0],
                                  marker='o',
                                  color=colorset[i],
                                  markerfacecolor='w',
                                  markersize=10,
                                  label=label)
                           )
            myArrow = FancyArrowPatch(posA=(idx, df['min'].iloc[idx] if not loess else loess_min[idx, 1]),
                                      posB=(idx, df['max'].iloc[idx] if not loess else loess_max[idx, 1]),
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
                           ymin=df['min'].iloc[full_spec_pos] if not loess else loess_min[full_spec_pos, 1],
                           ymax=df['max'].iloc[full_spec_pos] if not loess else loess_max[full_spec_pos, 1],
                           color='k',
                           label='Full Model'))
    markers.append(Line2D([0], [0],
                          marker='o',
                          color='k',
                          markerfacecolor='w',
                          markersize=10,
                          label='Full Model')
                   )
    myArrow = FancyArrowPatch(
        posA=(full_spec_pos, df['min'].iloc[full_spec_pos] if not loess else loess_min[full_spec_pos, 1]),
        posB=(full_spec_pos, df['max'].iloc[full_spec_pos] if not loess else loess_max[full_spec_pos, 1]),
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

    ax.legend(handles=markers,
              frameon=True,
              edgecolor=(0, 0, 0, 1),
              fontsize=11,
              loc="lower right",
              ncols=2,
              framealpha=1,
              facecolor=((1, 1, 1, 0)
              )
              )

    axis_formatter(ax, r'Coefficient Estimates', 'Ordered Specifications', title)
    ax.set_xlim(0, len(results_object.specs_names))
    ax.set_ylim(ax.get_ylim()[0] - (np.abs(ax.get_ylim()[1]) - np.abs(ax.get_ylim()[0])) / 10,
                ax.get_ylim()[1] + (np.abs(ax.get_ylim()[1]) - np.abs(ax.get_ylim()[0])) / 10)
    final_string = f"Median coef: {results_object.inference['median']:.3f} (Z: {results_object.inference['Stouffers'][0]:.3f})"
    ax.text(
        0.05, 0.95,
        (f'Number of specifications: {len(results_object.specs_names)}\n' +
         f'Number of bootstraps: {results_object.draws}\n' +
         f'Number of folds: {results_object.kfold}\n' +
         final_string
         ),
        transform=ax.transAxes,
        verticalalignment='top',
        horizontalalignment='left',
        color='black',
        fontsize=12,
        bbox=dict(facecolor='white',
                  edgecolor='black',
                  boxstyle='round,pad=1')
    )
    sns.despine(ax=ax)
    return ax

def plot_ic(results_object,
            ic,
            specs=None,
            ax=None,
            colormap='Spectral_r',
            title='',
            despine_left=True):
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
    axis_formatter(ax, f'{ic.upper()} curve', 'Ordered Specifications', title)

    if specs:
        key = get_selection_key(specs)
        full_spec = list(results_object.specs_names.iloc[-1])
        full_spec_key = get_selection_key([full_spec])
        df['idx'] = df.spec_name.isin(key)
        df['full_spec_idx'] = df.spec_name.isin(full_spec_key)
        df = df.sort_values(by=ic).reset_index(drop=True)
        ax.plot(df[ic], color='#002d87')
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
        if despine_left is True:
            sns.despine(ax=ax, right=False, left=True)
            ax.yaxis.set_label_position("right")
        else:
            ax.yaxis.set_label_position("left")
            sns.despine(ax=ax)
    else:
        df = df.sort_values(by=ic).reset_index(drop=True)
        ax.plot(df[ic], color='#002d87')
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
        if despine_left is True:
            ax.yaxis.set_label_position("right")
            sns.despine(ax=ax, right=False, left=True)
        else:
            ax.yaxis.set_label_position("left")
            sns.despine(ax=ax)


def plot_bdist(results_object,
               specs=None,
               ax=None,
               colormap='Spectral_r',
               title='',
               despine_left=True,
               legend_bool=False
               ):
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

    def make_leg(results_object):
        df = results_object.summary_df.copy()
        df = df.sort_values(by='median')
        df = df.reset_index(drop=True)
        key = get_selection_key(specs)
        full_spec = list(results_object.specs_names.iloc[-1])
        full_spec_key = get_selection_key([full_spec])
        df['idx'] = df.spec_name.isin(key)
        df['full_spec_idx'] = df.spec_name.isin(full_spec_key)
        idxs = df.index[df['idx']].tolist()
        leg = []
        for idx, i in zip(idxs, range(len(specs))):
            control_names = list(df.spec_name.iloc[idx])
            label = ', '.join(control_names).title()
            leg.append([colorset[i], label])
        leg.append(['k', 'Full Model'])
        return leg

    colorset = get_colormap_colors(colormap, len(specs))
    if ax is None:
        ax = plt.gca()
    df = results_object.estimates.copy().T
    df.columns = results_object.specs_names
    if specs:
        key = get_selection_key(specs)
        matching_cols = [col for col in df.columns if col in key]
        for i, col in enumerate(matching_cols):
            sns.kdeplot(df[col].squeeze(), common_norm=True, ax=ax, color=colorset[i], legend=False)
    sns.kdeplot(df.iloc[:, -1:].squeeze(), common_norm=True, legend=False,
                color='black', ax=ax, label='Full Model')
    markers=[]
    if legend_bool:
        leg = make_leg(results_object)
        for ele in leg:
            markers.append(Line2D([0], [0],
                                  marker='o',
                                  color=ele[0],
                                  markerfacecolor='w',
                                  markersize=0,
                                  label=ele[1])
                           )
        # Temporarily place the legend at "upper left" to check for overlap
        temp_legend = ax.legend(handles=markers,
                                frameon=True,
                                edgecolor=(0, 0, 0, 1),
                                fontsize=9,
                                loc="upper left",  # Start with "upper left"
                                ncols=1,
                                framealpha=1,
                                facecolor=((1, 1, 1, 0)
                                )
                                )

        # Get the bounding box of the plot and the legend
        plot_bbox = ax.get_tightbbox(plt.gcf().canvas.get_renderer())
        legend_bbox = temp_legend.get_window_extent(plt.gcf().canvas.get_renderer())

        # Check if the legend overlaps with the plot area
        if legend_bbox.overlaps(plot_bbox):
            # Remove the temporary legend
            temp_legend.remove()

            # Place the legend in the upper right to avoid overlap
            ax.legend(handles=markers,
                      frameon=True,
                      edgecolor=(0, 0, 0, 1),
                      fontsize=9,
                      loc="upper right",  # Move to "upper right" if overlapping
                      ncols=1,
                      framealpha=1,
                      facecolor=((1, 1, 1, 0)
                      )
                      )
    axis_formatter(ax, 'Density', 'Coefficient Estimate', title)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(5))
    if despine_left is True:
        ax.yaxis.set_label_position("right")
        sns.despine(ax=ax, right=False, left=True)
    else:
        sns.despine(ax=ax)
    return ax

def plot_kfolds(results_object,
                colormap,
                ax=None,
                title='',
                despine_left=True
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
    legend_elements = [
        Line2D([0], [0], color=get_colormap_colors(colormap, 100)[99], lw=2, linestyle='-',
                   label=r'Density', alpha=1),
        Patch(facecolor=get_colormap_colors(colormap, 100)[0], edgecolor=(0, 0, 0, 1),
              label=r'Histogram')]
    ax.legend(handles=legend_elements,
              loc='upper left',
              frameon=True,
              fontsize=9,
              #title='Out-of-Sample',
              title_fontsize=8,
              framealpha=1,
              facecolor='w',
              edgecolor=(0, 0, 0, 1),
              ncols=1
              )
    ax.set_xlim(ax.get_xlim()[0] - (np.abs(ax.get_xlim()[1]) - np.abs(ax.get_xlim()[0])) / 15,
                ax.get_xlim()[1])
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.grid(linestyle='--', color='k', alpha=0.1, zorder=-1)
    ax.set_axisbelow(True)
    if results_object.name_av_k_metric=='rmse':
        metric = results_object.name_av_k_metric.upper()
    elif results_object.name_av_k_metric.upper=='R-SQUARED':
        metric = r'R$^2'
    else:
        metric = results_object.name_av_k_metric.title()
    axis_formatter(ax, 'Density', r'OOS Metric: ' + metric, title)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(5))
    if despine_left is True:
        ax.yaxis.set_label_position("right")
        sns.despine(ax=ax, right=False, left=True)
    else:
        sns.despine(ax=ax)


def plot_bma(results_object, colormap, ax, feature_order, title=''):
    """
    Plots the Bayesian Model Averaging (BMA)
    """
    bma = results_object.compute_bma()
    bma = bma.set_index('control_var')
    bma = bma.reindex(feature_order)
    bma['probs'].plot(kind='barh',
                      ax=ax,
                      alpha=1,
                      color=get_colormap_colors(colormap, 100)[0],
                      edgecolor='k',
                      )
    axis_formatter(ax, r'', 'BMA Probabilities', title)
    sns.despine(ax=ax)


def title_setter(ax, title):
    return ax.set_title(title, loc='left', fontsize=16, y=1)


def plot_results(results_object,
                 loess=True,
                 specs=None,
                 ic=None,
                 colormap='Spectral_r',
                 figsize=(16, 16),
                 ext='pdf',
                 project_name='no_project_name'
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

    figpath = os.path.join(os.getcwd(), 'figures', project_name)
    if not os.path.exists(figpath):
        os.makedirs(figpath)

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(9, 24, wspace=0.5, hspace=1.5)
    ax1 = fig.add_subplot(gs[0:3, 0:12])
    ax2 = fig.add_subplot(gs[0:3, 13:24])
    ax4 = fig.add_subplot(gs[3:5, 6:14])
    ax3 = fig.add_subplot(gs[3:5, 0:6])
    ax5 = fig.add_subplot(gs[3:5, 14:23])
    ax6 = fig.add_subplot(gs[5:9, 0:13])
    ax7 = fig.add_subplot(gs[5:7, 13:23])
    ax8 = fig.add_subplot(gs[7:9, 13:23])

    shap_vals = np.delete(results_object.shap_return[0], 0, axis=1)
    shap_x = results_object.shap_return[1].drop(results_object.x_name, axis=1).to_numpy()
    shap_cols = results_object.shap_return[1].drop(results_object.x_name, axis=1).columns


    plot_hexbin_r2(results_object, ax1, fig, colormap, title='a.')
    plot_hexbin_log(results_object, ax2, fig, colormap, title='b.')
    feature_order = shap_violin(ax4, shap_vals, shap_x, shap_cols, title='d.', clear_yticklabels=True)
    plot_bma(results_object, colormap, ax3, feature_order, title='c.')
    plot_kfolds(results_object, colormap, ax5, title='e.', despine_left=True)
    plot_curve(results_object=results_object, loess=loess, specs=specs, ax=ax6, colormap=colormap, title='f.')
    plot_ic(results_object=results_object, ic=ic, specs=specs, ax=ax7, colormap=colormap, title='g.', despine_left=True)
    plot_bdist(results_object=results_object, specs=specs, ax=ax8, colormap=colormap, title='h.', despine_left=True)
    plt.savefig(os.path.join(figpath, project_name + '_all.'+ext), bbox_inches='tight')


    fig, ax = plt.subplots(figsize=(8.5, 5))
    plot_hexbin_r2(results_object, ax, fig, colormap)
    plt.savefig(os.path.join(figpath, project_name + '_R2hexbin.'+ext), bbox_inches='tight')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    plot_hexbin_log(results_object, ax, fig, colormap)
    plt.savefig(os.path.join(figpath, project_name + '_LLhexbin.'+ext), bbox_inches='tight')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    feature_order = shap_violin(ax, shap_vals, shap_x, shap_cols, clear_yticklabels=False)
    plt.savefig(os.path.join(figpath, project_name + '_SHAP.'+ext), bbox_inches='tight')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    plot_bma(results_object, colormap, ax, feature_order)
    plt.savefig(os.path.join(figpath, project_name + '_BMA.'+ext), bbox_inches='tight')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    plot_kfolds(results_object, colormap, ax, despine_left=False)
    plt.savefig(os.path.join(figpath, project_name + '_OOS.'+ext), bbox_inches='tight')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 7))
    plot_curve(results_object=results_object, specs=specs, ax=ax, colormap=colormap)
    plt.savefig(os.path.join(figpath, project_name + '_curve.'+ext), bbox_inches='tight')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    plot_ic(results_object=results_object, ic=ic, specs=specs, ax=ax, colormap=colormap, title='g.', despine_left=False)
    plt.savefig(os.path.join(figpath, project_name + '_IC.'+ext), bbox_inches='tight')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    plot_bdist(results_object=results_object, specs=specs, ax=ax, colormap=colormap, despine_left=False, legend_bool=True)
    plt.savefig(os.path.join(figpath, project_name + '_bdist.'+ext), bbox_inches='tight')
    plt.close(fig)
