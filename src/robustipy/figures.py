from __future__ import annotations

import os
import warnings
from typing import Union, Optional, List, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import gaussian_kde

from shap import Explanation
from shap.plots._labels import labels

from robustipy.utils import get_selection_key, get_colormap_colors

plt.rcParams['axes.unicode_minus'] = False


def _legend_side_from_hist(ax, *, tau: float = 0.6) -> str:
    """
    Decide whether 'upper left' or 'upper right' is safer, given the
    rectangular patches in *ax* produced by seaborn.histplot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes containing seaborn.histplot patches.
    tau : float, default=0.6
        Safety threshold in (0,1). A bar exceeding `tau * ylim_max` on one side
        forces the legend to the opposite.

    Returns
    -------
    str
        Either 'upper left' or 'upper right'.
    """
    bars = [p for p in ax.patches if isinstance(p, Rectangle) and p.get_height() > 0]
    if not bars:  # fall-back if no histogram rendered
        return 'upper left'
    # Split bars at the sample median
    median_x = np.median([p.get_x() + p.get_width() / 2 for p in bars])
    left_max = max((p.get_height() for p in bars if (p.get_x() + p.get_width() / 2) < median_x), default=0.0)
    right_max = max((p.get_height() for p in bars if (p.get_x() + p.get_width() / 2) >= median_x), default=0.0)
    f_max = max(left_max, right_max)
    ylim_top = ax.get_ylim()[1]

    # ‘Legend altitude’ test
    left_hits = left_max > tau * ylim_top
    right_hits = right_max > tau * ylim_top

    if left_hits and (left_max >= right_max):
        return 'upper right'
    if right_hits and (right_max > left_max):
        return 'upper left'
    return 'upper left'  # default / tie


def axis_formatter(
        ax: plt.Axes,
        ylabel: str,
        xlabel: str,
        title: str,
        side: str = 'left'
) -> None:
    """
    Apply consistent styling to a Matplotlib Axes: grids, fonts, labels, and title placement.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to format.
    ylabel : str
        Label text for the y-axis.
    xlabel : str
        Label text for the x-axis.
    title : str
        Title text for the plot.
    side : {'left', 'right'}, default='left'
        Side on which to draw the y-axis label and title.
        - 'left': y-label on left, title aligned slightly left.
        - 'right': y-label on right, title aligned to the right side of the axes.

    Returns
    -------
    None
        This function modifies `ax` in place and does not return anything.
    """
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.grid(linestyle='--', color='k', alpha=0.1, zorder=-1)
    ax.set_axisbelow(True)
    ax.set_ylabel(ylabel, fontsize=13)
    title_setter(ax, title, side)
    ax.set_xlabel(xlabel, fontsize=13)


def plot_hexbin_r2(
        results_object,
        ax: plt.Axes,
        fig: plt.Figure,
        oddsratio: bool,
        colormap: Union[str, cm.Colormap],
        title: str = "",
        side: str = "left",
) -> None:
    """
    Hex-bin density plot of boot-strapped coefficient estimates versus in-sample
    :math:`R^2`, together with a marginal colour-bar of observation counts.

    Parameters
    ----------
    results_object : Any
        Must expose ``results_object.estimates`` and ``results_object.r2_values``,
        each supporting ``.stack()`` to obtain 1-d views.
    ax : matplotlib.axes.Axes
        Target axes.
    fig : matplotlib.figure.Figure
        Parent figure, needed for colour-bar geometry.
    colormap : str | matplotlib.colors.Colormap
        Matplotlib-compatible colormap.
    title : str, optional
        Axes title.
    side : {'left', 'right'}, optional
        * ``'left'``  – conventional layout: y-axis on the left, colour-bar on the
          right.
        * ``'right'`` – mirror layout: y-axis (ticks, label, spine) on the right,
          colour-bar on the left; the left spine is removed.

    Returns
    -------
    None
        Draws in place on `ax`.

    Raises
    ------
    ValueError
        If `side` is not 'left' or 'right'.

    Notes
    -----
    Only the presentation layer is mirrored; the data are not transformed.
    """
    # ------------------------------------------------------------------ #
    # 1.  Hex-bin and colour-bar                                         #
    # ------------------------------------------------------------------ #
    if oddsratio is True:
        image = ax.hexbin(
            results_object.estimates_exp.stack(),
            results_object.r2_values.stack(),
            cmap=colormap,
            gridsize=20,
            mincnt=1,
            edgecolor="k",
        )
    else:
        image = ax.hexbin(
            results_object.estimates.stack(),
            results_object.r2_values.stack(),
            cmap=colormap,
            gridsize=20,
            mincnt=1,
            edgecolor="k",
        )
    # Place the colour-bar opposite the y-axis
    cb_location = "right" if side == "left" else "left"
    cb = fig.colorbar(
        image,
        ax=ax,
        spacing="uniform",
        pad=0.05,
        extend="max",
        location=cb_location,  # Matplotlib ≥ 3.3
    )

    # ------------------------------------------------------------------ #
    # 2.  Colour-bar tick formatting                                     #
    # ------------------------------------------------------------------ #
    data = image.get_array()
    ticks = np.linspace(data.min(), data.max(), num=6)
    cb.set_ticks(ticks)

    if 1_000 <= data.max() < 10_000:
        cb.set_ticklabels([f"{t / 1_000:.1f}k" for t in ticks])
    elif data.max() >= 10_000:
        cb.set_ticklabels([f"{t / 1_000:.0f}k" for t in ticks])
    else:
        cb.set_ticklabels([f"{t:.0f}" for t in ticks])

    cb.ax.set_title('Count')

    # ------------------------------------------------------------------ #
    # 2a.  Optional trimming & baseline alignment (only when on the left)#
    # ------------------------------------------------------------------ #
    if cb_location == "left":  # i.e. side == "right"
        frac = 0.1  # fraction to trim (from the top)
        ax_box = ax.get_position()  # main axes box (for baseline)
        cb_box = cb.ax.get_position()

        new_height = cb_box.height * (1 - frac)
        cb.ax.set_position([
            cb_box.x0,  # keep x-position
            ax_box.y0,  # align bottom with axes baseline
            cb_box.width,
            new_height  # shorten from the top only
        ])

    # ------------------------------------------------------------------ #
    # 3.  Axes labels, title, tick locators                              #
    # ------------------------------------------------------------------ #
    #
    if results_object.model_name=='Logistic Regression Robust':
        axis_formatter(
            ax,
            r"R$_{\mathrm{McF}}^2$",
            r"Bootstrapped Estimand",
            title,
            side,
        )
    else:
        axis_formatter(
            ax,
            r"In-Sample $\bar{\mathrm{R}}^2$",
            r"Bootstrapped Estimand",
            title,
            side,
        )

    ax.xaxis.set_major_locator(mticker.MaxNLocator(4))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(4))

    # ------------------------------------------------------------------ #
    # 4.  Side-dependent spines, ticks, labels                           #
    # ------------------------------------------------------------------ #
    if side == "right":
        # shift ticks and label to the right
        ax.yaxis.set_ticks_position("right")
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")

        # ensure both marks *and* labels appear on the right
        ax.tick_params(axis="y",
                       which="both",
                       right=True, labelright=True,
                       left=False, labelleft=False)

        # keep right spine, hide left spine
        ax.spines["right"].set_visible(True)
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(True)

    elif side == "left":
        ax.yaxis.set_ticks_position("left")
        ax.yaxis.tick_left()
        ax.yaxis.set_label_position("left")

        ax.tick_params(axis="y",
                       which="both",
                       right=False, labelright=False,
                       left=True, labelleft=True)

        ax.spines["left"].set_visible(True)
        ax.spines["right"].set_visible(False)
        sns.despine(ax=ax, left=False, right=True)

    else:
        raise ValueError("`side` must be 'left' or 'right'.")

    # ------------------------------------------------------------------ #
    # 5.  Void function – all graphics modified in-place                 #
    # ------------------------------------------------------------------ #
    return None


def plot_hexbin_log(
        results_object,
        ax: plt.Axes,
        fig: plt.Figure,
        oddsratio: bool,
        colormap: Union[str, cm.Colormap],
        title: str = ''
) -> None:
    """
    Plot a hex-bin density of full-sample coefficient estimates vs. log-likelihood.

    Parameters
    ----------
    results_object : object
        Must expose:
          - `all_b`/all_b_exp: list/array of full-sample coefficient arrays
          - `summary_df['ll']`: corresponding log-likelihood values
    ax : matplotlib.axes.Axes
        The axes on which to draw the hex-bin.
    fig : matplotlib.figure.Figure
        Parent figure (needed to place the colorbar).
    colormap : str or Colormap
        Name or object of a Matplotlib colormap.
    title : str, optional
        Title displayed above the plot (default: '').

    Returns
    -------
    None
    """
    if oddsratio is True:
        image = ax.hexbin(results_object.all_b_exp,
                          results_object.summary_df['ll'],
                          cmap=colormap,
                          gridsize=20,
                          mincnt=1,
                          edgecolor='k'
                          )
    else:
        image = ax.hexbin([arr[0][0] for arr in results_object.all_b.copy()],
                          results_object.summary_df['ll'],
                          cmap=colormap,
                          gridsize=20,
                          mincnt=1,
                          edgecolor='k'
                          )

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
    axis_formatter(ax, r'Full Model Log Likelihood', r'Full-Sample Estimand', title)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(4))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(4))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
    sns.despine(ax=ax)


def shap_violin(
        ax: plt.Axes,
        shap_values: Union[np.ndarray, List[np.ndarray], Explanation],
        features: Optional[Union[np.ndarray, pd.DataFrame, List[str]]] = None,
        feature_names: Optional[List[str]] = None,
        max_display: int = 10,
        color: Optional[Union[str, Sequence]] = None,
        alpha: float = 1.0,
        cmap: str = 'Spectral_r',
        use_log_scale: bool = False,
        title: str = '',
        clear_yticklabels: bool = False
) -> List[str]:
    """Create a SHAP beeswarm plot, colored by feature values when they are provided.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes on which to draw the plot.
    shap_values : array-like or Explanation
        SHAP value matrix (#samples×#features), or a list thereof for multiclass,
        or a SHAP Explanation object.
    features : array-like, DataFrame, or list of str, optional
        Feature value matrix (#samples×#features), or just a feature_names list.
        Default: None (no coloring).
    feature_names : list of str, optional
        Names of each feature. Default: None (will infer or auto‐label).
    max_display : int, default=10
        Maximum number of top features (by mean(|SHAP|)) to show.
    color : str or sequence, optional
        Single color for all points when no feature values given.
    alpha : float, default=1.0
        Opacity for scatter points.
    cmap : str, default='Spectral_r'
        Colormap for coloring points.
    use_log_scale : bool, default=False
        If True, use symlog x-axis scaling.
    title : str, optional
        Title text for the axes.
    clear_yticklabels : bool, default=False
        If True, hide the y-tick labels.

    Returns
    -------
    List[str]
        Ordered list of feature names actually plotted.
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
        if shap_values.shape[1] == features.shape[1] + 1:
            shap_values = shap_values[:, :-1]   # drop bias
            num_features -= 1
        elif shap_values.shape[1] != features.shape[1]:
            raise ValueError(
                f"shap_values has {shap_values.shape[1]} columns but "
                f"features has {features.shape[1]} – shapes don’t match."
            )

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
                rasterized=True
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
                rasterized=True,
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
            rasterized=True
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


def plot_curve(
        results_object,
        loess: bool = True,
        ci: float = 1,
        oddsratio: bool = False,
        specs: Optional[List[List[str]]] = None,
        ax: Optional[plt.Axes] = None,
        highlights: bool = True,
        inset: bool = True,
        title: str = ''
) -> plt.Axes:
    """
    Plot the specification-curve of median and CI for coefficient estimates.

    Parameters
    ----------
    results_object : object
        Must expose `.summary_df` (with columns 'median'), `.specs_names`,
        `.estimates` (DataFrame of bootstrap draws), `.draws`, `.kfold`,
        and `.inference` dict.
    loess : bool, default=True
        Whether to smooth the lower/upper CI bounds with LOESS.
    ci : float, default=1
        The confidence-level (e.g. 0.95 for a 95% interval).
    specs : list of control-lists, optional
        Up to three specs to highlight. Default: None (no highlights).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.  Default: current axes.
    colormap : str, default='Spectral_r'
        Colormap for the main curve.
    title : str, optional
        Title text for the axes.
    highlights : bool, default=False
        If True, highlights the full model and the null model in the plot.
    inset  : bool, default=True
        If True, adds an inset with the full model and null model highlights.
    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.
    """
    # Prepare axes
    if ax is None:
        ax = plt.gca()

    # Compute quantiles and annotate DataFrame
    df = pd.DataFrame()
    alpha = 1 - ci
    if oddsratio is True:
        qs = results_object.estimates_exp.quantile(q=[alpha / 2, 1 - alpha / 2],
                                                   axis=1,
                                                   interpolation='nearest'
                                                   )
        df['median'] = results_object.estimates_exp.quantile(q=0.5, axis=1)
    else:
        qs = results_object.estimates.quantile(q=[alpha / 2, 1 - alpha / 2],
                                               axis=1,
                                               interpolation='nearest'
                                               )
        df['median'] = results_object.estimates.quantile(q=0.5, axis=1)
    df['q_low'] = qs.iloc[0].values
    df['q_high'] = qs.iloc[1].values

    # Flag specs
    if highlights:
        full_spec, null_spec = (list(results_object.specs_names.iloc[-1]),
                                list(results_object.specs_names.iloc[0]))
        df['full_spec_idx'] = results_object.specs_names.isin(get_selection_key([full_spec]))
        df['null_spec_idx'] = results_object.specs_names.isin(get_selection_key([null_spec]))
    if specs:
        df['idx'] = results_object.specs_names.isin(get_selection_key(specs))
        df['specs_names'] = results_object.specs_names

    # Sort by median
    df = df.sort_values(by='median').reset_index(drop=True)
    n = len(df)

    # Sample colours: first for null, next for highlights, last for full
    n_hl = len(specs) if specs else 0
    # colourset = get_colormap_colors(colormap, n_hl + 2)
    colourset = get_colormap_colors(n_hl + 2)
    null_color = colourset[0]
    spec_colors = colourset[1:-1]
    full_color = colourset[-1]

    # Plot median
    # median_color = get_colormap_colors(colormap, 100)[0]
    median_color = 'k'
    df['median'].plot(ax=ax, color=median_color, linestyle='-')

    # Plot CI bounds
    # hi_color = get_colormap_colors(colormap, 100)[-1]
    hi_color = 'gray'
    if loess:
        frac = max(2 / n, 0.3)
        lo_low = sm.nonparametric.lowess(df['q_low'], df.index, frac=frac)
        lo_high = sm.nonparametric.lowess(df['q_high'], df.index, frac=frac)
        ax.plot(lo_low[:, 0], lo_low[:, 1], color=hi_color, linestyle='--')
        ax.plot(lo_high[:, 0], lo_high[:, 1], color=hi_color, linestyle='--')
        ax.fill_between(df.index, lo_low[:, 1], lo_high[:, 1], facecolor='#fee08b', alpha=0.15)
    else:
        ax.plot(df.index, df['q_low'], color=hi_color, linestyle='--', label='Lower CI')
        ax.plot(df.index, df['q_high'], color=hi_color, linestyle='--', label='Upper CI')
        ax.fill_between(df.index, df['q_low'], df['q_high'], facecolor='#fee08b', alpha=0.15)

    # Zero line
    y0, y1 = ax.get_ylim()
    if y0 < 0 < y1:
        ax.axhline(0, color='k', ls='--')

    # Highlights
    handles = []
    if specs:
        idxs = df.index[df['idx']].tolist()
        for j, idx in enumerate(idxs):
            col = spec_colors[j]
            lbl = ', '.join(df['specs_names'].iloc[idx])
            low = lo_low[idx, 1] if loess else df.at[idx, 'q_low']
            high = lo_high[idx, 1] if loess else df.at[idx, 'q_high']
            ax.vlines(idx, ymin=low, ymax=high, color=col)
            arrow = FancyArrowPatch((idx, low), (idx, high), arrowstyle='<|-|>', color=col,
                                    mutation_scale=20, shrinkA=0, shrinkB=0)
            ax.add_artist(arrow)
            ax.plot(idx, df.at[idx, 'median'], 'o', markeredgecolor=col, markerfacecolor='w', markersize=12)
            handles.append(Line2D([0], [0], marker='o', color=col, markerfacecolor='w', markersize=10, label=lbl))

    # Full model
    if highlights:
        pos_full = df.index[df['full_spec_idx']].item()
        low_f = lo_low[pos_full, 1] if loess else df.at[pos_full, 'q_low']
        high_f = lo_high[pos_full, 1] if loess else df.at[pos_full, 'q_high']
        ax.vlines(pos_full, ymin=low_f, ymax=high_f, color=full_color)
        arrow_f = FancyArrowPatch((pos_full, low_f), (pos_full, high_f), arrowstyle='<|-|>', color=full_color,
                                  mutation_scale=20, shrinkA=0, shrinkB=0)
        ax.add_artist(arrow_f)
        ax.plot(pos_full, df.at[pos_full, 'median'], 'o', markeredgecolor=full_color, markerfacecolor='w', markersize=12)
        if max(len(t) for t in results_object.y_name) == 1:
            handles.append(
                Line2D([0], [0], marker='o', color=full_color, markerfacecolor='w', markersize=10, label='Full Model'))
        else:
            handles.append(
                Line2D([0], [0], marker='o', color=full_color, markerfacecolor='w', markersize=10, label='All Data Used'))
    if highlights:
        # Null model
        pos_null = df.index[df['null_spec_idx']].item()
        low_n = lo_low[pos_null, 1] if loess else df.at[pos_null, 'q_low']
        high_n = lo_high[pos_null, 1] if loess else df.at[pos_null, 'q_high']
        ax.vlines(pos_null, ymin=low_n, ymax=high_n, color=null_color)
        arrow_n = FancyArrowPatch((pos_null, low_n), (pos_null, high_n), arrowstyle='<|-|>', color=null_color,
                                  mutation_scale=20, shrinkA=0, shrinkB=0)
        ax.add_artist(arrow_n)
        ax.plot(pos_null, df.at[pos_null, 'median'], 'o', markeredgecolor=null_color, markerfacecolor='w', markersize=12)
        if max(len(t) for t in results_object.y_name) == 1:
            handles.append(
                Line2D([0], [0], marker='o', color=null_color, markerfacecolor='w', markersize=10, label='No Controls'))
        else:
            handles.append(
                Line2D([0], [0], marker='o', color=null_color, markerfacecolor='w', markersize=10, label=r'First y Only'))
        # Legend and formatting
    if highlights and (specs is not None):
        ax.legend(handles=handles, frameon=True, edgecolor='black', fontsize=11,
                  loc='lower right', ncols=2, framealpha=1, facecolor='w')
    axis_formatter(ax, r'Estimand of Interest', 'Ordered Specifications', title)
    ax.set_xlim(0, n - 1)
    pad = (y1 - y0) * 0.1
    ax.set_ylim(y0 - pad, y1 + pad)

    # Summary inset
    if inset:
        median_inf = results_object.inference['median']
        z_score = results_object.inference['Stouffers'][0]
        info_text = (
            f'Specifications: {n}\n'
            f'Bootstraps: {results_object.draws}\n'
            f'Folds: {results_object.kfold}\n'
            f'Median: {median_inf:.3f} (Z: {z_score:.3f})'
        )
        ax.text(0.05, 0.95, info_text, transform=ax.transAxes, va='top', ha='left',
                fontsize=12, color='black', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=1'))

    sns.despine(ax=ax)
    return ax


def plot_ic(
        results_object,
        ic: str,
        specs: Optional[List[List[str]]] = None,
        ax: Optional[plt.Axes] = None,
        colormap: str = 'Spectral_r',
        title: str = '',
        despine_left: bool = True
) -> plt.Axes:
    """
    Plots the information criterion (IC) curve, colouring:
      • “No Controls” in the first colormap colour
      • each user‐highlighted spec in the next colours
      • “Full Model” in the last colormap colour
    """
    # Validate IC column
    if ic not in results_object.summary_df.columns:
        available_ics = [c for c in results_object.summary_df.columns
                         if c.lower() in {'aic', 'bic', 'hqic'}]
        raise ValueError(f"[plot_ic] '{ic}' not found. Available: {available_ics}")

    if ax is None:
        ax = plt.gca()

    # grab exactly len(specs)+2 colours
    n_specs = len(specs) if specs else 0
    # colorset = get_colormap_colors(colormap, n_specs + 2)
    colorset = get_colormap_colors(n_specs + 2)

    df = results_object.summary_df.copy()
    df = df.sort_values(by=ic).reset_index(drop=True)
    axis_formatter(ax, f'{ic.upper()} curve', 'Ordered Specifications', title)

    if specs:
        key = get_selection_key(specs)
        full_spec_key = get_selection_key([list(results_object.specs_names.iloc[-1])])
        null_spec_key = get_selection_key([list(results_object.specs_names.iloc[0])])
        df['idx'] = df.spec_name.isin(key)
        df['full_spec_idx'] = df.spec_name.isin(full_spec_key)
        df['null_spec_idx'] = df.spec_name.isin(null_spec_key)
        df = df.sort_values(by=ic).reset_index(drop=True)

        # main IC curve stays the same colour you had
        ax.plot(df[ic], color='k')

        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax)

        lines, markers = [], []
        # highlighted specs
        idxs = df.index[df['idx']].tolist()
        for idx, i in zip(idxs, range(n_specs)):
            col = colorset[i + 1]  # shift by one
            label = ', '.join(df.spec_name.iloc[idx])
            lines.append(ax.vlines(idx, ymin, df.at[idx, ic], color=col, label=label))
            markers.append(Line2D([0], [0], marker='o', color=col,
                                  markerfacecolor='w', markersize=10, label=label))
            ax.plot(idx, df.at[idx, ic], 'o',
                    markeredgecolor=col, markerfacecolor='w', markersize=15)

        # full‐model in last colour
        full_pos = df.index[df['full_spec_idx']].item()
        col_f = colorset[-1]
        lines.append(ax.vlines(full_pos, ymin, df.at[full_pos, ic], color=col_f, label='Full Model'))
        markers.append(Line2D([0], [0], marker='o', color=col_f,
                              markerfacecolor='w', markersize=10, label='Full Model'))
        ax.plot(full_pos, df.at[full_pos, ic], 'o',
                markeredgecolor=col_f, markerfacecolor='w', markersize=15)

        # no‐controls in first colour
        null_pos = df.index[df['null_spec_idx']].item()
        col_n = colorset[0]
        lines.append(ax.vlines(null_pos, ymin, df.at[null_pos, ic], color=col_n, label='No Controls'))
        markers.append(Line2D([0], [0], marker='o', color=col_n,
                              markerfacecolor='w', markersize=10, label='No Controls'))
        ax.plot(null_pos, df.at[null_pos, ic], 'o',
                markeredgecolor=col_n, markerfacecolor='w', markersize=15)

        ax.legend(handles=markers,
                  frameon=True, edgecolor='black',
                  fontsize=9, loc="upper left",
                  ncols=1, framealpha=1, facecolor='w')

        if despine_left:
            sns.despine(ax=ax, right=False, left=True)
            ax.yaxis.set_label_position("right")
        else:
            sns.despine(ax=ax)

    else:
        df['full_spec_idx'] = df.spec_name.isin(get_selection_key([list(results_object.specs_names.iloc[-1])]))
        df['null_spec_idx'] = df.spec_name.isin(get_selection_key([list(results_object.specs_names.iloc[0])]))
        df = df.sort_values(by=ic).reset_index(drop=True)

        ax.plot(df[ic], color='k')
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax)

        lines, markers = [], []
        # full model
        full_pos = df.index[df['full_spec_idx']].item()
        col_f = colorset[-1]
        lines.append(ax.vlines(full_pos, ymin, df.at[full_pos, ic], color=col_f, label='Full Model'))
        markers.append(Line2D([0], [0], marker='o', color=col_f,
                              markerfacecolor='w', markersize=10, label='Full Model'))
        ax.plot(full_pos, df.at[full_pos, ic], 'o',
                markeredgecolor=col_f, markerfacecolor='w', markersize=15)

        # no controls
        null_pos = df.index[df['null_spec_idx']].item()
        col_n = colorset[0]
        lines.append(ax.vlines(null_pos, ymin, df.at[null_pos, ic], color=col_n, label='No Controls'))
        markers.append(Line2D([0], [0], marker='o', color=col_n,
                              markerfacecolor='w', markersize=10, label='No Controls'))
        ax.plot(null_pos, df.at[null_pos, ic], 'o',
                markeredgecolor=col_n, markerfacecolor='w', markersize=15)

        ax.legend(handles=markers,
                  frameon=True, edgecolor='black',
                  fontsize=9, loc="upper left",
                  ncols=1, framealpha=1, facecolor='none')

        if despine_left:
            ax.yaxis.set_label_position("right")
            sns.despine(ax=ax, right=False, left=True)
        else:
            sns.despine(ax=ax)

    ax.xaxis.set_major_locator(mticker.MaxNLocator(5))
    return ax


def plot_bdist(
        results_object,
        oddsratio: bool,
        specs: Optional[List[List[str]]] = None,
        ax: Optional[plt.Axes] = None,
        title: str = '',
        despine_left: bool = True,
        legend_bool: bool = False,
        bw_adjust: float = 0.5,
        highlights: bool = True
) -> plt.Axes:
    """
    Plot density‐scaled histograms and KDEs of coefficient distributions,
    in a fully generalisable way.
    """
    # 1. Build a long‐form DataFrame with one row per (draw, spec)
    if oddsratio is True:
        draws_df = results_object.estimates_exp.T.copy()
    else:
        draws_df = results_object.estimates.T.copy()
    # flatten the spec‐names into strings
    spec_labels = [s for s in results_object.specs_names]
    draws_df.columns = spec_labels
    if highlights:
        # pick out the special ones
        null_label = spec_labels[0]
        full_label = spec_labels[-1]
    highlight = []
    if specs:
        requested: Set[frozenset] = {frozenset(sp) for sp in specs}
        highlight = [lab for lab in spec_labels if lab in requested]

    # define the order we'll plot (so colors map consistently)
    if highlights:
        order = [null_label] + highlight + [full_label]
    else:
        order = highlight
    if specs is None and highlights is False:
        df_long = draws_df.melt(var_name='spec', value_name='coef')
        # ensure the order is preserved
        palette = get_colormap_colors(1)
        hue=None
    else:
        df_long = draws_df[order].melt(var_name='spec', value_name='coef')
        # ensure the order is preserved
        palette = get_colormap_colors(len(order))
        hue='spec'

    if ax is None:
        ax = plt.gca()

    sns.kdeplot(
        data=df_long,
        x='coef',
        hue=hue,
        common_norm=False,  # each group integrates to 1
        bw_adjust=bw_adjust,  # controls smoothness (h ∝ bw_adjust)
        palette=palette,  # line colours for each 'spec'
        linewidth=2,
        fill=True,  # fill under the curve
        alpha=0.3,  # light shading
        ax=ax,
        legend=False
    )

    # 3. Optionally draw a custom legend
    if legend_bool:
        handles = []
        for col, lab in zip(palette, order):
            handles.append(
                Line2D([0], [0],
                       marker='s',
                       color=col,
                       markerfacecolor=col,
                       markersize=10,
                       linestyle='',
                       label=lab)
            )
        ax.legend(handles=handles, title='Specification',
                  frameon=True, loc='upper right')

    # 4. Final formatting
    axis_formatter(ax, 'Density', 'Bootstrapped Estimand', title)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(5))
    if despine_left:
        ax.yaxis.set_label_position("right")
        sns.despine(ax=ax, right=False, left=True)
    else:
        sns.despine(ax=ax)
    return ax


def plot_kfolds(
        results_object,
        colormap: Union[str, matplotlib.colors.Colormap],
        ax: Optional[plt.Axes] = None,
        title: str = '',
        despine_left: bool = True,
        tau: float = 0.6
) -> plt.Axes:
    """
    Plot the cross-validation metric distribution (density + histogram),
    with an adaptive legend positioned safely around the tallest bars.

    Parameters
    ----------
    results_object : object
        Must expose:
          - summary_df : pandas.DataFrame containing column 'av_k_metric'
          - name_av_k_metric : str, the metric name (e.g. 'r-squared', 'rmse')
    colormap : str or Colormap
        Matplotlib colormap name or object used for plotting.
    ax : matplotlib.axes.Axes, optional
        Axes on which to draw; if None a new (4×3) figure and axes are created.
    title : str, default=''
        Title to display above the plot.
    despine_left : bool, default=True
        If True, move y-axis ticks & label to the right spine; otherwise keep on the left.
    tau : float in (0,1), default=0.6
        Safety factor for legend placement: bars taller than tau*ylim are
        considered “in the way” and flip the legend to the opposite side.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes containing the completed plot.
    """

    if ax is None:
        _, ax = plt.subplots(figsize=(4, 3))

    # KDE & histogram
    data = results_object.summary_df['av_k_metric']
    colors = get_colormap_colors(2)
    sns.kdeplot(data, ax=ax, alpha=1, color=colors[1])
    sns.histplot(data, ax=ax, alpha=1, color=colors[0], bins=30, stat='density')

    # Symmetric x-padding
    val_range = data.max() - data.min()
    ax.set_xlim(data.min() - 0.1 * val_range, data.max() + 0.1 * val_range)
    ax.set_xlim(ax.get_xlim()[0] - 0.066 * val_range, ax.get_xlim()[1])  # original tweak

    # Adaptive legend location
    legend_loc = _legend_side_from_hist(ax, tau=tau)

    legend_elements = [
        Line2D([0], [0], color=colors[1], lw=2, label='Density'),
        Patch(facecolor=colors[0], edgecolor=(0, 0, 0, 1), label='Histogram')
    ]
    ax.legend(handles=legend_elements,
              loc=legend_loc,
              frameon=True,
              fontsize=9,
              title='Out-of-Sample',
              title_fontsize=10,
              framealpha=1,
              facecolor='w',
              edgecolor=(0, 0, 0, 1))

    # Cosmetic axes work
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.grid(linestyle='--', color='k', alpha=0.1, zorder=-1)
    ax.set_axisbelow(True)

    name = results_object.name_av_k_metric
    metric = r'R$^2$' if name.lower() == 'r-squared' else (name.upper() if name == 'rmse' else name.title())
    axis_formatter(ax, 'Density', f'OOS Metric: {metric}', title)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(5))

    if despine_left:
        ax.yaxis.set_label_position("right")
        sns.despine(ax=ax, right=False, left=True)
    else:
        sns.despine(ax=ax)


def plot_bma(
        results_object,
        colormap: Union[str, matplotlib.colors.Colormap],
        ax: plt.Axes,
        feature_order: Sequence[str],
        title: str = ''
) -> plt.Axes:
    """
    Plot Bayesian Model Averaging (BMA) inclusion probabilities as a horizontal bar chart.

    Parameters
    ----------
    results_object : object
        Must implement `compute_bma()` returning a DataFrame with columns:
          - 'control_var'
          - 'probs'
    colormap : str or Colormap
        Matplotlib colormap name or object used to pick the bar color.
    ax : matplotlib.axes.Axes
        Axes on which to draw the horizontal bar chart.
    feature_order : sequence of str
        Ordered list of control variable names to display on the y-axis.
    title : str, default=''
        Title to display above the plot.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes containing the completed BMA plot.
    """
    bma = results_object.compute_bma()
    bma = bma.set_index('control_var')
    bma = bma.reindex(feature_order)
    bma['probs'].plot(kind='barh',
                      ax=ax,
                      alpha=1,
                      color=get_colormap_colors(1),
                      edgecolor='k',
                      )
    axis_formatter(ax, r'', 'BMA Probabilities', title)
    sns.despine(ax=ax)


def title_setter(
        ax: plt.Axes,
        title: str,
        side: str = 'left'
) -> None:
    """
    Set a title on `ax`, aligned on the left but positioned differently
    depending on whether the y-axis is on the left or right.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes whose title you wish to set.
    title : str
        The title text.
    side : {'left', 'right'}, default='left'
        - 'left': standard positioning.
        - 'right': shifts the title so it doesn’t overlap a right-side y-axis.
    """
    if side == 'right':
        return ax.set_title(title, loc='left', fontsize=16, y=1, x=-.26)
    else:
        return ax.set_title(title, loc='left', fontsize=16, y=1)


def _sanitize_specs(
        specs: Optional[List[List[str]]],
        max_len: int = 4
) -> Optional[List[List[str]]]:
    """
    Truncate the specs list to at most `max_len`, warning if we had to cut.
    """
    if specs is not None and len(specs) > max_len:
        warnings.warn(
            f"Received {len(specs)} specs; only the first {max_len} will be used.",
            UserWarning
        )
        return specs[:max_len]
    return specs


def _prepare_output_dir(figpath: Optional[Path], project: Optional[str]) -> Path:
    base = figpath or Path.cwd()
    if project:
        base = base / project
    try:
        base.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise RuntimeError(f"Could not create output directory {base!r}") from e
    return base


def plot_results(
        results_object,
        loess: bool = True,
        ci: float = 0.95,
        specs: Optional[List[List[str]]] = None,
        ic: Optional[str] = None,
        colormap: Union[str, matplotlib.colors.Colormap] = 'Spectral_r',
        figsize: Tuple[int, int] = (16, 16),
        ext: str = 'pdf',
        figpath=None,
        highlights = True,
        oddsratio=False,
        project_name: str = None
) -> None:
    """
    Plots the coefficient estimates, IC curve, and distribution plots for the given results object.

    Parameters
    ----------
    results_object : object
        An OLSResult-like object (must expose attributes `y_name`, `x_name`,
        `shap_return`, `summary_df`, `specs_names`, etc.).
    loess : bool, default=True
        Whether to apply LOESS smoothing to the coefficient–specification curve.
    ci: float, default=0.95
        The confidence interval to use.
    specs : list of list of str, optional
        Up to three specs (lists of control names) to highlight in the curve, IC, and distribution panels.
    ic : str, optional
        Information criterion name to plot (one of 'aic','bic','hqic').
    colormap : str or Colormap, default='Spectral_r'
        Colormap used consistently for all panels.
    figsize : (width, height), default=(16,16)
        Size of the full figure in inches.
    figpath : str or Path, optional
        Directory in which to save outputs; if None, uses current working dir.
    ext : str, default='pdf'
        File extension to save each panel (e.g. 'png','pdf').
    project_name : str, default=None
        Directory and filename prefix under `./figures/`.
    oddsratio bool, default=False
        Whether to exponentiate the coefficients (e.g. for odds ratios).
    highlights bool, default=True
        Whether to highlight certain specifications.

    Notes
    -----
    - Saves a combined “_all” figure plus individual panels named:
      `_R2hexbin`, `_OOS`, `_curve`, `_LLhexbin`, `_SHAP`, `_BMA`, `_IC`, `_bdist`.
      for the case when len(y_name) == 1, and a subset for when >1.
    """

    # If 'draws' or 'kfold' is a list/tuple, assume this is a merged‐results object:
    if isinstance(results_object.draws, (list, tuple)) or isinstance(results_object.kfold, (list, tuple)):
        warnings.warn(
            "plot_results was passed a *merged* results object (draws/kfold are lists).  "
            "This function does not support plotting a merged‐results object.  "
            "This is because it's difficult to know exactly what you want to do.  "
            "It's much safer for you to extract individual result objects and plot them separately.  "
            "Please raise an issue on GitHub.com/robustipy/ and we can discuss!"
            "Exiting without plotting.",
            UserWarning
        )
        return

    ext = ext.strip()
    specs = _sanitize_specs(specs, max_len=6)
    if oddsratio is True:
        if results_object.model_name=='Logistic Regression Robust':
            results_object.all_b_exp = np.exp([arr[0][0] for arr in results_object.all_b.copy()])
            results_object.estimates_exp = np.exp(results_object.estimates.copy())
        else:
            raise ValueError("`oddsratio` option is only valid for logistic regression models.")
    if not (0 <= ci <= 1):
        sys.exit(f"`ci` must lie strictly between 0 and 1; received ci={ci!r}")

    outdir = _prepare_output_dir(Path(figpath) if figpath else None, project_name)

    if max(len(t) for t in results_object.y_name) == 1:
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
        shap_x = results_object.shap_return[1].drop(columns=results_object.x_name, errors="ignore").to_numpy()
        shap_cols = results_object.shap_return[1].drop(columns=results_object.x_name, errors="ignore").columns
        plot_hexbin_r2(results_object, ax1, fig,
                       oddsratio, colormap, title='a.')
        plot_hexbin_log(results_object, ax2, fig,
                        oddsratio, colormap, title='b.')
        feature_order = shap_violin(ax4, shap_vals, shap_x,
                                    shap_cols, title='d.',
                                    clear_yticklabels=True)
        plot_bma(results_object, colormap,
                 ax3, feature_order, title='c.')
        plot_kfolds(results_object, colormap, ax5,
                    title='e.', despine_left=True)
        plot_curve(results_object=results_object, loess=loess, ci=ci,
                   specs=specs, ax=ax6, highlights=highlights,
                   title='f.', oddsratio=oddsratio)
        plot_ic(results_object=results_object, ic=ic, specs=specs,
                ax=ax7, colormap=colormap, title='g.', despine_left=True)
        plot_bdist(results_object=results_object, specs=specs,
                   ax=ax8, oddsratio=oddsratio, highlights=highlights,
                   title='h.', despine_left=True)
        plt.savefig(os.path.join(outdir, project_name + '_all.' + ext), bbox_inches='tight')
    else:
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(6, 24, wspace=-.25, hspace=5)
        ax1 = fig.add_subplot(gs[0:6, 0:16])
        ax2 = fig.add_subplot(gs[0:3, 17:24])
        ax3 = fig.add_subplot(gs[3:6, 17:24])
        plot_curve(results_object=results_object, loess=loess,
                   ci=ci, specs=specs, ax=ax1, highlights=highlights,
                   title='a.', oddsratio=oddsratio)
        plot_hexbin_r2(results_object, ax2, fig, oddsratio,
                       colormap, title='b.', side='right')
        plot_bdist(results_object=results_object, specs=specs,
                   ax=ax3, oddsratio=oddsratio, highlights=highlights,
                   title='c.', despine_left=True)
        plt.savefig(os.path.join(outdir, project_name + '_all.' + ext), bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(8.5, 5))
    plot_hexbin_r2(results_object, ax, fig, oddsratio, colormap)
    plt.savefig(os.path.join(outdir, project_name + '_R2hexbin.' + ext), bbox_inches='tight')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    plot_bdist(results_object=results_object, specs=specs, ax=ax,
               oddsratio=oddsratio,
               despine_left=False)
    plt.savefig(os.path.join(outdir, project_name + '_OOS.' + ext), bbox_inches='tight')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 7))
    plot_curve(results_object=results_object, loess=loess, ci=ci,
               oddsratio=oddsratio, highlights=highlights,
               specs=specs, ax=ax)
    plt.savefig(os.path.join(outdir, project_name + '_curve.' + ext), bbox_inches='tight')
    plt.close(fig)

    if max(len(t) for t in results_object.y_name) == 1:
        fig, ax = plt.subplots(figsize=(8.5, 5))
        plot_hexbin_log(results_object, ax, fig, oddsratio, colormap)
        plt.savefig(os.path.join(outdir, project_name + '_LLhexbin.' + ext), bbox_inches='tight')
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8.5, 5))
        feature_order = shap_violin(ax, shap_vals, shap_x, shap_cols, clear_yticklabels=False)
        plt.savefig(os.path.join(outdir, project_name + '_SHAP.' + ext), bbox_inches='tight')
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8.5, 5))
        plot_bma(results_object, colormap, ax, feature_order)
        plt.savefig(os.path.join(outdir, project_name + '_BMA.' + ext), bbox_inches='tight')
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8.5, 5))
        plot_ic(results_object=results_object, ic=ic, specs=specs, ax=ax, colormap=colormap, title='g.',
                despine_left=False)
        plt.savefig(os.path.join(outdir, project_name + '_IC.' + ext), bbox_inches='tight')
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8.5, 5))
        plot_bdist(results_object=results_object, specs=specs, ax=ax,
                   oddsratio=oddsratio,
                   despine_left=False, highlights=highlights,
                   legend_bool=False)
        plt.savefig(os.path.join(outdir, project_name + '_bdist.' + ext), bbox_inches='tight')
        plt.close(fig)
