"""Figures of the comparisons: cumulative IO recall, detection curves and forest plots of differences."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, DrawingArea, HPacker, TextArea, VPacker

# Colour and line style of every approach (approach name: style), the same in every figure
APPROACH_STYLES = {
    'coretweets':                {'color': "#f14c4f", 'linestyle': '--'},
    'coretweets_idf':            {'color': "#DDA919", 'linestyle': '-.'},
    'coretweets_inv_m':          {'color': '#EE99AA', 'linestyle': ':'},
    'coretweets_weighted_12h':   {'color': '#004488', 'linestyle': '-'},
    'coretweets_weighted_1day':  {'color': "#3376B9", 'linestyle': (0, (8, 3))},
    'coretweets_weighted_2days': {'color': "#79BAFC", 'linestyle': '-.'},
    'lexicographic':             {'color': "#489FB5", 'linestyle': (0, (5, 1.5, 1, 1.5, 1, 1.5))},
}

ZOOM_COLOUR, ZOOM_LINEWIDTH = 'gray', 1    # frame of the insets, rectangle of the zoomed region and connectors
INSET_TICK_SIZE = 10                         # font size of the tick labels of the insets
INSET_LINE_SCALE = 0.8                      # width of the curves of the insets relative to the main panel


def approach_style(name):
    """{'color', 'linestyle'} of an approach (APPROACH_STYLES), a new dict the caller may extend."""
    if name not in APPROACH_STYLES:
        raise KeyError(f"no style for approach {name} (known: {sorted(APPROACH_STYLES)})")
    return dict(APPROACH_STYLES[name])


def save_figure(fig, *paths, dpi=300):
    """Save a figure to every path (tight bounding box) and close it."""
    for path in paths:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved: {path}")
    plt.close(fig)


def tie_block_curve(X, Y, n_samples=50, x_min=1.0):
    """
    Points of a cumulative curve that is linear in x inside every tie block, from (X_{j-1}, Y_{j-1})
    to (X_j, Y_j), sampled densely (n_samples points per block, geometrically spaced so that the
    shape is also right on a log x axis) instead of drawn as a straight segment between endpoints.
    Points with x < x_min are dropped (log axis).
    """
    xs, ys = [], []
    for a, b, ya, yb in zip(X[:-1], X[1:], Y[:-1], Y[1:]):
        if b < x_min or b <= a:
            continue
        x = np.geomspace(max(a, x_min), b, n_samples)
        xs.append(x)
        ys.append(ya + (x - a) * (yb - ya) / (b - a))
    return np.concatenate(xs), np.concatenate(ys)


def plot_cumulative_recall_grid(panels, styles, output_path, recall_levels=(0.8, 0.9, 0.95),
                                marker_recall=0.9):
    """
    Grid of cumulative IO recall curves (users inspected on a log axis vs IO recall).

    Args:
        panels: rows of panels; each panel is a dict with 'title', 'n' (users of the panel),
                'n_io', 'curves' {method: (X, Y) block endpoints} and 'fc' {method: users to
                reach marker_recall}
        styles: {method: {'label', 'color', 'linestyle'}}, in legend order
        output_path: PNG path
        recall_levels: thin grey dotted reference lines
        marker_recall: recall of the marker drawn on every curve at (fc, marker_recall)
    """
    n_rows, n_cols = len(panels), max(len(row) for row in panels)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 3.9 * n_rows),
                             sharey=True, squeeze=False)
    for r, row in enumerate(panels):
        for col, panel in enumerate(row):
            ax = axes[r, col]
            for level in recall_levels:
                ax.axhline(level, color='0.6', linestyle=':', linewidth=0.8, zorder=1)
            for method, style in styles.items():
                if method not in panel['curves']:
                    continue
                X, Y = panel['curves'][method]
                x, y = tie_block_curve(np.asarray(X, float), np.asarray(Y, float) / panel['n_io'])
                ax.plot(x, y, color=style['color'], linestyle=style['linestyle'], linewidth=1.8,
                        label=style['label'], zorder=3)
                fc = panel['fc'].get(method)
                if fc is not None and np.isfinite(fc):
                    ax.plot([fc], [marker_recall], marker='o', markersize=6, color=style['color'],
                            markeredgecolor='white', markeredgewidth=0.8, linestyle='none', zorder=4)
            ax.set_xscale('log')
            ax.set_xlim(1, panel['n'])
            ax.set_ylim(0, 1)
            ax.set_title(panel['title'])
            ax.grid(True, which='major', alpha=0.25)
            ax.spines[['top', 'right']].set_visible(False)
            if r == n_rows - 1:
                ax.set_xlabel('Users inspected (log scale)')
            if col == 0:
                ax.set_ylabel('IO recall')
        for col in range(len(row), n_cols):
            axes[r, col].axis('off')

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(labels), frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, output_path)


def text_table_box(fig, cells, xy, transform, fontsize=8.5, sep=12, bboxprops=None):
    """
    Box with a table of text cells (mathtext allowed), the first column left-aligned and the others
    right-aligned, anchored by its top centre at xy. Every cell is a separate text, so the columns
    stay aligned whatever the rendered width of a mathtext label.
    """
    renderer = fig.canvas.get_renderer()
    to_points = 1.0 / renderer.points_to_pixels(1.0)
    areas = [[TextArea(text, textprops=dict(family='monospace', fontsize=fontsize)) for text in row]
             for row in cells]
    for area in (a for row in areas for a in row):
        area.set_figure(fig)
    widths = [max(row[j].get_bbox(renderer).width * to_points for row in areas) for j in range(len(cells[0]))]
    rows = []
    for row in areas:
        packed = []
        for j, area in enumerate(row):
            pad = DrawingArea(widths[j] - area.get_bbox(renderer).width * to_points, 0)
            packed.append(HPacker(children=[area, pad] if j == 0 else [pad, area], sep=0, pad=0,
                                  align='baseline'))
        rows.append(HPacker(children=packed, sep=sep, pad=0, align='baseline'))
    return AnnotationBbox(VPacker(children=rows, sep=2, pad=0, align='left'), xy, xycoords=transform,
                          box_alignment=(0.5, 1.0), bboxprops=bboxprops, pad=0.3, annotation_clip=False)


def detection_area_differences(X, Y, X_base, Y_base, n):
    """
    Total area difference and negative area between two detection curves given by their tie-block
    endpoints: the curves are evaluated at every number of users studied x = 0..n (linear inside a
    tie block) and sum(y - y_base) and sum(min(0, y - y_base)) are returned (units: IO users x users
    studied).
    """
    x = np.arange(n + 1)
    diff = np.interp(x, X, Y) - np.interp(x, X_base, Y_base)
    return float(diff.sum()), float(np.minimum(0.0, diff).sum())


def plot_detection_curves_with_areas(panels, styles, baseline, output_path, recall_levels=(0.8, 0.9, 0.95)):
    """
    Detection figure: number of IO users detected against the number of users studied (linear
    axes), every method in the same panel, and below each panel a box with the total area
    difference and the negative area of every method against the baseline
    (detection_area_differences).

    Args:
        panels: rows of panels; each panel is a dict with 'title', 'n' (users of the panel),
                'n_io' and 'curves' {method: (X, Y) block endpoints, Y in IO users}, and optionally
                the axis limits 'x_min' (default 0), 'x_max' (default n), 'y_min' (default 0) and 'y_max'
                (default: automatic)
        styles: {method: {'label', 'color', 'linestyle'} and optionally 'linewidth' (default 1.8)}, in
                legend order
        baseline: method id of the baseline
        recall_levels: thin grey dotted lines at these fractions of the IO users, labelled at
                       the right end below the line
    """
    fig, _ = detection_curves_figure(panels, styles, baseline, recall_levels)
    save_figure(fig, output_path)


def detection_curves_figure(panels, styles, baseline, recall_levels):
    """Figure and axes of plot_detection_curves_with_areas (same arguments), not saved."""
    n_rows, n_cols = len(panels), max(len(row) for row in panels)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.2 * n_cols, 5.6 * n_rows), squeeze=False)
    others = [m for m in styles if m != baseline]
    for r, row in enumerate(panels):
        for col, panel in enumerate(row):
            ax = axes[r, col]
            for level in recall_levels:
                ax.axhline(level * panel['n_io'], color='0.6', linestyle=':', linewidth=1.4, zorder=1)
                ax.annotate(f"{level:.0%} of IO", xy=(1, level * panel['n_io']), xycoords=('axes fraction', 'data'),
                            xytext=(-2, -2), textcoords='offset points', ha='right', va='top', fontsize=8,
                            color='0.4', zorder=2)
            for method, style in styles.items():
                X, Y = panel['curves'][method]
                ax.plot(X, Y, color=style['color'], linestyle=style['linestyle'], linewidth=style.get('linewidth', 1.8),
                        label=style['label'] + (' (baseline)' if method == baseline else ''), zorder=3)
            ax.set_xlim(panel.get('x_min', 0), panel.get('x_max', panel['n']))
            ax.set_ylim(panel.get('y_min', 0), panel.get('y_max'))
            ax.set_title(panel['title'])
            ax.set_xlabel('Number of users studied')
            ax.set_ylabel('Number of IO users detected')
            ax.grid(True, alpha=0.3)
            ax.spines[['top', 'right']].set_visible(False)
            ax.ticklabel_format(axis='x', style='plain')
            Xb, Yb = panel['curves'][baseline]
            cells = [['vs ' + styles[baseline]['label'], 'Total area diff', 'Negative area']]
            for method in others:
                total, negative = detection_area_differences(*panel['curves'][method], Xb, Yb, panel['n'])
                cells.append(['    ' + styles[method]['label'], f"{total:+,.0f}", f"{negative:,.0f}"])
            ax.add_artist(text_table_box(fig, cells, (0.5, -0.2), ax.transAxes, fontsize=8.5,
                                         bboxprops=dict(boxstyle='round', facecolor='wheat', alpha=0.8)))
        for col in range(len(row), n_cols):
            axes[r, col].axis('off')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=6.0)
    return fig, axes


def plot_detection_curves_with_areas_zoomed(panels, styles, baseline, output_path, zooms,
                                            recall_levels=(0.8, 0.9, 0.95)):
    """
    The figure of plot_detection_curves_with_areas with the axes of some panels cut to a region, so that the
    differences between the curves near the top are visible. The areas of the boxes are unchanged.

    Args:
        zooms: {panel title: {'x_min', 'x_max', 'y_min', 'y_max'}}, any subset of the limits (the missing
               ones: those of the full panel); the other panels are not cut
        the other arguments: as in plot_detection_curves_with_areas
    """
    zoomed = [[{**panel, **zooms.get(panel['title'], {})} for panel in row] for row in panels]
    plot_detection_curves_with_areas(zoomed, styles, baseline, output_path, recall_levels)


def plot_detection_curves_with_areas_inset(panels, styles, baseline, output_path, zooms,
                                           recall_levels=(0.8, 0.9, 0.95), bounds=(0.40, 0.10, 0.56, 0.40)):
    """
    The figure of plot_detection_curves_with_areas (full axes) with, in the panels of zooms, an inset that
    zooms the curves of a region, a rectangle marking the region and two lines joining the lower left and the
    upper right corners of the rectangle to those of the inset (Axes.inset_axes and Axes.indicate_inset_zoom).

    Args:
        zooms: {panel title: {'x_min', 'x_max', 'y_min', 'y_max'}}, any subset of the limits of the region
               (the missing ones: those of the full panel); the other panels have no inset
        bounds: (x0, y0, width, height) of the inset in axes fraction (default: bottom right)
        the other arguments: as in plot_detection_curves_with_areas
    """
    fig, axes = detection_curves_figure(panels, styles, baseline, recall_levels)
    for r, row in enumerate(panels):
        for col, panel in enumerate(row):
            if panel['title'] not in zooms:
                continue
            zoom, ax = zooms[panel['title']], axes[r, col]
            inset = ax.inset_axes(bounds)
            for level in recall_levels:
                inset.axhline(level * panel['n_io'], color='0.6', linestyle=':', linewidth=1.4, zorder=1)
            for method, style in styles.items():
                inset.plot(*panel['curves'][method], color=style['color'], linestyle=style['linestyle'],
                           linewidth=INSET_LINE_SCALE * style.get('linewidth', 1.8), zorder=3)
            inset.set_xlim(zoom.get('x_min', 0), zoom.get('x_max', panel['n']))
            inset.set_ylim(zoom.get('y_min', 0), zoom.get('y_max', ax.get_ylim()[1]))
            inset.grid(True, alpha=0.3)
            inset.tick_params(labelsize=INSET_TICK_SIZE)
            inset.ticklabel_format(axis='x', style='plain')
            inset.xaxis.set_major_locator(plt.MaxNLocator(4))
            for spine in inset.spines.values():
                spine.set(edgecolor=ZOOM_COLOUR, linewidth=ZOOM_LINEWIDTH)
            indicator = ax.indicate_inset_zoom(inset, edgecolor=ZOOM_COLOUR, linewidth=ZOOM_LINEWIDTH, alpha=1.0)
            # connectors: (lower left, upper left, lower right, upper right); keep lower left and upper right
            for connector, visible in zip(indicator.connectors, (True, False, False, True)):
                connector.set_visible(visible)
    save_figure(fig, output_path)


def plot_difference_forest(diffs, blocks, comparisons, styles, metrics, output_path,
                           metric_labels=None, xlabel='Difference (percentage points)',
                           hollow_label='Hollow marker: interval contains 0',
                           interval_labels=None, row_groups=None, marker_size=7, panel_width=3.4,
                           fontsize=9, legend_ncol=4):
    """
    Forest plot of paired differences with their intervals: one panel per metric, one group of
    rows per block, one row (colour and marker) per comparison. A hollow marker means that the
    interval contains 0 ('dagger'). If diffs has columns lo2 and hi2, a second interval of each
    difference is drawn as a thin line with end caps, behind the thick first one.

    Args:
        diffs: DataFrame comparison, scenario, dataset, metric, estimate, lo, hi, dagger
               (optionally lo2, hi2)
        blocks: [(scenario, dataset)] from top to bottom
        comparisons: [comparison id], top to bottom inside a block
        styles: {comparison id: {'label', 'color', 'marker'}}
        metrics: metric ids, one panel each
        hollow_label: legend text of the hollow marker
        interval_labels: (legend text of the thick interval, of the thin one)
        row_groups: optional [[(scenario, dataset)]], one row of panels per group, each panel with
                    its own x-axis (for blocks whose differences have very different sizes);
                    default: a single row with all the blocks
        marker_size: size of the point-estimate markers
        panel_width: width of each panel (inches)
        fontsize: base font size (titles +2, tick labels +1)
        legend_ncol: maximum number of legend columns
    """
    d = diffs.set_index(['comparison', 'scenario', 'dataset', 'metric'])
    second = 'lo2' in d.columns and 'hi2' in d.columns
    n_cmp = len(comparisons)
    step = 1.0 / (n_cmp + 1)
    groups = row_groups or [blocks]
    n_rows_total = sum(len(g) for g in groups)
    fig, axes = plt.subplots(len(groups), len(metrics),
                             figsize=(panel_width * len(metrics) + 1.2,
                                      0.42 * n_cmp * n_rows_total + 1.4 + 0.5 * (len(groups) - 1)),
                             sharey='row', squeeze=False,
                             gridspec_kw={'height_ratios': [len(g) for g in groups]})
    for row, group in enumerate(groups):
        for col, metric in enumerate(metrics):
            ax = axes[row, col]
            ax.axvline(0.0, color='0.35', linewidth=0.9, zorder=1)
            for b, (scenario, dataset) in enumerate(group):
                if b % 2 == 0:
                    ax.axhspan(-b - 0.5, -b + 0.5, color='0.94', zorder=0, linewidth=0)
                for c, comparison in enumerate(comparisons):
                    key = (comparison, scenario, dataset, metric)
                    if key not in d.index:
                        continue
                    r = d.loc[key]
                    y = -b + 0.5 - (c + 1) * step
                    style = styles[comparison]
                    if second and np.isfinite(r['lo2']) and np.isfinite(r['hi2']):
                        ax.plot([r['lo2'], r['hi2']], [y, y], color=style['color'], linewidth=0.9,
                                marker='|', markersize=6, markeredgewidth=0.9, zorder=2)
                    ax.plot([r['lo'], r['hi']], [y, y], color=style['color'], linewidth=3.2,
                            solid_capstyle='butt', zorder=2)
                    ax.plot([r['estimate']], [y], marker=style['marker'], markersize=marker_size,
                            linestyle='none', color=style['color'],
                            markerfacecolor='white' if r['dagger'] else style['color'],
                            markeredgewidth=1.2, zorder=3)
            if row == 0:
                ax.set_title((metric_labels or {}).get(metric, metric.upper()), fontsize=fontsize + 2,
                             fontweight='bold')
            if row == len(groups) - 1:
                ax.set_xlabel(xlabel, fontsize=fontsize)
            ax.grid(True, axis='x', alpha=0.3, linestyle='--')
            ax.spines[['top', 'right', 'left']].set_visible(False)
            ax.tick_params(axis='y', length=0)
            ax.tick_params(axis='x', labelsize=fontsize + 1)
        axes[row, 0].set_yticks([-b for b in range(len(group))])
        axes[row, 0].set_yticklabels([f"{s} · {ds}" for s, ds in group], fontsize=fontsize + 1)
        axes[row, 0].set_ylim(-len(group) + 0.5, 0.5)
    # Legend drawn explicitly: filled markers for the comparisons, plus the meaning of a hollow one
    handles = [Line2D([], [], color=styles[c]['color'], marker=styles[c]['marker'], markersize=7, linewidth=2.0,
                      label=styles[c]['label']) for c in comparisons]
    handles.append(Line2D([], [], color='0.35', marker='o', markersize=7, markerfacecolor='white',
                          markeredgewidth=1.4, linestyle='none', label=hollow_label))
    if second and interval_labels:
        handles += [Line2D([], [], color='0.35', linewidth=3.2, label=interval_labels[0]),
                    Line2D([], [], color='0.35', linewidth=0.9, marker='|', markersize=6,
                           label=interval_labels[1])]
    ncol = min(len(handles), legend_ncol)
    rows = -(-len(handles) // ncol)
    fig.legend(handles=handles, loc='upper center', ncol=ncol, frameon=False, bbox_to_anchor=(0.5, 1.0),
               fontsize=fontsize)
    # Room for the legend: about 0.25 inch per legend row at fontsize 9, proportional to the font
    fig.tight_layout(rect=(0, 0, 1, 1.0 - (0.25 * fontsize / 9 * rows + 0.1) / fig.get_figheight()))
    save_figure(fig, output_path)
