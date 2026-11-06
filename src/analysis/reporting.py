"""
LaTeX tables and console output of the experiment scripts.

Inputs of the result tables are long DataFrames (method, scenario, dataset, metric, value) and the evaluated
blocks as dicts {'scenario', 'dataset', 'n', 'n_io'}, in table order.
"""

from pathlib import Path

import numpy as np

LATEX_PREAMBLE_NOTE = '% Requires \\usepackage{booktabs, makecell}'
METRIC_HEADERS = {'auc': 'AUC', 'ap': 'AP', 'ndcg': 'NDCG', 'ed': 'ED', 'p100': 'P@100', 'p500': 'P@500',
                  'fc80': 'FC@80\\%', 'fc90': 'FC@90\\%', 'fc95': 'FC@95\\%', 'red90': 'FC@90\\% saved (\\%)',
                  'pF': 'P@$k_F$'}
SCORE_FORMATS = {'auc': '.4f', 'ap': '.4f', 'ndcg': '.4f', 'ed': '.1f', 'p100': '.2f', 'p500': '.3f', 'pF': '.2f'}


def print_section(title):
    print("\n" + "=" * 80 + f"\n{title}\n" + "=" * 80)


def tex_escape(text):
    """Escape the '&' and '_' of a plain-text name."""
    return str(text).replace('&', '\\&').replace('_', '\\_')


def thousands(value):
    """Integer with a thin-space thousands separator: 137340 -> '137\\,340'."""
    return f"{int(round(value)):,}".replace(',', '\\,')


def table_lines(caption, label, colspec, header_lines, body_lines, env='table*', size='\\footnotesize',
                preamble_note=False):
    """Lines of a booktabs table: [preamble note], environment, caption, label, tabular with header and body."""
    return ([LATEX_PREAMBLE_NOTE] if preamble_note else []) + [
        f'\\begin{{{env}}}[t]', '\\centering', f'\\caption{{{caption}}}', f'\\label{{{label}}}', size,
        '\\setlength{\\tabcolsep}{4pt}', f'\\begin{{tabular}}{{{colspec}}}', '\\toprule', *header_lines,
        '\\midrule', *body_lines, '\\bottomrule', '\\end{tabular}', f'\\end{{{env}}}']


def write_lines(path, lines):
    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f"Saved: {path}")


def block_title(block):
    return f"{block['scenario']} $\\cdot$ {block['dataset']}"


def block_header(block, n_cols, unit='users', extra=''):
    """Full-width row, e.g. 'S1 · Honduras: 1,610 IO of 137,340 users (prevalence 1.17%)'."""
    prevalence = 100.0 * block['n_io'] / block['n']
    return (f"\\multicolumn{{{n_cols}}}{{l}}{{\\textbf{{{block_title(block)}}}: "
            f"{thousands(block['n_io'])} IO of {thousands(block['n'])} {unit} "
            f"(prevalence {prevalence:.2f}\\%){extra}}} \\\\")


def format_score(metric, value):
    if not np.isfinite(value):
        return '--'
    if metric.startswith('fc'):
        return thousands(value)
    return format(value, SCORE_FORMATS[metric])


def _bold_best(cells, metric):
    """Bold the best displayed value(s) of a column (lowest for FC)."""
    shown = [float(c.replace('\\,', '')) for c in cells if c != '--']
    if not shown:
        return cells
    best = (min if metric.startswith('fc') else max)(shown)
    return [f"\\textbf{{{c}}}" if c != '--' and float(c.replace('\\,', '')) == best else c for c in cells]


def results_table_lines(scores, blocks, columns, rows, caption, label, header_row=block_header):
    """
    table* with one block per scenario-dataset: a full-width header row (header_row(block, n_cols)) and one row
    per method, bold marking the best displayed value of each column.

    Args:
        columns: metric ids of the columns
        rows: [(method id, row label)]
    """
    s = scores.set_index(['method', 'scenario', 'dataset', 'metric'])['value']
    body = []
    for b, block in enumerate(blocks):
        if b:
            body.append('\\midrule')
        body.append(header_row(block, 1 + len(columns)))
        sc, ds = block['scenario'], block['dataset']
        table = [[format_score(m, s.get((method, sc, ds, m), np.nan)) for m in columns] for method, _ in rows]
        for j, m in enumerate(columns):
            for i, cell in enumerate(_bold_best([row[j] for row in table], m)):
                table[i][j] = cell
        body += [' & '.join([row_label] + cells) + ' \\\\' for (_, row_label), cells in zip(rows, table)]
    header = ' & '.join(['Approach'] + [METRIC_HEADERS[m] for m in columns]) + ' \\\\'
    return table_lines(caption, label, f"l{'r' * len(columns)}", [header], body, preamble_note=True)


def _format_interval(metric, lo, hi):
    if metric.startswith('red'):
        return f"[{lo:.1f}, {hi:.1f}]"
    return f"[{lo:.2f}, {hi:.2f}]"


def interval_pair_table_lines(first, second, blocks, comparisons, metrics, labels, summary, caption, label):
    """
    One row per (comparison, block); per metric the interval of two procedures side by side. An interval of the
    second procedure whose conclusion (above 0, below 0, contains 0) differs from the first one is in bold; a
    dagger marks an interval that contains 0.

    Args:
        first, second: DataFrames (comparison, scenario, dataset, metric, lo, hi, dagger)
        comparisons: [(comparison id, row label)]
        labels: (header of the first procedure, header of the second)
        summary: text of the final full-width line
    """
    f = first.set_index(['comparison', 'scenario', 'dataset', 'metric'])
    s = second.set_index(['comparison', 'scenario', 'dataset', 'metric'])
    top = ' & & ' + ' & '.join(f"\\multicolumn{{2}}{{c}}{{{METRIC_HEADERS[m]}}}" for m in metrics) + ' \\\\'
    rules = ' '.join(f"\\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(len(metrics)))
    sub = 'Comparison & Block & ' + ' & '.join(f'{labels[0]} & {labels[1]}' for _ in metrics) + ' \\\\'

    def conclusion(row):
        return '0' if row['dagger'] else ('+' if row['lo'] > 0 else '-')

    def interval(row, metric, bold=False):
        text = _format_interval(metric, row['lo'], row['hi']) + ('^{\\dagger}' if row['dagger'] else '')
        return f"$\\mathbf{{{text}}}$" if bold else f"${text}$"

    body = []
    for i, (comparison, row_label) in enumerate(comparisons):
        if i:
            body.append('\\midrule')
        for j, block in enumerate(blocks):
            sc, ds = block['scenario'], block['dataset']
            cells = [row_label if j == 0 else '', block_title(block)]
            for m in metrics:
                a, b = f.loc[(comparison, sc, ds, m)], s.loc[(comparison, sc, ds, m)]
                cells += [interval(a, m), interval(b, m, conclusion(a) != conclusion(b))]
            body.append(' & '.join(cells) + ' \\\\')
    body += ['\\midrule', f"\\multicolumn{{{2 + 2 * len(metrics)}}}{{l}}{{{summary}}} \\\\"]
    return table_lines(caption, label, 'll' + 'cc' * len(metrics), [top, rules, sub], body, size='\\scriptsize',
                       preamble_note=True)
