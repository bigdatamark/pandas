# [MODULE DOC]
# Standalone script that benchmarks the expression evaluation and query methods
# of a data frame with the python and numexpr engines over a range of row
# counts, and plots the timings (log-log) to image files used by the
# documentation.
# Public contents: string templates setup_common, setup_with and setup_subset
# (setup code for timed snippets); functions bench_with, bench_subset, bench and
# plot_perf. Requires numpy, seaborn and matplotlib.
# Import-time side effects: imports seaborn (hard dependency) and pandas.
# Run as a script it benchmarks, then writes two png files into the static
# folder of the documentation source tree.
from timeit import repeat as timeit

import numpy as np
import seaborn as sns

from pandas import DataFrame

setup_common = """from pandas import DataFrame
import numpy as np
df = DataFrame(np.random.randn(%d, 3), columns=list('abc'))
%s"""

setup_with = "s = 'a + b * (c ** 2 + b ** 2 - a) / (a * c) ** 3'"


def bench_with(n, times=10, repeat=3, engine="numexpr"):
    # [FUNCTION DOC]
    # Times evaluating an arithmetic expression string over three columns of a
    # frame of n rows of standard normal random numbers (columns a, b, c).
    # Parameters: n (row count), times (executions per repetition, default 10),
    # repeat (number of repetitions, default 3), engine (name of the evaluation
    # engine, default numexpr). The setup (frame creation) is not timed.
    # Returns a numpy array with one entry per repetition: the average seconds
    # per execution (total time divided by times).
    return (
        np.array(
            timeit(
                f"df.eval(s, engine={engine!r})",
                setup=setup_common % (n, setup_with),
                repeat=repeat,
                number=times,
            )
        )
        / times
    )


setup_subset = "s = 'a <= b <= c ** 2 + b ** 2 - a and b > c'"


def bench_subset(n, times=20, repeat=3, engine="numexpr"):
    # [FUNCTION DOC]
    # Same as the evaluation benchmark but for the row-filtering query method
    # with a chained comparison and boolean condition. Parameters: n (row
    # count), times (executions per repetition, default 20), repeat
    # (repetitions, default 3), engine (default numexpr). Returns an array of
    # average seconds per execution, one per repetition.
    return (
        np.array(
            timeit(
                f"df.query(s, engine={engine!r})",
                setup=setup_common % (n, setup_subset),
                repeat=repeat,
                number=times,
            )
        )
        / times
    )


def bench(mn=3, mx=7, num=100, engines=("python", "numexpr"), verbose=False):
    # [FUNCTION DOC]
    # Runs both benchmarks across a logarithmic range of row counts.
    # Parameters: mn and mx (base-ten exponents of the smallest and largest row
    # counts, defaults 3 and 7), num (number of sizes, default 100), engines
    # (sequence of engine names, default python and numexpr), verbose (print a
    # progress line every tenth size when true).
    # Row counts are num log-spaced values rounded to integers. For each engine
    # and size one repetition of one execution is timed and the mean stored.
    # Returns two frames (evaluation timings, query timings), each with one
    # column per engine and a size column of row counts. Side effect: prints
    # progress when verbose.
    # KNOWN ISSUE: the verbose check combines a boolean and a comparison with
    # the bitwise and operator rather than the logical operator; it works for
    # booleans but is unidiomatic.
    r = np.logspace(mn, mx, num=num).round().astype(int)

    ev = DataFrame(np.empty((num, len(engines))), columns=engines)
    qu = ev.copy(deep=True)

    ev["size"] = qu["size"] = r

    for engine in engines:
        for i, n in enumerate(r):
            if verbose & (i % 10 == 0):
                print(f"engine: {engine!r}, i == {i:d}")
            ev_times = bench_with(n, times=1, repeat=1, engine=engine)
            ev.loc[i, engine] = np.mean(ev_times)
            qu_times = bench_subset(n, times=1, repeat=1, engine=engine)
            qu.loc[i, engine] = np.mean(qu_times)

    return ev, qu


def plot_perf(df, engines, title, filename=None) -> None:
    # [FUNCTION DOC]
    # Plots timing curves, one line per engine, on log-log axes of row count
    # versus seconds. Parameters: df (frame with a size column and one column
    # per engine), engines (names of the columns to draw), title (plot title),
    # filename (optional path; the figure is saved there if given).
    # Side effects: applies the seaborn default style and the Set2 palette
    # globally; creates a matplotlib figure of 4 by 3 inches at 120 dpi (not
    # closed afterwards) and writes the image file when filename is given.
    # Returns nothing.
    from matplotlib.pyplot import figure

    sns.set()
    sns.set_palette("Set2")

    fig = figure(figsize=(4, 3), dpi=120)
    ax = fig.add_subplot(111)

    for engine in engines:
        ax.loglog(df["size"], df[engine], label=engine, lw=2)

    ax.set_xlabel("Number of Rows")
    ax.set_ylabel("Time (s)")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.tick_params(top=False, right=False)

    fig.tight_layout()

    if filename is not None:
        fig.savefig(filename)


if __name__ == "__main__":
    import os

    pandas_dir = os.path.dirname(
        os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
    )
    static_path = os.path.join(pandas_dir, "doc", "source", "_static")

    join = lambda p: os.path.join(static_path, p)

    fn = join("eval-query-perf-data.h5")

    engines = "python", "numexpr"

    ev, qu = bench(verbose=True)  # only this one

    plot_perf(ev, engines, "DataFrame.eval()", filename=join("eval-perf.png"))
    plot_perf(qu, engines, "DataFrame.query()", filename=join("query-perf.png"))
