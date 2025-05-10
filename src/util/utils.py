import re
from typing import TypeVar, List, Iterator, Sequence, Collection

T = TypeVar('T')


def get_sorted_vars(model):
    vars_list = model.getVars()
    # Sort the variables by their variable name
    return sorted(vars_list, key=lambda v: v.VarName)

def proper_subsets(collection: Collection[T]) -> Iterator[List[T]]:
    """
    Generates all non-empty proper subsets of a collection (i.e. power set excluding the empty set and the full set)

    Example:
        >>> list(proper_subsets([1, 2, 3]))
        [[1], [2], [1, 2], [3], [1, 3], [2, 3]]

    :return: Yields items of type List[T] for each non-empty proper subset of the input iterable.
    """

    total_length = len(collection)
    masks = [1 << i for i in range(total_length)]

    for i in range(1, (1 << total_length) - 2):
        yield [subset for mask, subset in zip(masks, collection) if i & mask]

def rooted_proper_subsets(collection: Sequence[T], root: T) -> Iterator[List[T]]:
    """
    Generates all non-empty proper subsets of a collection (i.e. power set excluding the empty set and the full set)
    that contain a specific element called 'root'.

    Example:
        >>> list(proper_subsets([1, 2, 3, 4], 1))
        [[1], [1, 2], [1, 3], [1,4], [1,2,3], [1,2,4], [1,3,4]]

    :return: Yields items of type List[T] for each non-empty proper subset of the input iterable.
    """

    # check if the root element is actually in the collection
    if root not in collection:
        raise ValueError("Passed item 'root' is not in the given collection.")

    n = len(collection)

    # Generate and yield all possible subsets using bit manipulation
    for i in range(1, (1 << n) - 1):  # Skip empty set (0) and the full set (2^n - 1)
        subset = [collection[j] for j in range(n) if (i & (1 << j))]
        if root in subset:
            yield subset


def latex_escape(s):
    """
    Escape characters that have a special meaning in the LaTeX semantics.
    """
    return re.sub(r'([&_#%])', r'\\\1', s)


def generate_latex_table(model, output_file="variable_table.txt"):
    lines = [
        r"\begin{longtable}{|c|c|}",
        r"\hline",
        r"\textbf{Variable} & \textbf{Value} \\",
        r"\hline",
        r"\endfirsthead",
        r"\hline \textbf{Variable} & \textbf{Value} \\ \hline",
        r"\endhead"
    ]

    for var in model.getVars():
        name = latex_escape(var.VarName)
        val = var.X
        lines.append(f"{name} & {val:.4f} \\\\")
        lines.append(r"\hline")

    lines.append(r"\end{longtable}")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def generate_three_tables_per_page(model, vars_per_table=25, output_file="variable_table.txt"):
    vars_list = sorted(model.getVars(), key=lambda v: v.VarName)
    total_vars = len(vars_list)

    tables = []
    for i in range(0, total_vars, vars_per_table):
        chunk = vars_list[i:i + vars_per_table]

        table = [
            r"\begin{tabular}{|c|c|}",
            r"\hline",
            r"\textbf{Variable} & \textbf{Value} \\",
            r"\hline"
        ]

        for var in chunk:
            name = latex_escape(var.VarName)
            val = var.X
            table.append(f"{name} & {val:.4f} \\\\")
            table.append(r"\hline")

        table.append(r"\end{tabular}")
        tables.append("\n".join(table))

    # Group every 3 tables with spacing and page breaks
    final_output = []
    for i in range(0, len(tables), 3):
        group = tables[i:i + 3]
        final_output.append(r"\noindent")
        for t in group:
            final_output.append(r"{\small")
            final_output.append(t)
            final_output.append(r"}")
            final_output.append(r"\vspace{1em}")
        final_output.append(r"\clearpage")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(final_output))


def generate_three_long_tables_per_page(model, vars_per_table=50, precision_digits=3, output_file="variable_table.txt"):
    # Sort the variables by their variable name
    vars_list = sorted(model.getVars(), key=lambda v: v.VarName)

    # Now, total_vars would still be the same
    total_vars = len(vars_list)

    tables = []
    for i in range(0, total_vars, vars_per_table):
        chunk = vars_list[i:i + vars_per_table]

        table = [
            r"\begin{tabular}{|c|c|}",
            r"\hline",
            r"\textbf{Variable} & \textbf{Value} \\",
            r"\hline"
        ]

        for var in chunk:
            name = latex_escape(var.VarName)
            val = var.X
            if val == 0:
                table.append(f"{name} & 0 \\\\")
            else:
                table.append(f"{name} & {val:.{precision_digits}f} \\\\")
            table.append(r"\hline")

        table.append(r"\end{tabular}")
        tables.append("\n".join(table))

    # Group every 3 tables, then insert a page break
    final_output = []
    for i in range(0, len(tables), 4):
        group = tables[i:i + 4]
        final_output.append(r"\noindent")
        for t in group:
            final_output.append(r"{\small")
            final_output.append(t)
            final_output.append(r"}")
            final_output.append(r"\vspace*{1em}")
        if i < len(tables) - 4:
            final_output.append(r"\clearpage")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(final_output))