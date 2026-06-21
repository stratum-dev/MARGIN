"""
Lightweight string-formatting helpers.
"""


def print_dict_pipe(d: dict):
    """
    Format a dictionary as a single pipe-delimited string.

    Example: ``{"a": 1, "b": 2}`` → ``"a=1 | b=2"``.

    Parameters
    ----------
    d : dict
        The dictionary to format.

    Returns
    -------
    str
        Pipe-delimited ``key=value`` string.
    """
    parts = []
    for k, v in d.items():
        parts.append(f"{k}={v}")
    return " | ".join(parts)
