#!/usr/bin/env python3
"""Early stopping function."""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Determines whether to stop training early.

    cost: current validation cost
    opt_cost: best (lowest) validation cost so far
    threshold: minimum improvement required
    patience: max allowed steps without improvement
    count: current number of consecutive "no improvement" steps

    Returns:
        (stop_training: bool, updated_count: int)
    """

    # check if improvement is significant
    if opt_cost - cost > threshold:
        # improvement happened → reset counter
        return False, 0

    # no significant improvement → increase counter
    count += 1

    # stop if patience exceeded
    if count >= patience:
        return True, count

    return False, count
