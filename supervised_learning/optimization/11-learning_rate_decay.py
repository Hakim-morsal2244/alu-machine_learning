#!/usr/bin/env python3
"""Learning rate decay"""


def learning_rate_decay(alpha, decay_rate,
                        global_step, decay_step):
    """
    updates the learning rate using inverse time decay

    Args:
        alpha: original learning rate
        decay_rate: decay rate
        global_step: current gradient descent step
        decay_step: number of steps before decay

    Returns:
        updated learning rate
    """

    return alpha / (
        1 + decay_rate * (global_step // decay_step)
    )
