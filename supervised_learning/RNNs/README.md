# RNNs

This project introduces **Recurrent Neural Networks (RNNs)** and their fundamental components. The goal is to understand how RNN cells process sequential data and maintain information from previous time steps.

## Learning Objectives

By completing this project, I will learn how to:

* Understand the purpose and architecture of Recurrent Neural Networks.
* Explain how hidden states are updated across time steps.
* Implement a basic RNN cell using NumPy.
* Understand the role of weights and biases in an RNN.
* Apply the `tanh` activation function to calculate the next hidden state.
* Apply the `softmax` activation function to calculate the output.
* Understand the relationship between inputs, hidden states, and outputs.

## Resources

* MIT 6.S191: Recurrent Neural Networks
* Introduction to RNNs
* Illustrated Guide to RNNs
* Illustrated Guide to LSTMs and GRUs
* RNNs Tutorials
* Bidirectional RNN
* Deep RNN

## Project Structure

```text
supervised_learning/
└── RNNs/
    └── 0-rnn_cell.py
```

## Task 0: RNN Cell

The first task is to create an `RNNCell` class that represents a cell of a simple RNN.

The class contains:

* `Wh`: weights used for the concatenated previous hidden state and current input.
* `Wy`: weights used to calculate the output.
* `bh`: bias for the hidden state.
* `by`: bias for the output.

The `forward` method performs one step of forward propagation.

Given:

* `h_prev`: the previous hidden state.
* `x_t`: the current input.

The cell calculates:

```text
h_next = tanh([h_prev, x_t] · Wh + bh)
```

and then calculates the output using softmax:

```text
y = softmax(h_next · Wy + by)
```

The method returns:

```text
h_next, y
```

## Requirements

* Python 3
* NumPy
* Ubuntu/WSL or a compatible Linux environment
* Code follows the project requirements and documentation standards.

## Repository

GitHub repository:

`alu-machine_learning`

Task directory:

`supervised_learning/RNNs`

Task file:

`0-rnn_cell.py`
