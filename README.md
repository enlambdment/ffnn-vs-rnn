# ffnn-vs-rnn

TODO: Fix the hand-written `softmax'` implementation, which is incorrect

Adaptation of the [untyped neural network implementation](https://blog.jle.im/entry/practical-dependent-types-in-haskell-1.html) 
at the start of Justin Le's blog series on dependent types for machine learning.

I adapt the feed-forward neural network (FFNN) to implement a modified neural network in the style of
recurrent neural networks (RNN.)

The problem that the model is trained to solve in all three examples: learn the infinite repeating sequence
`0, 1, 2, 0, 1, 2, ...`

Note that the back-propagation method used for both RNN examples (the logistic-final and softmax-final ones)
is _not_ back-propagation through time (BPTT) as described in _9. 1. 2. Training_ of [Jurafsky](https://web.stanford.edu/~jurafsky/slp3/9.pdf).
See also this [blog post](http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/)

```
-- FFNN on directly encoded numerals; sample inputs are 3-grams
> displayTraining 0.25 training_data       
-- RNN on directly encoded numerals; sample inputs are single integers 
> displayRNNTraining 0.25 rnn_training_data
-- RNN on indirectly encoded numerals; sample inputs / outputs are one-hot 3-by-1 vectors
> displayRNNTraining_softmax 0.25 rnn_training_data_onehot
```