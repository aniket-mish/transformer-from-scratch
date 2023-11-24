# transformer-from-scratch

I am implementing transformers from scratch to understand better the concepts involved.

## History

Before the transformer came into the picture, RNNs were used for most of the seq-to-seq tasks.

### Recurrent Neural Networks (RNN)

![image](https://github.com/aniket-mish/transformers-from-scratch/assets/71699313/171071ce-0bfb-43ad-8414-35cdf3c00128)
A recurrent neural network can be thought of as multiple copies of the same network, each passing a message to a successor.

[Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

Disadvantages of RNNs:
- Cannot capture long-term dependencies
- Computationally expensive
- Vanishing/exploding gradient problem
- Cannot process inputs in parallel

![image](https://github.com/aniket-mish/transformers-from-scratch/assets/71699313/65bc8013-dd6e-4d81-9bf8-f813ace2d977)

## Transformers

[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

## Input Embedding
Converts the input tokens into vectors.

## Positional Encoding
Tells the model about the position of a word in the sentence. It is computed once and then reused during training and inference.

## Add & Norm

## FeedForward

## Multi-Head Attention


## References

[1] [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

[2] [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

[3] [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
