# Learnings from lecture 2

So, neural network training involves 4 steps :

- sample a batch of data from training data
- perform forward pass to get loss
- perform backprop to calculate gradients
- Use those gradients to update parameters so that network classifies images correctly thereby minimizing loss

#### Weight initialization

Refer weight initialization from [lecture-1](./lecture-1.md)

_If we initialize weights with very small values, activations becomes 0. And if initialize them with large values, then activations explodes_

---

## Regularization: Dropout

> [!Note]  
> Before this, understanding Overfitting & underfitting is very important. Not just theory but in-depth. And to understand it, you will need to understand Bias, variance and bias variance tradoff

Understand dropout

watch lecture 6 CS231n andrej lecture "dropout" section
