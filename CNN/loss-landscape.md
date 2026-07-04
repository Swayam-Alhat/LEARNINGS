# Yt video of Tom Goldstein - Loss landscape..

- Non-convex functions are formed by combining or multiplying function that contains operations like log, exponentiation or sine. We know functions or operation like sin, cos, log, exp generates multiple peaks , valleys.  
  Non convex functions can be formed by combining multiple non-linear functions

**What is minimizers**

```
In Artificial Neural Networks (ANNs), minimizers refer to the set of weights that yield the lowest possible training loss. The distinction between flat and sharp minimizers describes the shape of the loss landscape around these optimal weight configurations.

Sharp Minimizers

- Definition: The training loss decreases to a minimum, but the surrounding loss surface curves upward very steeply.
- Behavior: Even a microscopic adjustment to the network weights or a slight change in the input data (noise) causes a massive spike in the loss.
- Generalization: These networks are highly sensitive to perturbations, meaning they have often simply "memorized" the training data rather than learning general patterns. They tend to perform poorly on unseen test data (poor generalization).
  Flat Minimizers
- Definition: The training loss settles into a minimum that sits at the bottom of a wide, smooth, and gentle valley.
- Behavior: The loss remains uniformly low even if the network weights are noticeably shifted or altered.
- Generalization: These networks are robust to noise and slight variations in the data. They are generally preferred because they consistently perform well on unseen test data.

The Role of Optimization

Standard optimizers like Stochastic Gradient Descent (SGD) can sometimes accidentally converge into sharp minima. To prevent this, deep learning researchers use specialized algorithms like Sharpness-Aware Minimization (SAM) or Stochastic Weight Averaging (SWA) to actively guide the network towards flat, stable minima
```

- SGD and small batches gives us flat minimizers (**Good**)
- Big batches and ADAM gives us sharp minimizers (**Bad**)

- **filter normalization** process helps us to know which optimizer is better by... At **14:23 video part**, he discuss filter normalization and how it helps to compare multiple optimizers

> [!Note]  
> **filter normalization** is way to compare optimizers and see which one is better. It gives us curvature..

- loss surface is very complex. adding skip connections makes it smooter and simple. this simplifies the training of deep neural nets.

- **Skip connections radically change behaviour of loss surface**

**_State-of-the-art modern nets uses skip connections to get good results_**

- As layers get increase, loss landscape becomes more complex and non-convex

- DenseNet-120 have very sophisticated set of skip connections that it has very smooth loss surface. Almost similarly to parabola
