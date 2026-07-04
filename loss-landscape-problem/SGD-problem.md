## Why SGD is so slow (Lecture explaination)

> Loss surfaces can be shaped unevenly. SGD struggles when one direction is steep and another is flat, because one learning rate can't handle both well. That's why Adam and batch norm exist.

To understand deeply about this, refer this [explaination](SGD-problem.md)

Also, read about research paper [Visualizing the Loss Landscape of Neural Nets](https://arxiv.org/pdf/1712.09913)

**Solution is Momentum SGD**

Other optimization technique is **AdaGrad Update**

> Read about Momentum SGD, AdaGrad Update, RMSProp & Adam.

All optimization techniques use learning rate as hyperparameter.

> [!Note]  
> **Adam is nice default choice to go with**  
> **Learning rate decay is best solution with this technique. So we should use learning rate decay**

**_Momentum SGD, AdaGrad Update, RMSProp & Adam is first order optimization technique because they only uses gradient information for optimization. Meaning they only use gradients or slope_**.

**Second order methods**

- Taylor expansion
- Newton parameter update

**They Don't use Learning rate.**

**They use gradient & Hessian (how curve the surface is) and directly jumps to target**

Good thing about them is they don't have hyperparameter  
But,  
**In practice, we DON'T use this ones**

---

## Why SGD is Slow — The Loss Landscape Problem (Explained by claude)

## 1. The Loss Surface

When a neural network has multiple weights (say `w1` and `w2`), the loss is a function of all of them. For every pair of values `(w1, w2)`, you get one loss number.

If you plot this — `w1` on one axis, `w2` on another, and loss as height — you get a **3D surface**. This is called the **loss landscape** or **loss surface**.

> This is different from the simple 2D loss curve you see with one weight. Two weights → 3D surface. Real networks have millions of weights, so the surface lives in very high dimensions — but the 2D intuition still holds.

---

## 2. The Shape of the Surface Matters

Not all loss surfaces are shaped the same. A round, symmetric bowl is ideal. But in practice, the surface is often shaped like a **long narrow valley** — elongated in one direction.

When you look at this surface from above (a contour plot), the level-set rings look like stretched ovals, not circles.

```
      w2
       ↑
       |   (  (  ( [minimum] )  )  )
       |
       +-------------------------→ w1
```

The rings are stretched wide along `w1` and narrow along `w2`. This shape directly tells you something important about how loss changes in each direction.

---

## 3. What "Steep" and "Shallow" Actually Mean

Take any point on this surface and ask: _if I nudge a weight slightly, how much does the loss change?_

**Moving along w1 (horizontal direction):**
The rings are spread far apart. You can move a large distance and barely cross into the next ring. Loss changes very little → **small gradient for w1**.

**Moving along w2 (vertical direction):**
The rings are packed tightly together. A small move crosses many rings at once. Loss changes a lot → **large gradient for w2**.

This is all "steep vs shallow" means:

- Steep = loss changes a lot per unit step = large gradient
- Shallow = loss changes very little per unit step = small gradient

---

## 4. The SGD Update Rule

SGD updates every weight using the same formula:

```
new_w = old_w − lr × gradient
```

One learning rate `lr`, applied to all weights.

---

## 5. Why This Causes Slow Convergence

With an elongated valley:

- **w2 has a large gradient** → `lr × large_gradient` = huge step → SGD **overshoots** the valley wall → bounces back → overshoots again → zig-zag
- **w1 has a small gradient** → `lr × small_gradient` = tiny step → SGD **crawls** toward the minimum

Now you're stuck in a contradiction:

| If you...                                    | Then...                                |
| -------------------------------------------- | -------------------------------------- |
| Increase `lr` to move faster in w1 direction | w2 steps become even larger → diverges |
| Decrease `lr` to stop w2 from oscillating    | w1 steps become even tinier → crawls   |

**One learning rate cannot fit both directions at the same time.**

The result: SGD zig-zags violently in the steep direction while making painfully slow progress in the shallow direction — where the actual minimum lies.

---

## 6. The Core Insight

> The problem is not gradient descent itself. The problem is using a single learning rate for all weights, on a surface that behaves very differently in different directions.

---

## 7. How This Gets Fixed

This problem directly motivates two important ideas you will encounter next:

**Batch Normalization** — reshapes the loss surface itself to be less elongated, so all directions have roughly similar curvature. SGD suffers less because the surface is closer to a round bowl.

**Adam optimizer** — keeps a separate effective learning rate per weight, adapted based on the history of gradients. Where the gradient is large (steep direction), the effective lr shrinks. Where the gradient is small (shallow direction), the effective lr grows. This directly solves the one-lr-fits-all problem.
