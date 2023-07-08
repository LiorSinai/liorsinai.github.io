---
layout: post
title:  "Backpropagation through a layer norm"
date:   2022-05-18
last_modified_at: 2023-07-08
author: Lior Sinai
categories: mathematics
tags: mathematics transformers 'machine learning' 'deep learning'
---

_Derivation of the backpropagation equations for layer normalization._

## Layer normalization

[LayerNorm]: https://arxiv.org/abs/1607.06450
Layer normalization is a normalization over each layer.
In practice it is implemented as a normalization over columns (column major languages like Julia) or rows (row major languages like Python). 
There are other kinds of normalization like batch normalization, which is a normalization across batches.
Interestingly layer norm was only popularised after batch normalization in this [2016 paper][LayerNorm].

The function used for layer norm is:

$$
    Z^l = a^{l}\frac{X^{l}-\mu^{l}}{\sigma^{l}+\epsilon} + b^{l}
$$

where $\mu^l$ and $\sigma^{l}$ are the mean and standard deviation for each layer $l$ respectively, 
$a$ and $b$ are trainable parameters and $\epsilon$ is a small value used for numerical stability.

The mean and standard deviation are calculated as follows:

$$
\begin{align}
    \mu^l &=  \frac{1}{n}\sum_r^n x_r^l \\
    \sigma^l &= \sqrt{\frac{1}{n}\sum^n_r (x_r^l - \mu^l)^2}
\end{align}
$$

### Backpropagation

Let us consider a single layer so that we can drop the $l$ superscript.
For the meantime we will ignore the trainable parameters.
We need to calculate the contribution of the vector $x$ to the final scalar loss $L$.
That is:

$$
    \frac{\partial L}{\partial x} = \frac{\partial z}{\partial x} \frac{\partial L}{\partial z} 
$$

Since $z$ and $x$ are vectors the Jacobian $\frac{\partial z}{\partial x}$ is a matrix.
It is calculated as the partial derivative of each component of $z$ with respect to every component of $x$.
That is:

$$
   \frac{\partial z}{\partial x} =  \begin{bmatrix}
        \frac{\partial z_1}{\partial x_1} & \frac{\partial z_2}{\partial x_1} & \dots & \frac{\partial z_n}{\partial x_1} \\
        \frac{\partial z_1}{\partial x_2} & \frac{\partial z_2}{\partial x_2} & \dots & \frac{\partial z_n}{\partial x_2} \\
        \vdots & \vdots & \ddots & \vdots \\
        \frac{\partial z_1}{\partial x_n} & \frac{\partial z_2}{\partial x_n} & \dots & \frac{\partial z_n}{\partial x_n}
    \end{bmatrix}
$$

Consider $z_1$: 

$$ z_1 = \frac{x_1-\mu}{\sigma+\epsilon} $$

It explicitly depends on $x_1$. 
However it also indirectly depends on all the other components of $x$ because $\sigma$ and $\mu$ are calculated with all of them.
Hence we have a whole column of derivatives instead of one value.

The derivative of $z_k$ with respect to a general component $x_i$ is:

$$
\begin{align}
    \frac{\partial z_k}{\partial x_i} &= \frac{\partial }{\partial x_i}(x_k -\mu)(\sigma + \epsilon)^{-1} \\
        &= (\sigma + \epsilon)^{-1}\frac{\partial }{\partial x_i}(x_k -\mu) + (x_k -\mu)\frac{\partial }{\partial x_i}(\sigma + \epsilon)^{-1} \\
        &= \frac{1}{\sigma + \epsilon}\left(\delta_{ik} - \frac{\partial \mu}{\partial x_i}\right) - 
            (x_k -\mu)(\sigma + \epsilon)^{-2}\left(\frac{\partial \sigma}{\partial x_i} + 0\right) \\
        &= \frac{1}{\sigma + \epsilon}\left(\delta_{ik} - \frac{\partial \mu}{\partial x_i}\right) - 
            \frac{x_k -\mu}{(\sigma + \epsilon)^{2}}\frac{\partial \sigma}{\partial x_i}
\end{align}
$$

using repeat applications of the product rule and chain rule.
The Kronecker delta symbol $\delta_{ik}$ is used for convenience because of that one explicit dependency.
For example the derivative of $z_1$ with respect to $x_1$ has a $1$ instead of a $0$.

$$ 
\begin{align}
    \delta_{i1} &= 
    \begin{cases}
        1 & i=1 \\
        0 & i\neq 1
    \end{cases}
\end{align}
$$

$\frac{\partial z_k}{\partial x_i}$ is dependent on the derivative of the mean:

$$
\begin{align}
    \frac{\partial \mu}{\partial x_i} &= \frac{\partial}{\partial x_i} \frac{1}{n}\sum_r^n x_r \\
                        &= \frac{1}{n}(0 + ... + 0 + 1 + 0 + ... + 0)  \\
                        &= \frac{1}{n}
\end{align}
$$

Intuitively if a component increases by $\Delta x$ it increases the whole mean by $\frac{\Delta x}{n}$.

It is also dependent on the derivative of the standard deviation:

$$
\begin{align}
    \frac{\partial \sigma}{\partial x_i} &= \frac{\partial}{\partial x_i} \left( \frac{1}{n}\sum^n_r (x_r - \mu)^2 \right)^{\frac{1}{2}} \\
        &= \frac{1}{2}\left( \frac{1}{n}\sum^n_r (x_r - \mu)^2 \right)^{-\frac{1}{2}} \frac{\partial}{\partial x_i} \left( \frac{1}{n}\sum^n_r (x_r - \mu)^2 \right) \\
        &= \frac{1}{2\sigma}\left(\frac{1}{n}\sum^n_{r}2(x_r - \mu)
        \left(\delta_{ir} - \frac{\partial \mu}{\partial x_i}\right) \right) \\
        &= \frac{1}{n\sigma}\left((x_i -\mu) - \sum^n_r (x_r -  \mu)\frac{\partial \mu}{\partial x_i} \right)\\
        &= \frac{x_i -\mu}{n\sigma}
\end{align}
$$

Because
$$
\sum^n_r (x_r -  \mu) = \sum^n_r x_r - \mu \sum^n_r 1 = (n\mu) - \mu(n) = 0
$$

This result can also be calculated with algebra and the Taylor series expansion of $\sqrt{1+x}$.

<p>
  <a class="btn" data-toggle="collapse" href="#std-dev-gradient" role="button" aria-expanded="false" aria-controls="collapseExample">
    Algebraic derivation &#8681;
  </a>
</p>
<div class="collapse" id="std-dev-gradient">
  <div class="card card-body ">
    $$
    \begin{align}
    \sigma_\Delta &= \frac{1}{\sqrt{n}}\sqrt{\sum^n_r (x_r - \mu)^2 + (x_i + \Delta x - \mu)^2 - (x_i - \mu)^2} \\
                  &= \frac{1}{\sqrt{n}}\sqrt{\sum^n_r (x_r - \mu)^2 + (\Delta x)^2 +2 \Delta x (x_i - \mu)} \\
                  &\approx \frac{1}{\sqrt{n}}\sqrt{\sum^n_r (x_r - \mu)^2 +2 \Delta x (x_i - \mu)} \quad , (\Delta x)^2 \ll \Delta x \\
                  &= \sqrt{\frac{\sum^n_r (x_r - \mu)^2}{n}} \sqrt{1 + \frac{2 \Delta x (x_i - \mu)}{\sum^n_r (x_r - \mu)^2}} \\
                  &= \sigma \sqrt{1 + \frac{2 \Delta x (x_i - \mu)}{n\sigma^2}} \\\
                  &\approx \sigma \left(1 + \frac{1}{2}\left(\frac{2 \Delta x (x_i - \mu)}{n\sigma^2}\right) \right) + O(h^2) \\
                  &= \sigma + \frac{(x_i - \mu)}{n\sigma}\Delta x
    \end{align}
    $$
  </div>
</div>

In summary (including the trainable parameters $a$ and $b$):

$$
\begin{align}
    \frac{\partial z_k}{\partial x_i} &= a\left(\frac{1}{\sigma + \epsilon}\left(\delta_{ik} - \frac{\partial \mu}{\partial x_i}\right) - 
            \frac{x_k -\mu}{(\sigma + \epsilon)^{2}}\frac{\partial \sigma}{\partial x_i} \right) \\
    \frac{\partial \mu}{\partial x_i} &= \frac{1}{n} \\    
    \frac{\partial \sigma}{\partial x_i} &=  \frac{x_i -\mu}{n\sigma} \\
    \frac{\partial z_k}{\partial a}  &= \frac{x_k -\mu}{\sigma + \epsilon} \\
    \frac{\partial z_k}{\partial b}  &= 1
\end{align}
$$

### Julia implementation

We can get the gradients directly from Flux:
{% highlight julia %}
using Flux
using StatsBase
using LinearAlgebra: I

n = 10
X = rand(n, 8)

model = LayerNorm(n, affine=false)
y, pull = Flux.pullback(model, X)
errors = randn(size(y)...)
grads = pull(errors)
{% endhighlight %}

Flux uses ChainRulesCore to define a `rrule` for backpropagation.
However it doesn't use the final rules derived above; instead it breaks the equations down into pieces much like was done in deriving them.
One disadvantage of this is that you don't get performance gains from terms cancelling out as happened with $\frac{\partial \sigma}{\partial x_i}$.

We can check our final equations match with the following:
{% highlight julia %}
k = 1
means = mean(X; dims=1)
stds = std(X; dims=1, corrected=false)
x = X[:, k]

dμ = ones(n, n) .* 1/n
dσ = (x .- means[k]) / (n * stds[k])
dx = (I(n) - dμ) / (stds[k] + model.ϵ) - 
     dσ * transpose(x .- means[k]) / (stds[k] + model.ϵ)^2
grads_k = dx * errors[:, k] 
{% endhighlight %}

The values in `grads[1][:, k]` and `grads_k` should differ by an order of $10^{-15}$ or less.

