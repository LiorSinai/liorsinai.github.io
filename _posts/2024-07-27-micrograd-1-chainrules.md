---
layout: post
title:  "MicroGrad.jl: Part 1 ChainRules"
date:   2024-07-27
author: Lior Sinai
last_modified_at: 2024-08-03
sidenav: true
categories: machine-learning
tags: mathematics transformers 'machine learning' 'deep learning'
---

_A series on automatic differentiation in Julia. Part 1 provides an overview and defines explicit chain rules._ 

This is part of a series. The other articles are:
- [Part 2: Automation with expressions][micrograd_expr].
- [Part 3: Automation with IR][micrograd_ir].
- [Part 4: Model demo][micrograd_demo].

[micrograd_chainrules]: {{ "machine-learning/2024/07/27/micrograd-1-chainrules" | relative_url }}
[micrograd_expr]: {{ "machine-learning/2024/08/03/micrograd-2-expr" | relative_url }}
[micrograd_ir]: {{ "machine-learning/2024/08/10/micrograd-3-ir" | relative_url }}
[micrograd_demo]: {{ "machine-learning/2024/07/27/micrograd-4" | relative_url }}
[MicroGrad.jl]: https://github.com/LiorSinai/MicroGrad.jl

All source code can be found at [MicroGrad.jl][MicroGrad.jl].

### Table of Contents

<nav id="toc"></nav>
<script src="/assets/makeTableOfContents.js"></script>

<h2 id="introduction">1 Introduction</h2>

A major convenience of modern machine learning frameworks is automatic differentiation (AD).
Training a machine learning model typically consist of two steps, a forward pass and a backwards pass.
The forward pass takes an input sample and calculates the result. 
Examples include a label in a classifier model or a word or image in a generative model.
In the backward pass, the result is compared to a ground truth sample and the error is backpropagated throughout the model, from the final layers through to the start.
Backpropagation is driven by gradients which are calculated with the differentiation rules of Calculus.

[PyTorch]: https://pytorch.org/
[Flux.jl]: https://fluxml.ai/Flux.jl/stable/

With modern machine learning frameworks, such as [PyTorch][PyTorch] or [Flux.jl][Flux.jl], only the forward pass needs to be defined and they will automatically generate the backward pass. This (1) makes them easier to use and (2) enforces consistency between the forward pass and backward pass.

<figure class="post-figure">
<img class="img-80"
    src="/assets/posts/micrograd/moons_decision_boundary.png"
	alt="Decision boundary"
	>
<figcaption>The decision boundary of a multi-layer perceptron trained on the moon's dataset with MicroGrad.jl.</figcaption>
</figure>


[micograd]: https://github.com/karpathy/micrograd
[micograd_video]: https://www.youtube.com/watch?v=VMj-3S1tku0
[zero_to_hero]: https://karpathy.ai/zero-to-hero.html

Andrej Kaparthy made an excellent [video][micograd_video] where he built a minimal automatic differentiation module called [Micrograd][micograd] in Python.
This is the first video in his [Zero to Hero][zero_to_hero] series.
He later uses it to train a multi-layer perceptron model.
I highly recommend it for anyone who wants to understand backpropagation.

The aim of this series is to create a minimal automatic differentiation package in Julia.
It is based on [Zygote.jl](https://fluxml.ai/Zygote.jl/stable/) and works very differently to the Python AD packages.
The latter are based on objects which implement custom implementations of mathematical operations.
These custom implementations calculate both the forward pass and a derivative for the backward pass.
All operations are only done with these objects.
For example, Micrograd defines a `Value` class that has a custom definition for `__add__`.
The same is true of the `Tensor` objects in PyTorch.
Zygote.jl is instead based on the principle that Julia is a functional programming language. 
It utilises Julia's multiple dispatch feature and its comprehensive metaprogramming abilities to generate new code for the backward pass.
Barring some limitations, it can be used to differentiate all existing functions as well as any custom code.

For the final code, see my [MicroGrad.jl][MicroGrad.jl] repository.

This is a series on backpropagation so naturally a good understanding of Calculus, in particular the rules of differentiation and partial differentiation, is required.

<h2 id="julia-ad-ecosystem">2 Julia AD Ecosystem</h2>

[ChainRules.jl]: https://juliadiff.org/ChainRulesCore.jl/stable/
[ChainRulesCore.jl]: https://github.com/JuliaDiff/ChainRulesCore.jl
[Zygote.jl]: https://fluxml.ai/Zygote.jl/stable/
[ZygoteRules.jl]: https://github.com/FluxML/ZygoteRules.jl
[Zygote_paper]: https://arxiv.org/abs/1810.07951
[IRTools.jl]: https://fluxml.ai/IRTools.jl/latest/
[Functors.jl]: https://fluxml.ai/Functors.jl/stable/

The Julia automatic differentiation ecosystem is centered around three packages: Flux.jl, ChainRules.jl and Zygote.jl.
- [Flux.jl][Flux.jl] is the high level machine learning framework. It uses either ChainRules.jl or Zygote.jl to differentiate code.
- [Zygote.jl][Zygote.jl] implements automatic differentiation through metaprogramming.
  - The main functions it exposes are `gradient`, `withgradient` and `pullback`. The `pullback` function is a light wrapper around `_pullback`, which does most of the heavy lifting.
  - The goal of `_pullback` is to dispatch a function, its arguments and its keyword arguments to a `ChainRule.rrule`. If it cannot, it will inspect the code, decompose it into smaller steps, and follow the rules of differentiation to dispatch  each of those to `_pullback` to recursively find an `rrule`. If this recursive process does not find a valid rule it will raise an error.
  - It also exposes a function called `adjoint`, which calculates the "adjoint of the Jacobian". It is not related to the built-in `adjoint` function in Julia, which calculates the "conjugate transpose". (But there is some type piracy!) The `adjoint` function is the same as ChainRules.jl's `rrule` function except it takes higher precedence in Zygote.jl. If a new differentiation rule needs to be added, the recommendation is to extend `rrule` and not `adjoint`.
  - The core functionality is defined in the minimal [ZygoteRules.jl][ZygoteRules.jl] package.
- [ChainRules.jl][ChainRules.jl] defines forward rules and reverse rules that functions can be dispatched to.
  - The main functions it exposes are `frule` and `rrule`. This series deals only with backpropagation, so it will only concentrate on `rrule`.
  - The core functionality is defined in the minimal [ChainRulesCore.jl][ChainRulesCore.jl] package.

Also important is [IRTools.jl][IRTools.jl], an extended metaprogramming package for working with an intermediate representation (IR) between raw Julia code and lowered code.

As an example, consider the function $f(x) = \sin(\cos(x))$. Using the chain rule of Calculus, it is differentiated as:

$$
\begin{align}
\frac{df}{dx} &= \frac{df}{dh}\frac{dh}{dx} \quad ; h(x)=cos(x)\\
              &= \frac{d}{dh}\sin(h)\frac{d}{dx}\cos(x) \\
              &= \cos(h)(-\sin(x)) \\
              &= -\cos(\cos(x))\sin(x)
\end{align}
$$

We can use `Zygote.withgradient`, exposed as `Flux.withgradient`, to calculate this:

{% highlight julia %}
using Flux
f(x) = sin(cos(x))
y, grad = Flux.withgradient(f, 0.9) # 0.5823, (-0.6368,)
grad[1] == -cos(cos(0.9))*sin(0.9) # true
{% endhighlight %}

More commonly we differentiate with respect to the model, not the data:
{% highlight julia %}
y, grad = Flux.withgradient(m->m(0.9), f) # 0.5823, (nothing,)
{% endhighlight %}

This is more useful for a model with parameters. For example a dense, fully connected layer:
{% highlight julia %}
model = Dense(3=>1)
x = rand(Float32, 3, 10)
y, grad = Flux.withgradient(m->sum(m(x)), model) # 1.5056f0, ((weight=[4.9142 6.235 5.3379],bias=Fill(10.0f0,1),σ=nothing),)
{% endhighlight %}

The aim of the rest of the series is to recreate this functionality.
This first part will focus solely on ChainRules.jl and recreating the `rrule` function.
Part 2 will focus on recreating the `Zygote._pullback` function. 
Part 3 will improve on part 2's solution by using the IRTools.jl package.
Finally, part 4 will extend part 3's solution and showcase simple machine learning examples.


<h2 id="chainrules">3 ChainRules</h2>
<h3 id="chainrules-definition">3.1 Definition</h3>

ChainRules.jl's `rrule` returns the output of the forward pass $f(x)$ and a function $\mathcal{B}$ which calculates the backward pass.
$\mathcal{B}$ takes as input $\Delta = \frac{\partial a}{\partial f}$, the gradient of some function $a$ with regards to the function $f$, and returns a tuple of $\left(\frac{\partial a}{\partial \text{self}}, \frac{\partial a}{\partial x_1}, ..., \frac{\partial a}{\partial x_n}\right)$, the gradient of $a$ with regards to itself and each of the input variables $x_i$.
(The extra gradient $\frac{\partial a}{\partial \text{self}}$ is needed for internal fields and closures.
See the `Dense` layer example above.)
According to the chain rule of Calculus, each gradient is calculated as:

$$
\mathcal{B_i}\left(\frac{\partial a}{\partial f}\right) = \frac{\partial a}{\partial x_i} = \frac{\partial a}{\partial f} \frac{\partial f}{\partial x_i}
$$

As a starting point $\frac{\partial a}{\partial f}=1$ is used to evaluate only $\frac{\partial f}{\partial x}$.

[Jacobian]: https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant

If $x$ and $y=f(x)$ are vectors, then the gradient $J=\frac{\partial f}{\partial x}$ is a [Jacobian][Jacobian]:

$$
J = \begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \dots & \frac{\partial f_1}{\partial x_n}  \\
\vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \dots & \frac{\partial f_m}{\partial x_n} 
\end{bmatrix}
$$

To maintain the correct order, we need to use the [conjugate transpose (adjoint) of the Jacobian](https://juliadiff.org/ChainRulesCore.jl/stable/maths/propagators.html). So the output of $\mathcal{B}_i$ can use be written as:

$$
\mathcal{B_i}(\Delta) = J_i^{\dagger} \Delta
$$

Note the Jacobian does not need to be explicitly need to be explicitly calculated; only the product needs to be. 
This is can be useful when coding the `rrule` for matrix functions.[^softmax]

To start, define a default fallback for `rrule` that returns `nothing` for any function with any number of arguments ([ChainRulesCore](https://github.com/JuliaDiff/ChainRulesCore.jl/blob/a95c181c662ead23aaf9904b8a560bebeb9022a3/src/rules.jl#L131)):

{% highlight julia %}
rrule(::Any, ::Vararg{Any}) = nothing
{% endhighlight %}

An `rrule` can now be defined for any function.
For it to be really useful however, `rrule` must cover a large set of functions.
Thankfully ChainRules.jl provides us with that.
In this post I'll only work through a limited set of examples: basic arithmetic, trigonometry and polynomials.

<h3 id="chainrules-arithmetic">3.2 Arithmetic</h3>

The simplest functions are the basic arithmetic functions: addition, subtraction, multiplication and division.

The derivatives of adding two variables is:

$$
\frac{\partial}{\partial x}(x+y) = 1 + 0; \frac{\partial}{\partial y}(x+y) = 0 + 1
$$

So the incoming gradient $Δ$ is passed through to the next layer.
There are no internal fields so $\frac{\partial a}{\partial \text{self}}$ is `nothing`.
We can return $\mathcal{B}$ as an anonymous function, but giving it the name `add_back`  helps with debugging ([ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl/blob/dba6cb57d73ba837c5ab6fd1f968f3a5d301ca9c/src/rulesets/Base/fastmath_able.jl#L167)).

{% highlight julia %}
function rrule(::typeof(+), x::Number, y::Number)
    add_back(Δ) = (nothing, true * Δ, true * Δ) # ∂self, ∂x, ∂y
    x + y, add_back
end
{% endhighlight %}

Usage:
{% highlight julia %}
z, back = rrule(+, 1, 2) # (3, var"#add_back#3"())
back(1.2) # (nothing, 1.2, 1.2)
{% endhighlight %}

Subtraction is almost identical:
{% highlight julia %}
function rrule(::typeof(-), x::Number, y::Number)
    minus_back(Δ) = (nothing, true * Δ, -1 * Δ) # ∂f, ∂x, ∂y
    x - y, minus_back
end
{% endhighlight %}

With multiplication, the incoming gradient is multiplied by the other variable:

$$
\frac{\partial}{\partial x}(xy) = y; \frac{\partial}{\partial y}(xy) = x
$$

In code ([ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl/blob/dba6cb57d73ba837c5ab6fd1f968f3a5d301ca9c/src/rulesets/Base/fastmath_able.jl#L254)):

{% highlight julia %}
function rrule(::typeof(*), x::Number, y::Number)
    times_back(Δ) = (nothing, y * Δ, x * Δ) # ∂self, ∂x, ∂y
    x * y, times_back
end
{% endhighlight %}

Note that Julia will create a _closure_ around the incoming `x` and `y` variables for `times_back`.
A closure is when the function stores the values of variables from its parents scope (it closes over the variables).
In other words, `x` and `y` will become constants in the `times_back` scope.
In this way, the `times_back` function will always "remember" what values it was called with:

Example:
{% highlight julia %}
z, back = rrule(*, 2, 3) # (6, var"#times_back#4"{Int64, Int64}(2, 3))
back.x # 2
back.y # 3
back(1.2) # (nothing, 3.6, 2.4)
{% endhighlight %}

Every call to `rrule` with `*` will return a different `back` function based on the input arguments.

Division is slightly different in that the derivatives look different for $x$ and $y$:

$$
\frac{\partial}{\partial x}\frac{x}{y} = \frac{1}{y}; \frac{\partial}{\partial y}\frac{x}{y}= -\frac{x}{y^2}
$$

Here we can calculate an internal variable `Ω` to close over, and use it for the $\frac{\partial}{\partial y}$ derivative ([ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl/blob/dba6cb57d73ba837c5ab6fd1f968f3a5d301ca9c/src/rulesets/Base/fastmath_able.jl#L169)):
{% highlight julia %}
function rrule(::typeof(/), x::Number, y::Number)
    Ω = x / y
    divide_back(Δ) = (nothing, 1 / y * Δ, -Ω/y * Δ) # ∂self, ∂x, ∂y
    Ω, divide_back
end
{% endhighlight %}

Example:
{% highlight julia %}
z, back = rrule(/, 2, 3) # (0.6667, var"#divide_back#5"{Int64, Float64}(3, 0.6667))
back.Ω # 0.6667
back.y # 3
back.x # ERROR
back(1.2) # (nothing, 0.4, -0.2667)
{% endhighlight %}


<h3 id="chainrules-trigonometry">3.3 Trigonometry</h3>

The derivatives of $\sin$ and $\cos$ are:

$$
\begin{align}
  \frac{\partial}{\partial x} \sin(x) &= \cos(x) \\
  \frac{\partial}{\partial x} \cos(x) &= -\sin(x)
\end{align}
$$

Because both use $\sin$ and $\cos$, we can use `sincos` to calculate both simultaneously and more efficiently than calculating each on its own. This shows the advantage of calculating the forward pass and backward pass at the same time ([ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl/blob/dba6cb57d73ba837c5ab6fd1f968f3a5d301ca9c/src/rulesets/Base/fastmath_able.jl#L12)):
{% highlight julia %}
function rrule(::typeof(sin), x::Number)
    s, c = sincos(x)
    sin_back(Δ) = (nothing, Δ * c) # ∂self, ∂x
    s, sin_back
end

function rrule(::typeof(cos), x::Number)
    s, c = sincos(x)
    cos_back(Δ) = (nothing, -Δ * s) # ∂self, ∂x
    c, cos_back
end
{% endhighlight %}

Let's now revisit the example from earlier, $f(x) = \sin(\cos(x))$.
We have the forward pass:

$$
\begin{align}
y_1 &= \cos(x) \\
y_2 &= \sin(y_1)\\
\end{align}
$$

And the backwards pass:

$$
\begin{align}
\frac{\partial y_2}{\partial y_1} &= (1.0)  \frac{\partial}{\partial y_1} \sin(y_1) \\
            &= \cos(y_1) \\
\frac{\partial y_2}{\partial x} &= \frac{\partial y_2}{\partial y_1} \frac{\partial}{\partial x} \cos(x) \\
         &= -\Delta_2 \sin(x)
\end{align}
$$

In code:
{% highlight julia %}
x = 0.9
y1, back1 = rrule(cos, x) # 0.6216, cos_back
y2, back2 = rrule(sin, y1) # 0.5823, sin_back
grad_sin, grad_y1 = back2(1.0) # nothing,0 .8129
grad_cos, grad_x = back1(grad_y1) # (nothing, -0.6368)
grad_x == -cos(cos(x))*sin(x) # true
{% endhighlight %}

<h3 id="chainrules-polynomial">3.4 Polynomials</h3>

The next section will use the `rrule` for polynomial curve fitting.
This requires an `rrule` for the `evalpoly` function.

For a general polynomial:

$$
y = a_0 + a_1x + a_2x^2 + ... + a_n x^n
$$

The derivatives are:

$$
\begin{align}
\frac{\partial y}{\partial x} &= 0 + a_1 + 2a_2x^1 + ... + n a_n x^{n-1} \\
\frac{\partial y}{\partial a_i} &= 0 + ... + x^{i} + ... + 0
\end{align}
$$

For the most efficiency implementation, the powers of $x$ can be calculated for both the forward and backwards pass at the same time.
For simplicity, I'm not going to do that ([ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl/blob/dba6cb57d73ba837c5ab6fd1f968f3a5d301ca9c/src/rulesets/Base/evalpoly.jl)):

{% highlight julia %}
function rrule(::typeof(evalpoly), x, coeffs::AbstractVector)
    y = evalpoly(x, coeffs)
    function evalpoly_back(Δ)
        xpow = one(x)
        dp = similar(coeffs, typeof(xpow * Δ))
        dx = zero(x)
        for i in eachindex(coeffs)
            dp[i] = Δ * xpow
            dx += (i-1) * coeffs[i] * xpow / x * Δ
            xpow *= x
        end
        return nothing, dx, dp
    end
    y, evalpoly_back
end
{% endhighlight %}

Usage:
{% highlight julia %}
y, back = rrule(evalpoly, 1.2, [2.0, 0.0, 3.0, 4.0]) # 13.232, evalpoly_back
back(1.0) # (nothing, 24.48, [1.0, 1.2, 1.44, 1.728]) 
{% endhighlight %}

<h3 id="chainrules-mse">3.5 MSE</h3>

The mean square error (MSE) will also be used for polynomial curve fitting.
It is:

$$
  MSE(\hat{y}, y) = \frac{1}{n}\sum^n_{i=1} (\hat{y}_i - y_i)^2
$$

with derivatives:

$$
\begin{align}
  \frac{\partial MSE}{\partial \hat{y}_i} &= \frac{1}{n}(0 + ... + 2(\hat{y}_i - y_i) + ... + 0) \\
        &= \frac{2(\hat{y}_i - y_i)}{n} \\
  \frac{\partial MSE}{\partial y_i} &= \frac{1}{n}(0 + ... - 2(\hat{y}_i - y_i) + ... + 0) \\
       &= -\frac{2(\hat{y}_i - y_i)}{n}
\end{align}
$$

In code it is:
{% highlight julia %}
using StatsBase
mse(ŷ::AbstractVecOrMat, y::AbstractVecOrMat) = mean(abs2.(ŷ - y))
{% endhighlight %}

Flux.jl does not define an `rrule` for its `mse` because it can be broken down into functions which already have an `rrule` (`-`, `broadcast`, `abs2` and `mean`). 
However since we don't have `rrule`s for these parts and have not yet automated decomposing it, it is simplest to create an `rrule` for the entire function: 

{% highlight julia %}
function rrule(::typeof(mse), ŷ::AbstractVecOrMat, y::AbstractVecOrMat)
    Ω = mse(ŷ, y)
    function mse_back(Δ)
        c = 2 * (ŷ - y) / length(y) * Δ
        return nothing, c, -c # ∂self, ∂ŷ, ∂y
    end
    Ω, mse_back
end
{% endhighlight %}

The `mse` can also be applied per individual data point and summed up separately.
This form is not common but will be useful for explanatory purposes in the polynomial curve fitting section:

{% highlight julia %}
mse(ŷ::Number, y::Number, n::Int) = abs2(ŷ - y)/n
function rrule(::typeof(mse), ŷ::Number, y::Number, n::Int)
    Ω = mse(ŷ, y, n)
    function mse_back(Δ)
        c = 2 * (ŷ - y) / n * Δ
        return nothing, c, -c # ∂self, ∂ŷ, ∂y
    end
    Ω, mse_back
end
{% endhighlight %}


<h2 id="gradient-descent">4 Gradient Descent</h2>
<h3 id="polynomial-curve-fitting">4.1 Polynomial curve fitting</h3>

Gradient descent is a great algorithm to illustrate the usefulness of the code developed so far.
The toy example of fitting a polynomial to data will be used.
This is a useful example because (1) we can start with a target curve and so have ground truth values to compare and (2) this problem can be solved analytically without gradients.

<figure class="post-figure">
<img class="img-80"
    src="/assets/posts/micrograd/polyfit_data.png"
	alt="Polynomial with noise"
	>
<figcaption></figcaption>
</figure>

Here is code to create the above data:
{% highlight julia %}
using StatsBase
target_weights = [15.0, -2.1, 13.9, 1.5]
noise_factor = 0.2
xs = (rand(100) .- 0.5) .* 10
ys = map(x -> evalpoly(x, target_weights), xs)
scale_factor = mean(abs.(ys))
ys .+= randn(length(ys)) * scale_factor * noise_factor
{% endhighlight %}

<p>
  <a class="btn" data-toggle="collapse" href="#poly-fit-analytical" role="button" aria-expanded="false" aria-controls="collapseExample">
    Analytical least squares fitting of polynomials &#8681;
  </a>
</p>
<div class="collapse" id="poly-fit-analytical">
  <div class="card card-body ">
		<p> For a polynomial of order $p$, if there are exactly $n=p+1$ training samples (including for the constant $a_0$) than there exactly $n$ equations for $n$ unknowns ($a_0$,...,$a_p$) and this can be solved as an ordinary linear system:
        $$
            \begin{align}
            &a_0 + a_1 x_1 + a_2x_1^2 + ... + a_p x_1^p = y_1 \\
            &\vdots \\
            &a_0 + a_1 x_n + a_2x_n^2 + ... + a_p x_n^p = y_n \\
            &\Rightarrow \begin{bmatrix}
            1 & x_1 & x_1^2 & \cdots & x_1^p \\
            \vdots & \vdots & \vdots & \ddots & \vdots \\
            1 & x_n & x_n^2 & \cdots & x_n^p
            \end{bmatrix}
            \begin{bmatrix}
            a_0 \\
            \vdots \\
            a_n
            \end{bmatrix}
            =
            \begin{bmatrix}
            y_1 \\
            \vdots \\
            y_n
            \end{bmatrix} \\
            &\Rightarrow XA=Y \\
            &\Rightarrow A = X^{-1}Y
            \end{align}
        $$
        Where $X^{-1}$ usually exists because $X$ is a square matrix.
        </p>
        <p>
        However usually $n > p + 1$ and thus $X^{-1}$ will not exist. In that case the pseudoinverse $X^+$, also called the <a href="https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse">Moore-Penrose inverse</a>, can be used instead:
        $$
            \begin{align}
            X^{+} &= (X^T X)^{-1} X^T \\
            \Rightarrow A &= X^{+}Y
            \end{align}
        $$
        It can be proven that this solution for $A$ minimises the least squared error.  
        </p>
        <p>
        Here is this solution in code:
{% highlight julia %}
using LinearAlgebra
function solve_poly_linear(order::Int, xs::AbstractVector, ys::AbstractVector)
    n = length(xs)
    X = zeros(n, order + 1)
    for (i, x) in enumerate(xs)
        xpow = 1
        for j in 1:(size(X, 2))
            X[i, j] = xpow
            xpow *= x
        end
    end
    pinv(X) * ys
end
{% endhighlight %}
        </p>
  </div>
</div>

Here is a simple version of gradient descent:
<blockquote>
<u><b>Gradient descent</b></u> <br>
<b>while</b> (criteria is not met) <b>do</b>:<br>
$\quad$ $\Delta = 0$ <br>
$\quad$ <b>for</b> sample, label in train_set <b>do</b>: <br>
$\quad\quad$ $\Delta \leftarrow \Delta + \frac{\partial}{\partial\theta_j}L$($m_{\theta_j}$(sample), label) <br>
$\quad$ $\theta_{j+1}$ $\leftarrow \theta_j - \alpha \Delta$
</blockquote>

where $m_\theta$ is the model with parameters $\theta$ and $L$ is the loss function.

This is Julia implementation for specifically applying the algorithm to polynomials.
The stopping condition is a maximum number of iterations, so the `while` loop has been replaced with a `for` loop.
The code also saves the loss so that the training progress can be analysed.

{% highlight julia %}
function gradient_descent_poly!(
    coeffs::AbstractVector,
    xs::AbstractVector,
    ys::AbstractVector
    ; learning_rate::AbstractFloat=0.1,
    max_iters::Integer=100
    )
    history = Float64[]
    n = length(xs)
    p = length(coeffs)
    for i in 1:max_iters
        loss_iter = 0.0
        Δcoeffs = zeros(p)
        for (x, y) in zip(xs, ys)
            # forward
            ŷ, back_poly = rrule(evalpoly, x, coeffs)
            loss_x, back_loss = rrule(mse, ŷ, y, n)
            # reverse
            Δloss, Δŷ, Δy = back_loss(1.0)    
            Δevalpoly, Δx, Δcoeffs_x = back_poly(Δŷ)
            # accumulate
            loss_iter += loss_x
            Δcoeffs += Δcoeffs_x
        end
        # update
        coeffs .-= learning_rate .* Δcoeffs
        # history
        push!(history, loss_iter)
    end
    history
end
{% endhighlight %}

Calling the code:
{% highlight julia %}
coeffs = rand(4)
history = gradient_descent_poly!(coeffs, xs, ys; learning_rate=1e-5, max_iters=2000)
{% endhighlight %}

Plotting the history:

<figure class="post-figure">
<img class="img-80"
    src="/assets/posts/micrograd/polyfit_training.png"
	alt="Gradient descent training history"
	>
<figcaption></figcaption>
</figure>

Comparing losses:

<table><thead>
  <tr>
    <th>Method</th>
    <th>Loss</th>
    <th>Coefficients</th>
  </tr></thead>
<tbody>
  <tr>
    <td>Target</td>
    <td>416.62</td>
    <td>(15.0, -2.1, 13.9, 13.9, 1.5)</td>
  </tr>
  <tr>
    <td>Analytical</td>
    <td>391.64</td>
    <td>(15.34, -3.24, 13.84, 1.46)</td>
  </tr>
  <tr>
    <td>Gradient Descent</td>
    <td>498.50</td>
    <td>(1.37, 0.54, 14.51, 1.26)</td>
  </tr>
</tbody>
</table>

And finally, comparing the curves:

<figure class="post-figure">
<img class="img-80"
    src="/assets/posts/micrograd/polyfit.png"
	alt="Fitted polynomial curves"
	>
<figcaption></figcaption>
</figure>

<h3 id="gradient-descent-map">4.2 Revisited with map</h3>

Its possible to replace the inner loop over the training data with `map`.

{% highlight julia %}
function gradient_descent_poly!(
    coeffs::AbstractVector,
    xs::AbstractVector,
    ys::AbstractVector
    ; learning_rate::AbstractFloat=0.1,
    max_iters::Integer=100
    )
    history = Float64[]
    for i in 1:max_iters
        # forward
        ys_and_backs = map(x->rrule(evalpoly, x, coeffs), xs)
        ŷ = map(first, ys_and_backs)
        loss_iter, back_loss = rrule(mse, ŷ, ys)
        # reverse
        Δmse, Δŷ, Δy = back_loss(1.0)
        ∂f_and_∂x_zipped = map(((_, pb), δ) -> pb(δ), ys_and_backs, Δŷ)
        Δcoeffs_unzipped = map(Δ->Δ[3], ∂f_and_∂x_zipped) # Δ[i] = (Δevalpoly, Δx, Δcoeffs)
        Δcoeffs = reduce(+, Δcoeffs_unzipped)
        # update
        coeffs .-= learning_rate .* Δcoeffs
        # history
        push!(history, loss_iter)
    end
    history
end
{% endhighlight %}

This is code is slightly more complex than the previous version.
The behaviour and performance is practically identical.
However, it is one step closer to being more generic.

In machine learning model usually execute on batch data.
We could make a polynomial model that does that:
{% highlight julia %}
struct Polynomial{V<:AbstractVector}
    weights::V
end
(m::Polynomial)(x) = evalpoly(x, m.weights)
(m::Polynomial)(x::AbstractVector) = map(x -> evalpoly(x, m.weights), x)
{% endhighlight %}

The goal then is to get gradients for the model's weights directly:

{% highlight julia %}
model = Polynomial(coeffs)
zs, back = pullback(m -> m(xs), model)
{% endhighlight %}

In the next sections we will write code that can inspect the model function call, recognise that it calls `map`, and calls a `pullback` for map.

Note it will not be calling an `rrule` for map. This is design choice. Both `rrule` and `pullback` have the same outputs. However `rrule` is intended for small, stand alone gradients, whereas `pullback` will potentially involve recursive calls to itself. 
With `map`, the next recursive call will be `pullback(evalpoly, xs)` which will directly pass through the arguments to our existing `rrule`. However in cases where an `rrule` is not defined it will have to do further code inspection.

<h2 id="conclusion">5 Conclusion</h2>

The next two sections will develop `pullback` for inspecting and decomposing code in order to pass each piece to `rrule`. 

[Part 2][micrograd_expr] will introduce metaprogamming Julia and generate expressions for the backpropagation code. 
However the code is unstable and prone to errors - it is recursive metaprogramming - so [part 3][micrograd_ir] will introduce more robust code making use of the [IRTools.jl][IRTools.jl] package. 
This code really pushes Julia's metaprogramming to its limits.

It is possible to jump straight to [part 3][micrograd_ir] if desired.

---

[^softmax]: As an example, see how NNLib.jl calculates the derivative of the [softmax](https://github.com/FluxML/NNlib.jl/blob/381a41f2b912eb924f1a6c256a9bf49a593e4b67/src/softmax.jl#L72). Instead of calculating the Jacobian and then $J^{\dagger}\Delta$, they calculate the product in a single step using a `broadcast` and `sum`.