---
layout: post
title:  "MicroGrad.jl: Part 4 Extensions"
date:   2024-08-17
author: Lior Sinai
last_modified_at: 2024-08-17
background: '/assets/posts/micrograd/matrix_green.jpg'
background-caption: 'https://wallpapers.com/'
sidenav: true
categories: machine-learning
tags: mathematics transformers 'machine learning' 'deep learning'
---

_A series on automatic differentiation in Julia. Part 4 extends part 3 to handle maps, getfield and anonymous functions. It creates a generic gradient descent and uses this to fit a polynomial._ 

This is part of a series. The other articles are:
- [Part 1: ChainRules][micrograd_chainrules].
- [Part 2: Automation with expressions][micrograd_expr].
- [Part 3: Automation with IR][micrograd_ir].
- [Part 5: MLP][micrograd_mlp].

[micrograd_chainrules]: {{ "machine-learning/2024/07/27/micrograd-1-chainrules" | relative_url }}
[micrograd_expr]: {{ "machine-learning/2024/08/03/micrograd-2-expr" | relative_url }}
[micrograd_ir]: {{ "machine-learning/2024/08/10/micrograd-3-ir" | relative_url }}
[micrograd_ext]: {{ "machine-learning/2024/08/17/micrograd-4-ext" | relative_url }}
[micrograd_mlp]: {{ "machine-learning/2024/08/19/micrograd-5-mlp" | relative_url }}
[MicroGrad.jl]: https://github.com/LiorSinai/MicroGrad.jl

All source code can be found at [MicroGrad.jl][MicroGrad.jl].

### Table of Contents

<nav id="toc"></nav>
<script src="/assets/makeTableOfContents.js"></script>

<h2 id="introduction">1 Introduction</h2>

At the end of part 3 it was established that the code developed so far fails for the polynomial model:
{% highlight julia %}
struct Polynomial{V<:AbstractVector}
    weights::V
end
(m::Polynomial)(x) = evalpoly(x, m.weights)
(m::Polynomial)(x::AbstractVector) = map(m, x)
model = Polynomial([3.0, 2.0, -3.0, 1.0])
x = [1.0, 2.0, 3.0, 4.0]
pullback(model, x) # ERROR: No method found for Tuple{typeof(fieldtype) ....}
{% endhighlight %}

Calling `@code_ir model(x)`, we can see that code is lowered as follows:

{% highlight plaintext %}
1: (%1, %2)
  %7 = Main.map(%6, %2)
  return %7
{% endhighlight %}

And further that `model(1.0)` is lowered to:
{% highlight plaintext %}
1: (%1, %2)
  %3 = Base.getproperty(%1, :weights)
  %4 = Main.evalpoly(%2, %3)
  return %4
{% endhighlight %}

We could have also defined the `map` using an anonymous function:
{% highlight plaintext %}
(m::Polynomial)(x::AbstractVector) = map(x->evalpoly(x, m.weights), x)
{% endhighlight %}

In which case it would have been lowered to:
{% highlight plaintext %}
1: (%1, %2)
  %3 = Main.:(var"#43#44")
  %4 = Core.typeof(%1)
  %5 = Core.apply_type(%3, %4)
  %6 = %new(%5, %1)
  %7 = Main.map(%6, %2)
  return %7
{% endhighlight %}

The calls to `Core.typeof` and `Core.apply_type` are in the list of ignored functions.
However we need to handle `map`, `getproperty` and `%new`.
These sort of functions do not have formal mathematical derivatives and so they do not have `rrule`s in ChainRules.jl.
Instead, Zygote.jl handles these functions with their own custom pullbacks.
Zygote also replaces some low level functions like `new`, `getproperty` and `getindex` entirely with custom code.

<h2 id="extending-pullback">2 Extending pullback</h2>
<h3 id="pullback-map">2.1 map</h3>

The pullback for `map` is fairly complex. What will be presented here is a simplified version.
It might also help to look at the less generic code in the example in [part 1](http://localhost:4000/machine-learning/2024/07/27/micrograd-1-chainrules.html#gradient-descent-map).

Consider the following code:
{% highlight julia %}
f(x) = sin(x)
x = [0.1, 0.2, 0.5]
map(f, x)
{% endhighlight %}

The `pullback` for `map` should return 3 values: $\text{s̄elf}$ for `map`, $\bar{f}$ for the function `f` and $\bar{x}$ for each value in `x`.

The code will start by getting pullbacks for each value in `x`:
{% highlight julia %}
ys_and_backs = map((xs...) -> pullback(f, xs...), x) # ((0.099, Pullback), (0.198, Pullback), (0.479, Pullback))
{% endhighlight %}

This list is in a "zipped" format: there are $n$ entries of $(y_i, \mathcal{B}_i)$ for an array length $n$.
This will be unzipped into two lists each of length $n$: $(y_1,...,y_n), (\mathcal{B}_1,...,\mathcal{B}_n)$:

{% highlight julia %}
Δ = ones(length(x))
ys = map(first, ys_and_backs) # (0.099, 0.198, 0.479)
∂f_and_∂x_zipped = map(((_, pb), δ) -> pb(δ), ys_and_backs, Δ) # ((nothing, 0.995), (nothing, 0.980), (nothing, 0.877))
{% endhighlight %}

The gradients lists of $n$ entries

$$
(\text{s̄elf}_1, \bar{x}_{11}, ..., \bar{x}_{k1}), ...,(\text{s̄elf}_n, \bar{x}_{1n}, ..., \bar{x}_{kn})
$$ 

needs to be further unzipped into $k+1$ lists entries for $\text{s̄elf}$ and $k$ arguments: 

$$
(\text{s̄elf}_1,...,\text{s̄elf}_{n}), (\bar{x}_{11},...,\bar{x}_{1n}), ... (\bar{x}_{k1},...,\bar{x}_{kn})
$$

This is done with an `unzip` function which generalises `first` to any index `i` ([source](https://github.com/FluxML/Zygote.jl/blob/3c3325d9987931f15bd478c932332be19c316de4/src/lib/array.jl#L137)):
{% highlight julia %}
struct StaticGetter{i} end
(::StaticGetter{i})(v) where {i} = v[i]
(::StaticGetter{i})(::Nothing) where {i} = nothing

function _unzip(tuples, ::Val{N}) where {N}
  getters = ntuple(n -> StaticGetter{n}(), N)
  map(g -> map(g, tuples), getters)
end

function unzip(tuples)
  N = length(first(tuples))
  _unzip(tuples, Val(N))
end
{% endhighlight %}

The result:
{% highlight julia %}
∂f_and_∂x = unzip(∂f_and_∂x_zipped) # [nothing, nothing, nothing], [0.995, 0.98, 0.877]
{% endhighlight %}

As a final step, all the gradients for the function are accumulated into one value:
{% highlight julia %}
∂f = reduce(accum, ∂f_and_∂x[1]) # nothing
{% endhighlight %}

Putting all this code in a single function ([source](https://github.com/FluxML/Zygote.jl/blob/3c3325d9987931f15bd478c932332be19c316de4/src/lib/array.jl#L185)):

{% highlight julia %}
function pullback(::typeof(map), f::F, args::Vararg{Any, N}) where {F, N}
    ys_and_backs = map((xs...) -> pullback(f, xs...), args...)
    ys = map(first, ys_and_backs)
    function map_pullback(Δ)
      # technically should apply f in reverse and reverse back afterwards in case f is stateful
      ∂f_and_∂x_zipped = map(((_, pb), δ) -> pb(δ), ys_and_backs, Δ)
      ∂f_and_∂x = unzip(∂f_and_∂x_zipped) 
      ∂f = reduce(accum, ∂f_and_∂x[1])
      ∂args = ∂f_and_∂x[2:end]
      return (nothing, ∂f, ∂args...)
    end
    ys, map_pullback
end
{% endhighlight %}

Testing:

{% highlight julia %}
x = [0.1, 0.2, 0.5]
z, back = pullback(map, sin, x) 
back(ones(length(x))) # (nothing, nothing, [0.995, 0.98, 0.877])
{% endhighlight %}

And also:
{% highlight julia %}
f(a,b)=a/(a+b*b)
z, back = pullback(map, f, [2.0, 4.0], [3.0, 5.0]) 
back([1.0, 1.0]) # (nothing, nothing, [0.074, 0.029], [-0.099, -0.047])
{% endhighlight %}


<h3 id="pullback-instrument">2.2 Instrument</h3>

Zygote.jl modifies some of the source code before creating the primal and reverse passes.
Here is a simplified version of this `instrument` function which only replaces `new` and `getfield` ([source](https://github.com/FluxML/Zygote.jl/blob/3c3325d9987931f15bd478c932332be19c316de4/src/compiler/reverse.jl#L121)):

{% highlight julia %}
function instrument(ir::IR)
    pr = Pipe(ir)
    for (v, st) in pr
        ex = st.expr
        if isexpr(ex, :new)
            pr[v] = xcall(Main, :__new__, ex.args...)
        elseif is_literal_getfield(ex)
            pr[v] = xcall(Main, :literal_getfield, ex.args[2], Val(unwrapquote(ex.args[3])))
        end
    end
    finish(pr)
end

iscall(x, m::Module, n::Symbol) = isexpr(x, :call) && x.args[1] == GlobalRef(m, n)
unwrapquote(x) = x
unwrapquote(x::QuoteNode) = x.value

is_literal_getfield(ex) =
  (iscall(ex, Core, :getfield) || iscall(ex, Base, :getfield)) &&
  ex.args[3] isa Union{QuoteNode,Integer}
{% endhighlight %}

Modify the existing `_generate_pullback_via_decomposition` and `_generate_callable_pullback` functions to call it:

{% highlight julia %}
function _generate_pullback_via_decomposition(T, world)
    m = meta(T; world=world)
    isnothing(m) && return nothing
    ir = IR(m)
    length(blocks(ir)) == 1 || error("control flow is not supported")
    ir = instrument(ir) # new
    pr, calls = primal(ir, T)
    m, pr, calls
end

function _generate_callable_pullback(j::Type{<:Pullback{S, T}}, world, Δ) where {S, T}
    m = meta(S; world=world)
    ir = IR(m)
    isnothing(ir) && return :(error("Non-differentiable function ", repr(args[1])))
    length(blocks(ir)) == 1 || error("control flow is not supported")
    ir = instrument(ir) # new
    back = reverse_differentiate(ir)
    back = slots!(inlineable!(back))
    ci = build_codeinfo_(back)
    ci.slotnames = [Symbol("#self#"), :Δ]
    ci
end 
{% endhighlight %}

Now we need to define `literal_getfield` and `__new__` and their pullbacks.

<h3 id="pullback-getfield">2.3 getfield</h3>

Calls to `getproperty` default to `getfield`, where a field is is declared in a struct's declaration.
The `getfield` function is substituted with `literal_getfield` ([source](https://github.com/FluxML/ZygoteRules.jl/blob/f9bf0e367fa259c5aa68f0e14ccbf2125d734bd6/src/ZygoteRules.jl#L19)):

{% highlight julia %}
literal_getfield(x, ::Val{f}) where f = getfield(x, f)
{% endhighlight %}

The pullback will return a `NamedTuple` for each field, where the gradient is `Δ` for the relevant field and nothing for the others ([source](https://github.com/FluxML/Zygote.jl/blob/3c3325d9987931f15bd478c932332be19c316de4/src/lib/lib.jl#L228)):

{% highlight julia %}
@generated nt_nothing(x) = Expr(:tuple, [:($f=nothing) for f in fieldnames(x)]...)
@generated pair(::Val{k}, v, _=nothing) where k = :($k = v,)

function pullback(::typeof(literal_getfield), x, ::Val{f}) where f
  val = getfield(x, f)
  function literal_getfield_back(Δ)
    if isimmutable(x)
      dx = (; nt_nothing(x)..., pair(Val(f), Δ)...)
      (nothing, dx, nothing)
    else
      error("multable stucts not supported")
    end
  end
  val, literal_getfield_back
end

pullback(::typeof(getfield), x, field_name::Symbol) = pullback(literal_getfield, x, Val(field_name))
{% endhighlight %}

For example:
{% highlight julia %}
struct Foo
    a
    b
    c
end
foo = Foo(1.0, 'a', "hello")
z, back = pullback(getfield, foo, :b) # ('a', literal_getfield_back)
back(1.0) # (nothing, (a = nothing, b = 1.0, c = nothing), nothing)
{% endhighlight %}

And for the polynomial model:
{% highlight julia %}
z, back = pullback(model, 1.0)
back(2.3) # ((weights = [2.3, 2.3, 2.3, 2.3],), -2.3)
{% endhighlight %}

For the first time we have a value $\text{s̄elf}$, which is the named tuple for the fields.

<h3 id="pullback-new">2.4 new</h3>

The code now works with:
{% highlight julia %}
(m::Polynomial)(x::AbstractVector) = map(m, x)
{% endhighlight %}

It returns $\text{s̄elf}$ and $\bar{x}$:
{% highlight julia %}
model = Polynomial([3.0, 2.0, -3.0, 1.0])
x = [1.0, 2.0, 3.0, 4.0]
z, back = pullback(model, x)
back(ones(4)) # ((weights = [4.0, 10.0, 30.0, 100.0],), [-1.0, 2.0, 11.0, 26.0])
{% endhighlight %}

However with an anonymous function:

{% highlight julia %}
(m::Polynomial)(x::AbstractVector) = map(x->evalpoly(x, m.weights), x)
{% endhighlight %}

`nothing` is returned for $\text{s̄elf}$:
{% highlight julia %}
z, back = pullback(model, x)
back(ones(4)) # (nothing, [-1.0, 2.0, 11.0, 26.0])
{% endhighlight %}

If we inspect the `primal(ir)`, we will see that its because no pullbacks and hence no gradients are recorded against variable `%1` (`self`):
{% highlight plaintext %}
1: (%1, %2)
  %3 = Main.:(var"#74#75")
  %4 = Core.typeof(%1)
  %5 = Core.apply_type(%3, %4)
  %6 = %new(%5, %1)
  %7 = Main.pullback(Main.map, %6, %2)
  %8 = Base.getindex(%7, 1)
  %9 = Base.getindex(%7, 2)
  %10 = Base.tuple(%9)
  %11 = (Pullback{Any})(%10)
  %12 = Base.tuple(%8, %11)
  return %12
{% endhighlight %}

The solution is to swap `%new` with a call to a custom function `__new__` with a pullback.
This function is as follows ([source](https://github.com/FluxML/Zygote.jl/blob/3c3325d9987931f15bd478c932332be19c316de4/src/tools/builtins.jl)):

{% highlight julia %}
macro __splatnew__(T, args)
  esc(Expr(:splatnew, T, args))
end

@inline __new__(T, args...) = @__splatnew__(T, args)
{% endhighlight %}

And the pullback is ([source](https://github.com/FluxML/Zygote.jl/blob/3c3325d9987931f15bd478c932332be19c316de4/src/lib/lib.jl#L289)):

{% highlight julia %}
using Base: RefValue
struct Jnew{T,G}
  g::G
end

Jnew{T}(g) where T = Jnew{T,typeof(g)}(g)

function pullback(::typeof(__new__), T, args...)
  x = __new__(T, args...)
  g = !ismutabletype(T) || fieldcount(T) == 0 ? nothing : grad_mut(x)
  x, Jnew{T,typeof(g)}(g)
end

@generated function (back::Jnew{T,G})(Δ::Union{NamedTuple,Nothing,RefValue}) where {T,G}
  !ismutabletype(T) && Δ == Nothing && return :nothing
  Δ = G == Nothing ? :Δ :
      Δ <: RefValue ? :(back.g[]) :
      :(accum(back.g[], Δ))
  quote
    x̄ = $Δ
    $(G == Nothing || :(back.g[] = nt_nothing($Δ)))
    (nothing, nothing, $(map(f -> :(x̄.$f), fieldnames(T))...))
  end
end
{% endhighlight %}

Now if we try the following (after redefining `@generated function pullback` and `function (methodinstance::Pullback)`) we should get the same results:
{% highlight julia %}
z, back = pullback(model, x)
back(ones(4)) # ((weights = [4.0, 10.0, 30.0, 100.0],), [-1.0, 2.0, 11.0, 26.0])
{% endhighlight %}

<h2 id="gradient-descent-revisited">3 Gradient Descent revisited</h2>
<h3 id="generic-gradient-descent">3.1 Generic Gradient Descent</h3>

Now that we have an automatic differentiation engine, it is possible to create a much more generic gradient descent function than in [part 1](/machine-learning/2024/07/27/micrograd-1-chainrules.html#gradient-descent).
This is what it looks like:

{% highlight julia %}
function gradient_descent!(
    model,
    loss,
    X::AbstractVecOrMat,
    Y::AbstractVecOrMat
    ; learning_rate::AbstractFloat=0.1,
    max_iters::Integer=100
    )
    losses = Float64[]
    for i in 1:max_iters
        loss_iter, back = pullback(model) do m
            result = m(X)
            loss(result, Y)
        end 
        Δf, Δm = back(1.0)
        update_params!(parameters(model), Δm; learning_rate=learning_rate)
        push!(losses, loss_iter)  
    end
    losses
end
{% endhighlight %}

Note that `pullback(m->f(m), model)` is directly equivalent to `pullback(model) do f(m) end`.

The `update_params!` function is defined as follows:

{% highlight julia %}
function update_params!(params::NamedTuple, grads::NamedTuple; options...)
    for key in keys(params)
        update_params!(params[key], grads[key]; options...)
    end
end

function update_params!(params::Tuple, grads::Tuple; options...)
    for (p, g) in zip(params, grads)
        update_params!(p, g; options...)
    end
end

function update_params!(params, grads; learning_rate::AbstractFloat=0.1)
    params .-= learning_rate .* grads # must broadcast to edit elements and not copies!
end
{% endhighlight %}

The `parameters` function is defined per model.
(Flux uses the generic Functors.jl library to accomplish something similar.)

<h3 id="polynomial-curve-fitting-revisited">3.2 Polynomial curve fitting revisited</h3>

Let's create the exact same data set from part 1:

{% highlight julia %}
using StatsBase
target_weights = [15.0, -2.1, 13.9, 1.5]
noise_factor = 0.2
xs = (rand(100) .- 0.5) .* 10
ys = map(x -> evalpoly(x, target_weights), xs)
scale_factor = mean(abs.(ys))
ys .+= randn(length(ys)) * scale_factor * noise_factor
{% endhighlight %}

The `Polynomial` model is defined in the introduction. All we need is a customer method for  `parameters`:
{% highlight julia %}
parameters(m::Polynomial) = (;weights=m.weights)
{% endhighlight %}

Define the model:
{% highlight julia %}
model = Polynomial(rand(4))
{% endhighlight %}

Some sanity checks:
{% highlight julia %}
x = [1.0, 2.0, 3.0]
z, back = pullback(model, x) # ([1.68, 7.21, 21.2], Pullback) 
back([1.0, 1.0, 1.0]) # ((weights = [3.0, 6.0, 14.0, 36.0],), [-1.0, 2.0, 11.0])
z, back = pullback(m->m(x), model) 
back([1.0, 1.0, 1.0]) # (nothing, (weights = [3.0, 6.0, 14.0, 36.0],))
y = [2.0, 4.0, 8.0]
z, back = pullback(m->mse(m(x), y), model) 
back(1.0) # (nothing, (weights = [10.7 30.5, 87.6, 254.6],))
{% endhighlight %}

Train the model:
{% highlight julia %}
history = gradient_descent!(model, mse, xs, ys; learning_rate=1e-5, max_iters=2000)
{% endhighlight %}

This works just as well as before.

<h2 id="conclusion">4 Conclusion</h2>

We now have a fully working AD package.
It has some limitations, such as it cannot handle control flow or keyword arguments.
However it can already work a wide variety of code.
All that might be needed is some explicit `rrule` definitions.
The next and final [part][micrograd_mlp] of this series is a demonstration of exactly that.

---
