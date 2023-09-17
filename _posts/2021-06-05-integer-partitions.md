---
layout: post
title:  "Integer partitions"
date:   2021-06-05
last_modified_at: 2023-09-10
author: Lior Sinai
categories: coding
categories: mathematics
tags: mathematics probability
---

_Algorithms for integer partitions._ 

<script src="https://cdn.plot.ly/plotly-2.25.2.min.js" charset="utf-8"></script>

## Table of contents

<ol>
    <li><a href="#introduction">Introduction</a></li>
    <li><a href="#recursion">Recursion</a>
        <ol type="i">
            <li><a href="#counting">Counting</a></li>
            <li><a href="#integer-partitions-recursion">Integer partitions</a></li>
            <li><a href="#bounded-partitions-recursion">Bounded partitions</a></li>
        </ol>
    </li>
    <li><a href="#linear">Linear</a>
        <ol type="i">
            <li><a href="#integer-partitions-linear">Integer partitions</a></li>
            <li><a href="#bounded-partitions-linear">Bounded partitions</a></li>
        </ol>
    </li>
</ol>

## Introduction

[wiki_partitions]: https://en.wikipedia.org/wiki/Partition_(number_theory)
[advanced_collisions]: {{ "mathematics/2021/06/05/advanced-collisions" | relative_url }}

As per [Wikipedia][wiki_partitions], an integer partition of $n$ is the number of ways of writing $n$ as a positive integer.
For example, for 5:

$$
\begin{align}
    &5\\
    &4+1\\
    &3+2\\
    &3+1+1\\
    &2+2+1\\
    &2+1+1+1\\
    &1+1+1+1+1
\end{align}
$$


For this post I'll look at slight variation where a maximum value can be set.
In that case, some of the combinations need to be excluded.
For example for $n=5$ and a maximum value of $3$, the top two ways are excluded.

The previous blog post [The Birthday Problem: Advanced][advanced_collisions] required something a little more complicated: integer partitions but bounded by certain values at each index.
For example, select 5 people from 3 teams where there are 2 people in the first team, 1 person in the second team, and 5 or more people in the third team. Here the partitions are:

$$
\begin{align}
    &2+1+2\\
    &2+0+3\\
    &1+1+3\\
    &1+0+4\\
    &0+1+4\\
    &0+0+5
\end{align}
$$

Note that in this problem permutations are distinct from each other, unlike the first problem.

The rest of this post will detail algorithms for these in Julia code.

## Recursion

### Counting

Define $p(n)$ as the count of integer partitions of $n$. Then $p(n) = \sum_{k=1}^n p(n - k)$. 
This can readily be converted into a recursive formula:

{%highlight julia %}
function count_integer_partitions(n::Integer, max_value::Int)
    if n == 0
        return 1
    elseif n <= 0
        return 0
    end
    count_ = 0
    for k in max_value:-1:1
        count_ += count_integer_partitions(n-k, min(k, n-k))
    end        
    count_
end

count_integer_partitions(n::Integer) = count_integer_partitions(n, n)
{% endhighlight %}

The bounded version requires the slight modification of keeping track of the index so that the correct maximum can be selected.
It also includes a zero case.
{%highlight julia %}
function count_integer_partitions(n::Integer, max_values::Vector{Int}, idx=1)
    if n == 0
        return 1
    elseif n <= 0
        return 0
    end
    count_ = 0
    max_value = (idx <= length(max_values)) ? min(n, max_values[idx]) : 0
    min_value = (idx + 1 <= length(max_values)) ? 0 : 1
    for k in max_value:-1:min_value
        count_ += count_integer_partitions(n-k, max_values, idx + 1)
    end    
    count_
end
count_integer_partitions(n::Integer) = count_integer_partitions(n, n)
{% endhighlight %}

<h3 id="integer-partitions-recursion"> Integer partitions </h3>

We can modify the counting formula to instead return arrays.
Then we can return the whole set of arrays.
{%highlight julia %}
function integer_partitions(n::Integer, max_value::Int)
    if n < 0
        throw(DomainError(n, "n must be nonnegative"))
    elseif n == 0
        return Vector{Int}[[]]
    end
    partitions = Vector{Int}[]
    for k in max_value:-1:1
        for p in integer_partitions(n-k, min(k, n-k))
            push!(partitions, vcat(k, p))
        end
    end        
    partitions
end
{% endhighlight %}

In Python you can use the `yield` keyword to return each partition one at a time.
Unfortunately you cannot do that in Julia, but changing to a linear algorithm we can get the same behaviour. 
See the [Linear integer partitions](#integer-partitions-linear) section.

<h3 id="bounded-partitions-recursion"> Bounded partitions </h3>

For the bounded version we can do a similar modification to its counting function:
{%highlight julia %}
function integer_partitions(n::Integer, max_values::Vector{Int}, idx=1)
    if n < 0
        throw(DomainError(n, "n must be nonnegative"))
    elseif n == 0 
        return Vector{Int}[[]]
    end
    partitions = Vector{Int}[]
    max_value = (idx <= length(max_values)) ? min(n, max_values[idx]) : 0
    min_value = (idx + 1 <= length(max_values)) ? 0 : 1
    for k in max_value:-1:min_value
        for p in integer_partitions(n-k, max_values, idx + 1)
            push!(partitions, vcat(k, p))
        end
    end    
    partitions
end
{% endhighlight %}

Caching can be added to improve the performance. 

## Linear
<h3 id="integer-partitions-linear"> Integer partitions </h3>

We can also write these functions in linear stateful versions. 
With this technique we don't have to hold the whole array of all partitions in memory.
(In Python you could use the `yield` keyword to accomplish this.)

The state is the last partition and the current index of the value that is being decremented.

First define a `struct` to be the iterator to dispatch on (see [Julia manual: interfaces](https://docs.julialang.org/en/v1/manual/interfaces/)):
{%highlight julia %}
struct IntegerPartitions
    n::Int
    max_value::Int
    function IntegerPartitions(n::Int, max_value::Int)
        if n < 0
            throw(DomainError(n, "n must be nonnegative"))
        end
        new(n, min(max_value, n))
    end
end

IntegerPartitions(n::Int) = IntegerPartitions(n, n)

Base.IteratorSize(::IntegerPartitions) = Base.SizeUnknown()
Base.eltype(::IntegerPartitions) = Vector{Int}
{% endhighlight %}

We could use the `count_integer_partitions` function to get the length.
But because this requires some computation, it is better to set the size as `Base.SizeUnknown()`.

Next, we iterate through the partitions as before.
But this time instead of jumping to a new function call, we move an index up and down (like a pointer).

{% highlight julia %}
function Base.iterate(iter::IntegerPartitions) 
    if iter.n == 0
        return nothing
    end
    parts0 = zeros(Int, iter.n) # shared state that is mutated
    parts0[1] = iter.max_value + 1
    iterate(iter, (parts0, 1))
end

function Base.iterate(iter::IntegerPartitions, state::Tuple{Vector{Int}, Int}) 
    partitions, idx = state
    k = partitions[idx] - 1
    while (k == 0)
        if (idx == 1)
            return nothing
        else
            idx -= 1
            k = partitions[idx] - 1
        end
    end
    partitions[idx] = k
    residue = iter.n - sum(partitions[1:idx])
    i = idx
    while residue != 0
        i += 1
        k = min(k, residue)
        if (k > 1)
            idx += 1
        end
        partitions[i] = k
        residue -= k
    end
    (partitions[1:i], (partitions, idx)) # copy, shared state
end
{% endhighlight %}

Example:
{% highlight julia %}
for part in IntegerPartitions(5, 3)
    println(part)
end
#[3, 2]
#[3, 1, 1]
#[2, 2, 1]
#[2, 1, 1, 1]
#[1, 1, 1, 1, 1]
{% endhighlight %}

<h3 id="bounded-partitions-linear"> Bounded partitions </h3>

The `struct` is similar to before.
We will set the maximum of `max_values` to `n` here.
{%highlight julia %}
struct BoundedPartitions
    n::Int
    max_values::Vector{Int}
    function BoundedPartitions(n::Int, max_values::Vector{Int})
        if n < 0
            throw(DomainError(n, "n must be nonnegative"))
        end
        max_values = copy(max_values)
        for (idx, val) in enumerate(max_values)
            if val > n
                @warn "maximum value=$val > n=$n; setting to n."
                max_values[idx] = n
            end
        end
        new(n, max_values)
    end
end

Base.IteratorSize(::BoundedPartitions) = Base.SizeUnknown()
Base.eltype(::BoundedPartitions) = Vector{Int}
{% endhighlight %}

The iterations require a few subtle modifications.
Parts which are not modified are not shown.
{%highlight julia %}
function Base.iterate(iter::BoundedPartitions) 
    if iter.n == 0
        return nothing
    end
    parts0 = zeros(Int, iter.n) # shared state that is mutated
    parts0[1] = iter.max_values[1] + 1 
    iterate(iter, (parts0, 1))
end

function Base.iterate(iter::BoundedPartitions, state::Tuple{Vector{Int}, Int}) 
    partitions, idx = state
    while (partitions[idx] - 1 < 0) && (idx < length(iter.max_values))
        idx += 1
    end
    k = min(partitions[idx] - 1, iter.max_values[idx])
    while k < 0 || 
        (k + sum(iter.max_values[idx+1:end])) < (iter.n - sum(partitions[1:(idx - 1)]))
        # current value + maximum possible sum < residue
        ...
    end
    ...
    while residue != 0
        i += 1
        k = min(iter.max_values[i], residue)
        if (k > 0)
            idx += 1
        end
        ...
    end
    (partitions[1:i], (partitions, idx)) # copy, shared state
end
{% endhighlight %}

Example:
{% highlight julia %}
for part in BoundedPartitions(5, [2, 1, 5])
    println(part)
end
#[2, 1, 2]
#[2, 0, 3]
#[1, 1, 3]
#[1, 0, 4]
#[0, 1, 4]
#[0, 0, 5]
{% endhighlight %}


---
