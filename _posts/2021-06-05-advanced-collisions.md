---
layout: post
title:  "The Birthday Problem: Advanced"
date:   2021-06-05
author: Lior Sinai
categories: coding
categories: mathematics
tags: mathematics probability
---

_Exact probabilities for counting the chance of collisions with distinct objects, using the birthday problem as a working example._ 

# Introduction

This follows from my previous blog post on [Intuitive explanations for non-intuitive problems: the Birthday Problem][BirthdayProblem]. 
Please see that post for a heuristic understanding of the problem.
It assumed independence of sharing birthdays.
This post makes no such assumption, so the mathematics is more advanced.
A background in counting methods, combinations and permutations is required.

[BirthdayProblem]: {{ "mathematics/2021/06/04/birthday-collisions" | relative_url }}

I have divided this section in to the following parts:
- [Probability of k collisions](#exact-k-collisions)
- [Probability of groups of collisions](#groups-collisions)
- [Probability of at least 1 collision](#b)
- [Probability when drawing from a distribution](#c)
- [Integer partitions](#d)

<h2 id="exact-k-collisions"> Probability of $k$ collisions </h2>

What is the chance of $i$ people in a group of $k$ people sharing a birthday? In the previous post I assumed independence between sharing birthdays. 
But sharing is dependent. If person A and person B share a birthday, then sharing a birthday with person B depends on sharing a birthday with person A. It is easiest to use counting methods to account for this dependence.

In a group of 23 people there are ${23 \choose 2}=253$ ways that two people can be chosen. 
Seperately we can imagine filling 22 slots with distinct birthdays (not 23 because one is shared). 
There are $^{365}P_{22}$ ways of filling these slots. The total probability is then:

$$ \begin{align} 
P(x=2 | k=23) &= {23 \choose 2} \frac{^{365}P_{22}}{365^{23}} \\
              &= 0.3634
\end{align}
$$ 

Simarily for three people sharing a birthday:

$$ \begin{align} 
P(x=3 | k=23) &= {23 \choose 3} \frac{^{365}P_{21}}{365^{23}} \\
              &= 0.0074
\end{align}
$$ 

<h2 id="groups-collisions"> Probability of groups of collisions </h2>a

There can also be two pairs of two people sharing birthdays. Or three pairs. Or two pairs and a triplet.

Let's take the case of two pairs of two people, and one set of three. There are 23 people in total.


<figure class="post-figure" id="partitions">
<img class="img-80" 
    src="/assets/posts/birthday-collisions/partitions.png"
	alt="partitions"
	>
	<figcaption></figcaption>
</figure>

## Assumptions

$$ 
\begin{align}
	1-p &= \left(1-\frac{1}{n}\right)^{(k^2-k)/2} \\
	ln(1-p) &= \frac{k^2-k}{2}ln\left(1-\frac{1}{n}\right) \approx  \frac{k^2-k}{2}\left(\frac{-1}{n}\right)  \quad\quad, ln(1+x) \approx x  \\
	k^2 &\approx  -2n\cdot ln(1-p)	 \quad\quad, k^2 >> k 
\end{align}
$$

Substitute $p=0.5$ to get $k\approx1.774\sqrt{n}$

---
