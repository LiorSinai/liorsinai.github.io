---
layout: post
title:  "Quaternions: Part 2"
date:   2021-11-28
author: Lior Sinai
categories: mathematics
tags: mathematics quaternions unity rotation 3d
---

_Rotations in 2D._ 


This is part of a series. The other articles are:
- [Part 1: introduction][introduction].
- [Part 3: quaternions][quaternions].
- [Part 4: interpolation][interpolation].

[introduction]: {{ "mathematics/2021/11/05/quaternion-1-intro" | relative_url }}
[2Drotations]: {{ "mathematics/2021/11/28/quaternion-2-2d" | relative_url }}
[quaternions]: {{ "mathematics/2021/12/03/quaternion-3" | relative_url }}
[interpolation]: {{ "mathematics/2021/12/06/quaternion-4-interpolation" | relative_url }}


### Table of Contents
- [Polar and Cartesian formula](#polar-and-cartesian-formula)
- [Complex numbers](#complex-numbers)
- [Conclusion](#conclusion)

# Rotations in 2D

Rotations in 2D are simple and will help prepare us for 3D rotations.
There are three methods to accomplish rotations in 2D:
- Polar co-ordinates and angles.
- Cartesian matrix formula.
- Complex numbers.

This post gives an overview of all these methods.

## Polar and Cartesian formula

Rotations in 2 dimensional space can be represented with cartesian co-ordinates $(x, y)$ or polar co-ordinates $(r, \theta$).

<figure class="post-figure">
<img class="img-40"
    src="/assets/posts/quaternions/rotation_2d_proof.png"
	alt="Rotation formula construction"
	>
<figcaption>Rotation formula construction. This is also a geometric proof of the addition and subtraction formulas for sine and cosine. </figcaption>
</figure>

From the figure, the rotated co-ordinate $x_r$ is related to $x$ with the angle $\beta$ by:

$$ 
\begin{align}
x_r &= rcos(\beta + \theta) \\
    &= \color{green}{rcos\beta cos\theta} - \color{red}{rsin\beta sin\theta} \\
    &=  (r\cos\theta)cos\beta - (rsin\theta)sin\beta \\
    &= (x)cos\beta - (y)sin\beta 
\end{align}
$$

Similarly for $y_r$:

$$ 
\begin{align}
y_r &= rsin(\beta + \theta) \\
    &= \color{red}{rsin\beta cos\theta} + \color{green}{rcos\beta sin\theta} \\
    &=  (r\cos\theta)sin\beta + (rsin\theta)cos\beta \\
    &= (x)sin\beta + (y)cos\beta 
\end{align}
$$

Grouping together in matrix form:

$$ 
\begin{bmatrix}
    x_r \\
    y_r
\end{bmatrix}
=
\begin{bmatrix}
    cos\beta & -sin\beta \\
    sin\beta & \phantom{+}cos\beta 
\end{bmatrix}
\begin{bmatrix}
    x \\
    y
\end{bmatrix}
$$

This matrix should be recognisable in the Euler rotation matrices in [part 1][euler].

[euler]: {{ "mathematics/2021/11/05/quaternion-1-intro#1-three-angles-and-an-order" | relative_url }}

Calling this matrix $R$, we find that:

$$ 
R(-\theta) = R^{-1} = R^T  
=
\begin{bmatrix}
    \phantom{+}cos\beta & sin\beta \\
    -sin\beta & cos\beta 
\end{bmatrix}
$$

The negative angle relation will not hold in 3D, but $R^{-1}=R^T$ is a useful relationship that can be expanded to higher dimensions.

<div class="card">
  <div class="card-body">
    <h5 class="card-title">Proof of the rotation matrix inverse</h5>
    <p class="card-text">
		A vector $v$'s magnitude can be found as $v^Tv=\|v\|$, where $v$ is a column vector.
        The rotated vector should have the same magnitude: 
        $$ (Rv)^T(Rv)=v^T R^T Rv = v^Tv =\|v\|$$
        This requires that $R^T R =I $ and hence $R^T=R^{-1}$.
	</p>
  </div>
</div>


## Complex numbers

Complex numbers are a completely different way to represent rotations.
A complex number $z$ is defined as:

$$ z = x + yi \; ; \; x, y \in \mathbb{R}$$

where $i=\sqrt{-1}$ as defined by Euler.

They are also called imaginary numbers, which is a historical misnomer.
This is because they were originally discovered by mathematicians while solving cubic equations.
[In 1572, Rafael Bombelli][history] introduced the imaginary unit $\sqrt{-1}$ while solving the equation:

[history]: https://www.math.uri.edu/~merino/spring06/mth562/ShortHistoryComplexNumbers2006.pdf

$$ x^3 = 15x + 4 $$

Which has a root at:

$$ x= \sqrt[3]{2 + \sqrt{-121}} + \sqrt[3]{2 - \sqrt{-121}} = 4$$

Cubic euqations were solved with a geometric meaning in mind, and in terms of geometry, the "square root of -1" has no meaning, because no square can have negative lengths. 
But in terms of spatial geometry, negative values have the valid interpretation as the "opposite direction"
and the imaginary number $i$ has the interpretation as a value orthogonal to the original direction.
They are no more imaginary than north is to east or left is to forward.

These numbers can be plotted on the complex plane:

<figure class="post-figure">
<img class="img-40"
    src="/assets/posts/quaternions/complex_plane_0.png"
	alt="The complex plane"
	>
<figcaption>The complex plane</figcaption>
</figure>

The multiplication of $1 \cdot i = i$ moves a point in a 90&deg; anti-clockwise arc from $1$ to $i$. 
Hence multiplication by $i$ is interpreted as a 90&deg; rotation. Following the same logic, a further 90&deg; rotation from $i$ will land back on the original axis at $-1$, hence:

$$i\cdot i = i^2=-1$$

We can plot a vector on this plane, such as $2 + 2i \equiv (2, 2)$:
<figure class="post-figure">
<img class="img-40"
    src="/assets/posts/quaternions/complex_plane_1.png"
	alt="the complex plane with vector 2+2i"
	>
<figcaption></figcaption>
</figure>

If a single number multiplied by $i$ is rotated 90&deg; what will happen if we multiply the vector $2 + 2i$ by $i$? 

$$ i(2+2i) = 2i + 2i^2 = 2i - 2 $$

<figure class="post-figure">
<img class="img-40"
    src="/assets/posts/quaternions/complex_plane_2.png"
	alt="the complex plane with vector -2+2i"
	>
<figcaption></figcaption>
</figure>

This is indeed the same vector rotated 90&deg;.

What if we multiply it by a vector with both a real and an imaginary part? 

$$ (0.5 + i)(2+2i) = 0.5(2 + 2i) + i(2+2i) = -1 + 3i$$

<figure class="post-figure">
<img class="img-40"
    src="/assets/posts/quaternions/complex_plane_3.png"
	alt="the complex plane with vector -1 + 3i"
	>
<figcaption></figcaption>
</figure>

The vector has rotated and has also increased in magnitude.

In general, a multiplication with a complex number rotates and scales a vector.
To avoid scaling, we are limited to vectors which lie on the unit circle. 
They can be parameterised with an angle $\theta$ as follows:

$$z = cos(\theta) + isin(\theta)$$

This gives an alternative to calculate rotations. 
Here is proof that this is the same formula as the matrix formula above:

$$ 
\begin{align}
    z (x + iy) &= ( cos(\beta) + isin(\beta)) (x + iy) \\
                &= cos(\beta)(x + iy) + isin(\beta) (x + iy) \\
                &= cos(\beta)(x + iy) + sin(\beta)(ix - y) \\
                &= (cos(\beta)x - sin(\beta)y) + i(sin(\beta)x + cos(\beta)y) \\
                &= x_r + iy_r
\end{align}
$$

It is sufficient to stop here, but there is another more powerful formula for rotations with complex numbers.
It is:

$$ e^{i\theta} = cos(\theta) + isin(\theta) $$

This surprising formula is known as Euler's formula.
It is a very famous formula and there are many explanations for it.
One of the better ones is this [video][3b1b] by 3Blue1Brown.

[3b1b]: https://www.youtube.com/watch?v=mvmuCPvRoWQ

Euler's number is a calculus constant, so it is natural that a proof can be provided based on calculus:

<div class="card">
  <div class="card-body">
    <h5 class="card-title">Proof of Euler's formula</h5>
    <p class="card-text">
		Consider a particle travelling along the path parameterised with time $t$ as
        $$s = cos(t) + isin(t)$$
        The object will move in a circle.
        The velocity is $$\frac{ds}{dt} = -sin(t) + icos(t) = i(cos(t) + isin(t))=is $$
        It is a well known fact that the velocity of an object moving in a circle is at 90&deg; to the cirlce with an acceleration directed inwards to the centre.
        Now consider a path $$s=e^{it}$$
        The velocity is:
        $$\frac{ds}{dt} = ie^{it} = is$$
        Since both formulas satisfy $\frac{ds}{dt} = is$, they must be equal.
	</p>
  </div>
</div>

This formula allows us to prove identities using the properties of exponentials.

As a basic example, multiplying two complex numbers has the same affect as adding their rotation angles and multiplying their magnitudes:

$$ 
\begin{align}
z_1 z_2 &= (r_1cos\theta + i r_1sin\theta)(r_2cos\beta + i r_2sin\beta) \\
        &= (r_1e^{i\theta})(r_2e^{i\beta}) \\
        &= r_1r_2e^{i(\theta + \beta)} \\
        &= r_1r_2(cos(\theta + \beta) + i sin(\theta + \beta))
\end{align}
$$


A more complex relation is de Moivre's formula:

$$ 
\begin{align}
(cos(\theta) + isin(\theta))^n &= (e^{i\theta})^n \\
                               &= e^{i(\theta n)} \\
                               &= cos(n\theta) + isin(n\theta)
\end{align}
$$

Finally, the inverse of this exponential function can be defined as:

$$ log(re^{i\theta}) = log(r) + i\theta log(e) = log(r) + i\theta $$


## Conclusion

This post has now primed us for quaternions which were very much inspired by complex numbers.
However there are important differences, most notably the non-commutativity resulting from the order dependent nature of rotations in 3D.

Please join me in the next [section][quaternions] where I describe quaternions in depth.


---

