---
layout: post
title:  "Quaternions: Part 4"
date:   2021-12-06
author: Lior Sinai
categories: mathematics
tags: mathematics quaternions unity rotation 3d
---

_Interpolation using quaternions._ 


This is part of a series. The other articles are:
- [Part 1: introduction][introduction].
- [Part 2: 2D rotations][2Drotations].
- [Part 3: quaternions][quaternions].

[introduction]: {{ "mathematics/2021/11/05/quaternion-1-intro" | relative_url }}
[2Drotations]: {{ "mathematics/2021/11/28/quaternion-2-2d" | relative_url }}
[quaternions]: {{ "mathematics/2021/12/03/quaternion-3" | relative_url }}
[interpolation]: {{ "mathematics/2021/12/06/quaternion-4" | relative_url }}

<script src="https://cdn.plot.ly/plotly-gl3d-2.5.1.min.js"> </script>
<link rel="stylesheet" href="/assets/posts/quaternions/style.css">

### Table of Contents
- [Introduction](#introduction)
- [Lerp](#lerp)
- [Slerp](#slerp)
	- [Formula 1](#formula-1)
	- [Formula 2](#formula-2)
- [Conclusion](#conclusion)

# Interpolation with Quaternions
## Introduction

This is a demo of the two interpolation methods that will be described in this article.
Using the default button sets the stick aeroplane to the same settings as the space cruiser in [part 1][introduction]. See the source code [here][Quaternion.js].

[Quaternion.js]: {{ "assets/posts/quaternions/Quaternion.js" | relative_url }}

<div class="plot3d-container">
	<form class="grid-container">
		Start position:  
		<label for="psiNumber">&psi;</label>
		<input type="number" id="psiStartNumber" min="-360" max="360" value="0">
		<label for="thetaNumber">&theta;</label>
		<input type="number" id="thetaStartNumber" min="-360" max="360" value="0">
		<label for="psiNumber">&phi;</label>
		<input type="number" id="phiStartNumber" min="-360" max="360" value="0">
		End position:
		<label for="psiNumber">&psi;</label>
		<input type="number" id="psiEndNumber" min="-360" max="360" value="0">
		<label for="thetaNumber">&theta;</label>
		<input type="number" id="thetaEndNumber" min="-360" max="360" value="0">
		<label for="psiNumber">&phi;</label>
		<input type="number" id="phiEndNumber" min="-360" max="360" value="0">
	</form>
	<div class="sliderContainer">
		<p class="sliderValue left">t</p>
		<input id="tSlider" type="range" min="0" max="1" step="0.01" value="0" class="slider">
		<p id='tSliderText' class="sliderValue right">0.00</p>
	</div>
	<form>
		<button type="button" id="resetButton">reset</button>
		<button type="button" id="defaultButton">default</button>
		<input type="radio" id="lerpRadio" checked name="interpRadios">
		<label for="lerpRadio">lerp</label>
		<input type="radio" id="slerpRadio" name="interpRadios">
		<label for="slerpRadio">slerp</label>
	</form>
	<div id="canvas" class="plotly">
		<script src="/assets/posts/quaternions/plotInterpolation.js" type="module"></script>
	</div>
</div>

In [part 1][introduction] the problem of finding a satisfying rotation halfway between two other rotations was presented.
We now extend this problem statement to any fraction: given an initial rotation at time $t=0$ and a final rotation at $t=1$ find a smooth formula for a rotation at time $t$ where $0 \leq t \leq 1$.

This post will describe two solutions which give slightly different results:
- Lerp: linear interpolation.
- Slerp: spherical linear interpolation.

The equation for lerp is simpler but slerp has a smoother profile. 
This however is hardly noticeable in above demo.

This post is based on [Quaternions, Interpolation and Animation by Erik B. Dam, Martin Koch and Martin Lillholm][Dam1998]. Please see this book for more detail on using quaternions in animations and the formulas presented in this post.

[Dam1998]: https://web.mit.edu/2.998/www/QuaternionReport1.pdf

## Lerp

Because quaternions encode rotations it is possible to linearly interpolate between the initial quaternion $q_0$ and the final quaternion $q_1$:

$$ \text{lerp}(q_0, q_1, t) = q_0 (1 - t) + q_1 t $$

The resulting quaternion has no guarantee of being a unit quaternion, so it needs to be scaled by its magnitude.

$$ \text{lerp}(q_0, q_1, t) = \frac{q_0 (1 - t) + q_1 t }{\left\| q_0 (1 - t) + q_1 t  \right \|} $$

This interpolation has a non-linear velocity profile which peaks at $t=0.5$. 
In other words, the animation will appear to go faster in the middle than at the start and finish.

## Slerp

In 2D the angle $\Omega$ between two unit vectors $z_0$ and $z_1$ is linearly interpolated with $t\Omega$.  
A simple formula for $z_t$ is:

$$ z_t = z_0 (cos(t\Omega  ) + i sin(t\Omega )) = cos(\theta_0 + t\Omega ) +  sin(\theta_0 + t\Omega ) i $$

The angular velocity will have a constant magnitude of $\Omega $ rad/s.

Two other formulas can also be derived for the same interpolation. These are more useful because they can be extended to 3D.
 
### Formula 1 

<figure class="post-figure">
<img class="img-50"
    src="/assets/posts/quaternions/slerp.png"
	alt="geometric construction for slerp"
	>
<figcaption></figcaption>
</figure>

From the diagram:

$$ 
\begin{align}
\vec{v}_1 &= \color{red}{cos\Omega \vec{v}_0 + sin\Omega \vec{v}_{\perp}} \\
\therefore \vec{v}_{\perp} &= \frac{1}{sin\Omega} (\vec{v}_1 - cos\Omega \vec{v}_0)
\end{align}
$$

Substitute in the expression for $\vec{v}_t$:

 $$ 
\begin{align}
\vec{v}_t &= \color{green}{cos (t \Omega )\vec{v}_0 + sin (t\Omega ) \vec{v}_{\perp}} \\
          &= cos (t \Omega )\vec{v}_0 + \frac{ sin (t\Omega )}{sin\Omega}(\vec{v}_1 - cos\Omega \vec{v}_0) \\
		  &= \frac{ 1 }{sin\Omega}((cos (t \Omega )sin\Omega  -sin (t\Omega )cos\Omega )\vec{v}_0 + sin (t\Omega ) \vec{v}_1 )\\
		  &= \frac{ 1 }{sin\Omega}(sin(\Omega -t\Omega)\vec{v}_0 + sin (t\Omega ) \vec{v}_1 )
\end{align}
$$

To convert this formula to 3D, replace the vectors with quaternions. That is all:

$$ 
\begin{align}
\text{slerp}(q_0, q_1, t) &= \frac{ 1 }{sin\Omega}(sin(\Omega -t\Omega)q_0 + sin (t\Omega ) q_1 ) \\
cos\Omega &= q_0 \cdot q_1 = s_0s_1 + x_0x_1 + y_0y_1 + z_0z_1
\end{align}
$$

This formula is used in the graph at the top and is the most common formula in quaternion animations. Please see the [reference][dam1998] for a formal proof.

My informal explanation is: this formula can be seen as the linear addition of four parts: real, $i$, $j$ and $k$. The real part will cancel out in $qvq^{*}$. The other parts will undergo the same transformation as the 2D complex part $iy$ in $\vec{v} \equiv z = x + iy$, and combined form the interpolation across a sphere.

If you have a better explanation, please let me know.

### Formula 2

Using Euler's formula:

$$ z_1 = z_0 e^{i \Omega} $$

Substitute in the expression for $z_t$:

$$ 
\begin{align}
	z_t &= z_0 e^{it \Omega} \\
	    &= z_0 (e^{i\Omega})^t \\
		&= z_0 \left( \frac{z_1}{z_0} \right)^t
\end{align}
$$

In quaternion form:

$$ \text{slerp}(q_0, q_1, t) = q_0 (q_0^{-1} q_1)^t = q_0 (q_0^{*} q_1)^t $$

<p>
  <a class="btn" data-toggle="collapse" href="#proofEqual" role="button" aria-expanded="false" aria-controls="collapseExample">
    More detail: proving the slerp formulas are equal &#8681;
  </a>
</p>
<div class="collapse" id="proofEqual">
  <div class="card card-body ">
		The two quaternion slerp formulas can be proven equal by a direct but somewhat tedious calculation:
		$$
		\begin{align}
			\text{slerp}(q_0, q_1, t) &=  q_0 (q_0^{*} q_1)^t \\
				&= (cos\theta_0 + sin\theta_0 \hat{n}_0)((cos\theta_0 - sin\theta_0 \hat{n}_0)(cos\theta_1 + sin\theta_1 \hat{n}_1))^t \\
				&= (cos\theta_0 + sin\theta_0 \hat{n}_0)(cos\Omega + v \: sin\Omega )^t \\
				&= (cos\theta_0 + sin\theta_0 \hat{n}_0)(cos(t\Omega) + v \: sin(t\Omega))
		\end{align}
		$$
		where:
		$$ v \: sin\Omega  = cos\theta_0 sin\theta_1 \hat{n}_1 - cos\theta_1 sin\theta_0 \hat{n}_0 - sin\theta_0 sin\theta_1 \hat{n}_0 \times \hat{n}_1 $$
		Substitute in for $v$ and simplify. The end result will be the other slerp formula.
  </div>
</div>

## Conclusion

If you've made it this far, thank you for following me along this journey.

I hope you've enjoyed this dive into the fascinating mathematics of quaternions.

---

