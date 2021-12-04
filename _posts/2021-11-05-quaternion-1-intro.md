---
layout: post
title:  "Quaternions: Part 1 (WIP)"
date:   2021-11-05
author: Lior Sinai
categories: mathematics
tags: mathematics quaternions unity rotation 3d
---

_Introduction to quaternions._ 


<script src="https://cdn.plot.ly/plotly-gl3d-2.5.1.min.js"> </script>
<link rel="stylesheet" href="/assets/posts/quaternions/style.css">


This is part of a series. The other articles are:
- [Part 2: 2D rotations][2Drotations].
- [Part 3: quaternions][quaternions].
- [Part 4: interpolation][interpolation].

[introduction]: {{ "mathematics/2021/11/05/quaternion-1-intro" | relative_url }}
[2Drotations]: {{ "mathematics/2021/11/28/quaternion-2-2d" | relative_url }}
[quaternions]: {{ "mathematics/2021/11/28/quaternion-3" | relative_url }}
[interpolation]: {{ "mathematics/2021/11/28/quaternion-4" | relative_url }}

[EulerAngles.js]: {{ "assets/posts/quaternions/EulerAngles.js" | relative_url }}
[Quaternion.js]: {{ "assets/posts/quaternions/Quaternion.js" | relative_url }}

This is a mathematical series and the following prerequisites are recommended: trigonometry, algebra, complex numbers, Euclidean geometry and linear algebra (matrices).

### Table of Contents
- [Animating in 3D](#animating-in-3d)
- [Mathematical representations of rotations in 3D](#mathematical-representations-of-rotations-in-3d)
	- [1. Three angles and an order](#1-three-angles-and-an-order)
	- [2. An axis and an angle](#2-an-axis-and-an-angle)
		- [Other forms of axis-angle rotations](#other-forms-of-axis-angle-rotations)
- [Outline](#outline)
- [References](#references)

# Quaternions
## Animating in 3D

A common problem in computer animations is rotating an object in fully 3D space. 
Think balls, spaceships and heroes tumbling and turning in complex sequences.
This is usually accomplished with an arcane mathematical object called a quaternion.[^quarternion]
For example, here is a spaceship rotating in [Unity][Unity], a popular game engine that is often used to make mobile games:

<figure class="post-figure">
<img class="img-60"
    src="/assets/posts/quaternions/slerp10.gif"
	alt="rotating cruiser"
	>
<figcaption>Space Cruiser 1 by <a href="https://assetstore.unity.com/packages/3d/vehicles/air/space-cruiser-1-124172">Gamer Squid</a></figcaption>
</figure>

[Rotate.cs]: {{ "assets/posts/quaternions/Rotate.cs" | relative_url }}
[Unity]: https://unity.com/

The [code][Rotate.cs] to implement this uses Unity's inbuilt `Quaternion`, making it very succinct: 

{%highlight csharp %}
var t = Time.time * speed;
transform.rotation = Quaternion.Slerp(init.rotation, final.rotation, t);
{% endhighlight %}

The `rotation` property returns a quaternion. 
In Unity's UI, the `init` and `final` rotations are specified by three angles, which are then transformed into quaternions in the backend. This means that printing a rotation will result in four numbers, not three.

What are these four numbers? Unity's own [documentation][unity] is very elusive on what quaternions are. It is worth quoting it:

[unity]: https://docs.unity3d.com/ScriptReference/Quaternion.html

<blockquote>
	<p>
	Quaternions are used to represent rotations. 
	</p>
	<p>
	They are compact, don't suffer from gimbal lock and can easily be interpolated. Unity internally uses Quaternions to represent all rotations.  
	</p>
	<p>
	They are based on complex numbers and are not easy to understand intuitively. You almost never access or modify individual Quaternion components ...
	</p>
</blockquote>

This is a common caveat next to the descriptions of properties:
<blockquote>
	Don't modify this directly unless you know quaternions inside out.
</blockquote>

I did not encounter quaternions in all my years of engineering, although a lecturer once eluded to them during a class in my masters.
They are frowned upon in favour of more intuitive and versatile vectors and matrices.
Those can also be used to calculate 3D rotations and that is an approach that I did use in engineering, especially in my masters.

So why then are computer game developers and animators left to battle with this obscure mathematics that engineers won't touch? Unity gives good reasons above: compactness (4 numbers vs 9 in a matrix), numerical stability ("don't suffer from Gimbol lock") and interpolation (easy to find rotations between other rotations).[^interpolation]
These are significant differentiating factors in computer games and animations, which need to compute many thousands of rotations every frame (especially with regards to light rays).

This series of posts seeks to illuminate quaternions so that you are one of those people who knows them "inside out".
As you will see, quaternions have a mathematical elegance to them that translates directly to a simplicity in code and superior computational performance. 
Later posts will build the logical foundations for quaternions and describe the mathematics in detail.
The rest of this post will be an interactive review of rotation methods in 3D.
It presents results and formulas without proofs.

## Mathematical representations of rotations in 3D

There are two method to describe 3D rotations in mathematics:
1. Three angles and an order: a single 3D rotation can be thought as of the net result of three 2D rotations done around three separate axes in a particular order.
2. An axis with three co-ordinates and an angle: a single 2D rotation can be described around an axis orientated in 3D space. 

In both cases four quantities are used to describe the rotation, but usually one degree of freedom is fixed.
For three angles, the order is fixed. For the axis-angle representation, the axis is set to have a unit length, so it can only exist on the unit sphere and hence two numbers are sufficient to describe it (latitude and longitude).
In either case, there are three remaining degrees of freedom. 

### 1. Three angles and an order

<div class="plot3d-container">
	Instructions: slide each angle in the following order: &psi;, &theta;, &phi;.
	<div class="sliderContainer">
		<p class="sliderValue left">&psi;</p>
		<input id="psiSlider-tb" type="range" min="-3.14" max="3.14" step="0.01" value="0" class="slider">
		<p id='psiSliderText-tb' class="sliderValue right">0.00</p>
	</div>
	<div class="sliderContainer">
		<p class="sliderValue left">&theta;</p>
		<input id="thetaSlider-tb" type="range" min="-3.14" max="3.14" step="0.01" value="0" class="slider">
		<p id='thetaSliderText-tb' class="sliderValue right">0.00</p>
	</div>
	<div class="sliderContainer">
		<p class="sliderValue left">&phi;</p>
		<input id="phiSlider-tb" type="range" min="-3.14" max="3.14" step="0.01" value="0" class="slider">
		<p id='phiSliderText-tb' class="sliderValue right">0.00</p>
	</div>
	<form>
		<button type="button" id="resetButton-tb">reset</button>
		<input type="checkbox" id="guidesCheckbox-tb">
		<label for="guidesCheckbox-tb">hide gimbol</label>
	</form>
	<div id="canvas-tb" class="plotly">
		<script src="/assets/posts/quaternions/plotTaitBryan.js" type="module"></script>
	</div>
</div>

An intuitive way to describe a rotation is with three separate angles. These are known as the Euler angles, after the great mathematician Leonhard Euler who first described them. 
There are many different conventions for Euler angles depending on the axes and orders chosen.
For the interactive graph I have used the Tait-Bryan angle representation, which is often used in engineering.
Here is an illustration:
<figure class="post-figure">
<img class="img-60"
    src="/assets/posts/quaternions/Plane_ZYX.png"
	alt="Tait-Bryran angles"
	>
<figcaption>Modified from <a href="https://en.wikipedia.org/wiki/File:Plane_with_ENU_embedded_axes.svg">Wikipedia</a></figcaption>
</figure>
The three angles are:
- $\psi$: yaw, about the $z$ axis.
- $\theta$: pitch, about the $y'$ axis (Note: $z'$ points downwards so that positive $\theta$ is clockwise).
- $\phi$: roll, about the $x''$ axis.

The order used is yaw-pitch-roll, also called $\psi$-$\theta$-$\phi$ or $z$-$y'$-$x''$ or ZYX. 
As an example of how the order affects the final rotation, here are two rotations done with a yaw of 30&deg; and a roll of 90&deg;. The left uses the yaw-roll order and the right a roll-yaw order. 
<figure class="post-figure">
<img class="img-95"
    src="/assets/posts/quaternions/Cruiser_euler.png"
	alt="Tait-Bryan angles"
	>
<figcaption>Yaw-roll (left) and roll-yaw (right). </figcaption>
</figure>
On the left, the cruiser first rotates 30&deg; to the left and then rolls on its side.
On the right, the cruiser rolls on it side first, so that a pilot sitting inside would perceive his left as _up_, and hence the yaw rotation results in a 30&deg; movement upwards.
This is why the order needs to be specified.

Given this order, each angle can be used to construct a 3&times;3 rotation matrix $R_{\alpha}$ which reprents a 2D rotation about an axis. See [part 2][2Drotations] for more detail.
Then the rotated vector $v_r$ is obtained from $v$ by multiplying each matrix in order:

$$ v_r = R_{\phi}R_{\theta}R_{\psi}v$$

<div class="card">
  <div class="card-body">
    <h5 class="card-title">Matrix defintions</h5>
    <p class="card-text">
		\[
		R_{\psi} =
		\begin{bmatrix} 
			cos(\psi) & -sin(\psi) & 0 \\
			sin(\psi) &  \phantom{+}cos(\psi) & 0 \\
			0 & 0 & 1 
		\end{bmatrix}
		\]
		\[
		R_{\theta} =
		\begin{bmatrix} 
			\phantom{+}cos(\theta) & 0 & sin(\theta) \\
			0 &  1  & 0 \\
			-sin(\theta) & 0 & cos(\theta) 
		\end{bmatrix}
		\]
		\[
		R_{\phi} =
		\begin{bmatrix} 
			1 & 0 & 0 \\
			0 & cos(\phi)  & -sin(\phi) \\
			0 & sin(\phi) &  \phantom{+}cos(\phi) 
		\end{bmatrix}
		\]
	</p>
  </div>
</div>


I would like to emphasise that the above graph is dynamically generated by using this equation to update the 3D co-ordinates for the Plotly charting library.
The result is a natural looking rotation.[^plotly_rotations]
The JavaScript code can be found [here][EulerAngles.js] or with your browser's inspection tools.

If you do follow the order, the base will rotate about one circle of the gimbol at the time.
But what if you don't follow it?
For example, you move $\theta$ and $\phi$ before $\phi$?
Go back and try this if you haven't already.
The answer is, the whole gimbol rotates to the orientation _where it would have been_ if the rotation order of $z$-$y'$-$x''$ was respected.
This path cannot be represented with the angles $\psi$, $\theta$ or $\phi$. (It can be with other Euler angles but that only shifts the problem.)
This already illustrates one of the biggest problems with Euler angles: it enforces unnatural constraints.

To see how this can create complexity, consider the interpolation problem in the initial GIF of this post.
Given the angles for the first and final rotation, how would you find the rotation at the middle point?
<figure class="post-figure">
<img class="img-95"
    src="/assets/posts/quaternions/missing.png"
	alt="cruiser static frames - missing middle"
	>
<figcaption></figcaption>
</figure>
We need to find a circle that connects the nose of the cruiser from its starting position to its final position.
Then for each point on the arc of the circle, we need to find some $\psi$, $\theta$ and $\phi$ applied in that order that will result in a rotation to that point.
This requires adjusting three angles simultaneously. 
One can imagine an algorithm where you adjust $\psi$, then $\theta$ then $\phi$ and if the point doesn't fall on the circle where it is supposed to go, start from the beginning at $\psi$ again.
A better algorithm is provided below, but it is still constrained by the same underlying process.
We'll see shortly that the quaternion algorithm is much, much simpler.

<p>
  <a class="btn" data-toggle="collapse" href="#EulerInterp" role="button" aria-expanded="false" aria-controls="collapseExample">
    More detail: interpolation with Euler angles &#8681;
  </a>
</p>
<div class="collapse" id="EulerInterp">
  <div class="card card-body ">
		<ol>
			<li>
				Using the inner product: $cos(\alpha) = \vec{p}_0 \cdot \vec{p}_1 = x_1 x_2 + y_1 y_2 + z_1 z_2 $
			</li>
			<li>
				The equation of the circle is $\vec{p}_t = rcos(\alpha t)\hat{x} + rsin(\alpha t)\hat{y}$
				<ul>
					<li> Define $r = \lvert  \vec{p}_0 \rvert = \lvert  \vec{p}_1 \rvert $ </li>
					<li> Define $\hat{x} = \frac{1}{r}\vec{p}_0 $ </li>
					<li> Calculate $\hat{y}$ from $\vec{p}_1 = rcos(\alpha)\hat{x} + rsin(\alpha)\hat{y}$  </li>
				</ul>
				We can use $t=0.5$ to find the middle vector.
			</li>
			<li>
				With (2) there is enough information to create the animation, but what if we want the rotations to be stateless? That is, independent of the starting and final position? 
				For example, we want to display the angles back to the user throughout the animation?
				This requires solving 3 highly non-linear trigonometric equations from $p_t=R_{\phi}R_{\theta}R_{\psi}p_0$.
				An alternative is to calculate the axis-angle rotation matrix from the Rodrigues formula below,
				with $\hat{n} = \hat{x} \times \hat{y}$ and $\theta = \alpha t$. Then compare terms with the Euler rotation matrix.
			</li>
		</ol>
  </div>
</div>

Another problem comes from gimbol lock. 
This is not an issue when the rotations are enforced, which is the situation in animations.
But going the other way, where rotations need to be calculated from accelerations and velocities - as is often the case in physics problems - this is a major problem.
In particular, when the pitch is 90&deg;, yaw and roll switch, and this provides enough ambiguity in the equations to inject significant numeric instability. (The same happens at a 90&deg; angle for yaw and roll but only the middle angle causes a problem.)
For this reason, Euler angles should never be used in physics simulations - I speak from experience.

<p>
  <a class="btn" data-toggle="collapse" href="#gimbollock" role="button" aria-expanded="false" aria-controls="collapseExample">
    More detail: proving gimbol lock &#8681;
  </a>
</p>
<div class="collapse" id="gimbollock">
  <div class="card card-body ">
  The symbol $\omega$ is used to denote an angular velocity e.g. $\omega_\alpha = \frac{d\alpha}{dt}$, measured in radians/second. In physics problems we often have the angular velocity in the world frame, $\omega^0$. From this we need to calculate the angular velocities of the Euler angles so that we can integrate them to get the angles and hence calculate the rotation matrix. These angular velocities are related as follows:
		\[
		\begin{align}
		\vec{\omega}^0
		&=
		\omega_{\psi}\hat{z} + \omega_{\theta}\hat{y}' + \omega_{\phi}\hat{x}'' \\
		\begin{bmatrix} 
			\omega_{x}^0 \\
			\omega_{y}^0 \\
			\omega_{z}^0
		\end{bmatrix} 
		&=
		\omega_{\psi}
		\begin{bmatrix} 
			0 \\
			0 \\
			1
		\end{bmatrix}
		+
		\omega_{\theta} R_{\psi}
		\begin{bmatrix} 
			0 \\
			1 \\
			0
		\end{bmatrix}
		+
		\omega_{\phi} R_{\psi}R_{\theta}
		\begin{bmatrix} 
			1 \\
			0 \\
			0
		\end{bmatrix} \\
		&=
		\omega_{\psi}
		\begin{bmatrix} 
			0 \\
			0 \\
			1
		\end{bmatrix}
		+
		\omega_{\theta}
		\begin{bmatrix} 
			-sin(\phi) \\
			\phantom{+}cos(\phi) \\
			0
		\end{bmatrix}
		+
		\omega_{\phi}
		\begin{bmatrix} 
			cos(\phi)cos(\theta) \\
			sin(\phi)cos(\theta) \\
			-sin(\theta)
		\end{bmatrix} \\
		&= 
		\begin{bmatrix} 
			cos(\phi)cos(\theta) & -sin(\phi) & 0 \\
			sin(\phi)cos(\theta) & \phantom{+}cos(\phi) & 0\\
			-sin(\theta)         & 0 & 1
		\end{bmatrix} 
		\begin{bmatrix} 
			\omega_{\phi} \\
			\omega_{\theta} \\
			\omega_{\psi}
		\end{bmatrix} 
		\end{align} 
		\]
		Invert the matrix to get the unknown Euler velocities in terms of the known world velocities:
		\[
		\begin{align}
		\Rightarrow
		\begin{bmatrix} 
			\omega_{\phi} \\
			\omega_{\theta} \\
			\omega_{\psi}
		\end{bmatrix}
		&=
		\frac{1}{cos(\theta)}
		\begin{bmatrix} 
			cos(\psi) & sin(\psi) & 0 \\
			-sin(\psi)cos(\theta) & cos(\psi)cos(\theta) & 0\\
			\phantom{+}cos(\psi)sin(\theta)  & sin(\psi)sin(\theta) & 1
		\end{bmatrix} 
		\begin{bmatrix} 
			\omega_{x}^0 \\
			\omega_{y}^0 \\
			\omega_{z}^0
		\end{bmatrix}
		\end{align} 
		\]
		This expression is undefined for $cos\theta = 0$, which happens at $\theta = n\frac{\pi}{2} \;, n \in \mathbb{Z}$.
		This is where gimbol lock happens.
  </div>
</div>

Concluding this section, we have discovered that Euler angles are intuitive and work well for static systems.
However they are not suited for dynamic systems, whether that be through interpolation or numerical integration.
Thankfully the next method does work well in the latter case.

### 2. An axis and an angle

<div class="plot3d-container">
	Instructions: slide the &alpha; and &beta; sliders to change the normal vector. Slide &theta; to rotate in a circle around the normal vector.
	<div class="sliderContainer">
		<p class="sliderValue left">&alpha;</p>
		<input id="alphaSlider-q" type="range" min="-3.14" max="3.14" step="0.01" value="0" class="slider">
		<p id='alphaSliderText-q' class="sliderValue right">0.00</p>
	</div>
	<div class="sliderContainer">
		<p class="sliderValue left">&beta;</p>
		<input id="betaSlider-q" type="range" min="-3.14" max="3.14" step="0.01" value="0" class="slider">
		<p id='betaSliderText-q' class="sliderValue right">0.00</p>
	</div>
	<div class="sliderContainer">
		<p class="sliderValue left">&theta;</p>
		<input id="thetaSlider-q" type="range" min="-6.28" max="6.28" step="0.01" value="0" class="slider">
		<p id='thetaSliderText-q' class="sliderValue right">0.00</p>
	</div>
	<form>
		<button type="button" id="resetButton-q">reset</button>
		<input type="checkbox" id="guidesCheckbox-q">
		<label for="guidesCheckbox-q">hide guides</label>
	</form>
	<p id="quaternionText-q">
		q = +1.000 + 0.000i + 0.000j + 0.000k
	</p>
	<div id="canvas-q" class="plotly">
		<script src="/assets/posts/quaternions/plotQuaternion.js" type="module"></script>
	</div>
</div>

Note that $\theta$ can rotate fully around a circle in both the clockwise and anti-clockwise direction.
Therefore every point on the circle can be reached twice: through the clockwise rotation or through the anti-clockwise rotation.
This is known as the double cover property.

There are multiple ways to calculate the axis-angle representation. 
This graph uses a quaternion. 
See the source code [here][Quaternion.js] or with your browser's inspection tools.
Other methods are briefly presented in the next section.

What is a quaternion? Here is a mathematical definition: 
<div class="card">
  <div class="card-body">
    <h5 class="card-title">Definition: Quaternion</h5>
	<h6 class="card-subtitle mb-2 text-muted">/kwəˈtəːnɪən/</h6>
    <p class="card-text">
		A quaternion is a number of the form $$s + xi+yj + zk \; ; \; s, x, y, z \in 	\mathbb{R}$$
		where the basis elements $i$, $j$ and $k$ obey the following rules of multiplication: 
		$$i^2 = j^2 = k^2 = ijk = -1 \;,\; ij=k \;,\; ji=-k$$
		By definition multiplication is in general non-commutative: $pq\neq qp$. Otherwise, normal distributive and associative rules of algebra apply.
	</p>
  </div>
</div>
It may be unusual that $ij=k$ but $ji=-k$, but this should not be surprising after the Euler angles section.
3D rotations are themselves non-commutative - they depend on order - and hence any mathematics that represents them must be non-commutative.
[Part 3][quaternions] will provide further physical justifications for this abstract definition.

The quaternion is calculated from the normal vector and $\theta$ as follows:

$$ 
\begin{aligned}
\hat{n} &= cos(\beta)cos(\alpha)i + cos(\beta)sin(\alpha)j + sin(\beta)k \\
		q &= cos(\tfrac{\theta}{2}) + sin(\tfrac{\theta}{2})\hat{n}
\end{aligned}
$$

The rotation is then done with this formula (proved in [part 3][quaternions]):

$$
\begin{aligned}
	v_r &= qvq^{*}\\
		&= (cos \tfrac{\theta}{2} + sin \tfrac{\theta}{2}\hat{n})v(cos \tfrac{\theta}{2}- sin \tfrac{\theta}{2} \hat{n})
\end{aligned}
$$

As already mentioned, there are other ways to calculate this rotation.
Other methods are intriguing in that they are based on 3D geometrical constructions, while quaternions are 4D, which means we cannot properly visualise them.
Why then should we prefer quaternions?
Their main advantage comes with interpolations.
Here again is the problem of finding the middle rotation between the starting and final rotations in the GIF:
<figure class="post-figure">
<img class="img-95"
    src="/assets/posts/quaternions/static.png"
	alt="cruiser static frames"
	>
<figcaption></figcaption>
</figure>
Again we draw the arc of a circle which the nose of the cruiser will travel along.
The axis-angle representation is a natural fit to this problem, because this arc can itself be represented with a normal vector and an angle.
Even so, the quaternion solution to the problem is unexpectedly simple.
For this special case of $t=0.5$, calculate the quaternion:

$$ q_{0.5} = \frac{q_{0.0} + q_{1.0}}{2}$$

and apply the rotation formula. Done.

In general the expression for $q_t$ is more complex, but comparable to similar equations in 2D.
But I hope this example gives a sense of the power of quaternions. 

#### Other forms of axis-angle rotations

Identical rotations could also be accomplished with other formulas, namely the Rodrigues' formulas for axis-angle rotations and with Pauli matrices.
I will give the formulas without going into detail; this is only to compare forms.

The Rodrigues' formulas are as follows:

$$
\begin{aligned}
\vec{v}_r &=  cos \theta \vec{v}+ 
			sin \theta (\hat{n} \times \vec{v})
			+ (1-cos \theta)(\vec{v} \cdot \hat{n}) \hat{n} \\
\vec{v}_r &= [I_3 + sin\theta N + (1-cos \theta) N^2]\vec{v} \;, \; N\vec{v} = \hat{n} \times \vec{v}
\end{aligned}
$$

Note that $\vec{v} = (x, y, z)^T \equiv 0 + xi + yj + zk = v$.


The Pauli matrices formula is a sort of intermediate form between quaternions and the Rodrigues formulas. It uses 2&times;2 matrices and complex numbers:

$$
\begin{aligned}
	\vec{v} \cdot \vec{\sigma} &= 
	\begin{bmatrix}
	z    & x + iy \\
	x - iy & -z
	\end{bmatrix} \\
	U &= cos \tfrac{\theta}{2} I_2 - sin \tfrac{\theta}{2}(i\hat{n} \cdot \vec{\sigma}) \\
	\vec{v}_r \cdot \vec{\sigma} &= U(\vec{v} \cdot \vec{\sigma} ) U^\dagger 
\end{aligned}
$$

Here again is the quaternion formula:

$$
\begin{aligned}
	v_r &= qvq^{*}\\
		&= (cos \tfrac{\theta}{2} + sin \tfrac{\theta}{2}\hat{n})v(cos \tfrac{\theta}{2}- sin \tfrac{\theta}{2} \hat{n})
\end{aligned}
$$

We now have four different formulas which are based on four different branches of mathematics (Euclidean geometry, linear algebra and complex numbers, quaternions) with multiple different types of multiplications (scalar multiplication, quaternion multiplication, dot products, vector cross products and matrix multiplication), yet these formulas all represent the same physical rotation and will result in identical vectors $\vec{v}_r$. 
It is possible through tedious calculations to show that these compact formulas are all equal by using the definitions of the various types of multiplications and expanding them out. This fact can also be numerically verified.

For a comparison of all formulas in Julia, please see this repository: [Rotations.jl][RotationsLior].

[RotationsLior]: https://github.com/LiorSinai/Rotations.jl

Quaternions are best for interpolations and hence animations.
But any of these methods will work for physics simulations.
It is straightforward to calculate the normal vector and angle from angular velocities.
Personally, I have successfully used the Rodrigues matrix formula on simulations of complex 3D robots.

<p>
  <a class="btn" data-toggle="collapse" href="#axisAngleVelocity" role="button" aria-expanded="false" aria-controls="collapseExample">
    More detail: axis-angle formulas from angular velocities &#8681;
  </a>
</p>
<div class="collapse" id="axisAngleVelocity">
  <div class="card card-body ">
		An angular velocity $\vec{\omega}$ represents the angular velocities that an object rotates about the axes of the world frame. The combined effect is to rotate about a circle with a normal axis parallel to $\vec{\omega}$. So the normal vector is $\hat{n} = \frac{\vec{\omega}}{\lvert \vec{\omega} \rvert}$. $\theta$ is a measure of the magnitude of this rotation. $ \theta = \lvert \vec{\omega} \rvert \Delta t = \sqrt{ \omega_x^2  + \omega_y^2  + \omega_z^2 }\Delta t $. These values can then be used directly in any of the axis-angle formulas.
  </div>
</div>


## Outline 

Using quaternions for 3D rotations is a very sensible choice for animation software.
They are essentially an array of four numbers with the normal rules for addition and subtraction and some special rules for multiplication involving minus signs. Encode that, and you get rotation functions and stateless, fast, interpolations almost for free.

I hope this post has illuminated some of their properties and advantages.
If you wish to learn more, the rest of the series will expand more on the mathematics of quaternions. 
- [Part 2][2Drotations] describes rotations in 2D. It uses both matrix and complex number representations.
It is a crash course in complex numbers, which can be seen as a simpler kind of quaternion. 
Some prior experience with complex numbers will help.
- [Part 3][quaternions] describes the fundamentals of quaternions and their mathematics. A few proofs of their properties are given. This is the longest section.
- [Part 4][interpolation] focuses on interpolation. Interpolation formulas are derived in 2D and transferred to 3D.
An interactive graph with a stick aeroplane in place of the Unity Spaceship is presented.

To get the most out of this series, you should be comfortable with trigonometry, algebra, complex numbers, Euclidean geometry and linear algebra (matrices).
This maths was covered in my first year of engineering.

This series is the kind I would have liked to see.
When I first learnt about quaternions I found that I had to consult many sources to understand them properly.
Nearly every source began with a story of an Irish mathematician, a bridge, and an epiphany that caused him to carve the fundamental formula of quaternions into the stone. It's a nice story but it is a confusing one to begin with. 
Why did he have this epiphany? What "magic" did he grasp that day and can we also?
In [part 3][quaternions] I do tell this story properly, but at a point where sufficient mathematics has been discussed so we can somewhat approximate the mathematician's thoughts that day. Instead I chose to lead with a different story; one about why quaternions are still relevant 178 years later. I hope this was appreciated.

This series is written from the perspective of an engineer. 
I try to introduce ideas and justify them in as intuitive a way as possible.
Mathematical proofs are only done for identities where that is difficult.
I also do not explore how quaternions fit into the general context of mathematical fields and algebras,
or more general versions of quaternion algebra.

## References

There are many sources I used to learn about quaternions.
I have tried to make this series as comprehensive as I could, but I still recommend reading these.
Each provides more detail in some area, whether it be animations, mathematical formulas, or visualisations.
- [Quaternions, Interpolation and Animation by Erik B. Dam, Martin Koch and Martin Lillholm (1998)][Dam1998]: a comprehensive guide of quaternion mathematics and how to use them in animations. The main source material for this series.
- [Quaternion Algebras by John Voight (2021)][Voight2021]: a complete textbook on Quaternion Algebras. It shows that quaternions as used in animations are a choice within a larger group of similar algebras. It presents many elegant proofs for the properties of quaternions. But it quickly goes beyond normal quaternions and most certainly this author's knowledge.
- [Representing Attitude: Euler Angles, Unit Quaternions, and Rotation Vectors by James Diebel (2006)][Deibel2006]: a concise guide to Euler angles and quaternions, with formulas for many different kinds and conversion formulas between each type. 
- [Visualizing quaternion by Grant Sanderson and Ben Eater][Sanderson2018]: a stunning interactive website with tutorials on visualising 4 dimensional quaternions in 3 dimensions through stereographic projections.
- [Dirac's belt trick, Topology, and Spin ½ particles by Noah Miller][Miller2021]: an interesting video on the applications of 3D rotations in quantum mechanics. I encountered the Pauli matrices rotation formula for the first time in this video.
- [Wikipedia][Wikipedia]: comprehensive articles on quaternions and related subjects.



[Voight2021]: https://link.springer.com/book/10.1007/978-3-030-56694-4
[Dam1998]: https://web.mit.edu/2.998/www/QuaternionReport1.pdf
[Deibel2006]: https://www.astro.rug.nl/software/kapteyn-beta/_downloads/attitude.pdf
[Sanderson2018]: https://eater.net/quaternions
[Miller2021]: https://www.youtube.com/watch?v=ACZC_XEyg9U
[Wikipedia]: https://en.wikipedia.org/wiki/Quaternion

---

[^quarternion]: It is quate**r**nion for the Latin word for 4, quattuor. Before knowing any better, I assumed it was qua**r**tonion for the anglicized word "quarter". 

[^interpolation]: Interpolation between rotations imposes kinematics without regards to the physics. In engineering it is often the other way round: physics defines the kinematics. In other words, an applied force changes the orientation of an object. This often follows a highly non-linear profile. For example, a strong jerk might cause the object to twist suddenly from one rotation to the next. Interpolation meanwhile sets the rotations, like an invisible hand guiding the object, often with linear schemes resulting in smooth animations.

[^plotly_rotations]: Let's get meta for a moment. Plotly itself gives you the ability to rotate the entire plot. How does it 	do that? It uses the Tait-Bryan angles with the same order used here. The relevant functions from the [source code](https://github.com/plotly/plotly.js/tree/master/stackgl_modules) are `rotateX`, `rotateY` and `rotateZ`.