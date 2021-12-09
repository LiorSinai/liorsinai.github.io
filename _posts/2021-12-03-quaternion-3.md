---
layout: post
title:  "Quaternions: Part 3"
date:   2021-12-03
author: Lior Sinai
categories: mathematics
tags: mathematics quaternions unity rotation 3d
---

_A detailed foundation of quaternion mathematics._ 


This is part of a series. The other articles are:
- [Part 1: introduction][introduction].
- [Part 2: 2D rotations][2Drotations].
- [Part 4: interpolation][interpolation].

[introduction]: {{ "mathematics/2021/11/05/quaternion-1-intro" | relative_url }}
[2Drotations]: {{ "mathematics/2021/11/28/quaternion-2-2d" | relative_url }}
[quaternions]: {{ "mathematics/2021/12/03/quaternion-3" | relative_url }}
[interpolation]: {{ "mathematics/2021/12/06/quaternion-4-interpolation" | relative_url }}


### Table of Contents
- [Extending the complex numbers](#extending-the-complex-numbers)
- [Quaternion foundations](#quaternion-foundations)
- [Properties of quaternions](#properties-of-quaternions)
	- [Addition and subtraction](#addition-and-subtraction)
	- [Multiplication](#multiplication)
	- [Inner product](#inner-product)
	- [Conjugation](#conjugation)
	- [Inverse and division](#inverse-and-division)
	- [Unit quaternions](#unit-quaternions)
	- [Exponential, logarithm, and power functions](#exponential-logarithm-and-power-functions)
- [Rotations with quaternions](#rotations-with-quaternions)
- [Conclusion](#conclusion)


# Quaternions
## Extending the complex numbers

In 1843, Sir William Rowan Hamilton was seeking to extend the complex numbers to 3 dimensions.
Complex numbers had been introduced in 1572 and it had taken a while for them to gain popular acceptance.
René Descartes coined them imaginary numbers in 1637.
Leonhard Euler and his peers had greatly improved our understanding of them in the 1700s.
Carl Freidrich Gauss - a famous contemporary mathematician of Hamilton - had proved more theorems involving complex numbers in the early 1800s.
By 1843, it was time to extend them.

Hamilton had already tried to do this for 10 years. The obvious idea was to try a number of the form:

$$ z = a + bi+ cj \;,\; i^2=j^2=-1 $$

This however leads to an inconsistent algebra.
Without going deep into the theory, a mathematical algebra needs to be consistent so that we can use logic to unambiguously derive more conclusions.
For example, in real number algebra there is only one number that can satisfy $xx^{-1}=1$ for a given $x$ because every number has a unique inverse. If $x=6$, then $x^{-1}$ can only be $\tfrac{1}{6}$.[^ambiguities]

<div class="card">
  <div class="card-body">
    <h5 class="card-title">Proof of inconsistency</h5>
    <p class="card-text">
		This is a proof by contradiction. There must be real numbers $a$, $b$ and $c$ that satisfy 
		$$ ij = a + bi + cj \; ; \; a,b,c \in \mathbb{R}$$
		But if we multiply by $i$:
		$$ 
		\begin{align}
		\times i: \; -j &= ai - b + cij \\
						&=	ai - b + c(a + bi + cj) \\
		\therefore 0    &= (ac - b) + i (a + bc) + j (c^2+1)
		\end{align}
		$$
		This requires that all coefficients equal zero, but then $c^2=-1$ and thus $c$ must also be imaginary.
		This is a contradiction because $c$ must be real.
	</p>
  </div>
</div>

Another problem is that multiplication must preserve length. 
With complex numbers, we take it for granted that:

$$ \| z_1 \| ^2 \| z_2 \| ^2 = \| z_1 z_2 \| ^2 $$

This is because multiplying two complex numbers results in a third complex number:

$$ (a_1 + b_1i)(a_2 + b_2i) = (a_1a_2 - b_1b_2) + (a_1b_2+b_1a_2)i$$

For the 3D complex number this is not always possible.
This can be proved for complex numbers with only integers through [Legendre's three-square theorem][legendre] which states that integers of the form $n=4^a(8b+7)$ cannot be represented as the sum of three squares of integers.
For example:

$$ (0^2 + 1^2 + 2^2)(2^2 + 2^2 + 2^2)=60=4(8 + 7)$$

Hence we have a number that is a product of two sums of three squares, but it cannot be written as a sum of three squares, and so cannot be the magnitude of a 3D complex number.[^3squares]

[legendre]: https://en.wikipedia.org/wiki/Legendre%27s_three-square_theorem

If we multiply two of these 3D complex numbers out, we get terms in $ij$ that have no meaning:

$$ 
\begin{align}
(a_1 + b_1i + c_1j)(a_2 + b_2i + c_2j) =& (a_1a_2 - b_1b_2 -c_1c_2) + \\
								       & (a_1b_2+b_1a_2)i + (a_1c_2+c_1a_2)j  + \\
									   & b_1c_2ij + b_2c_1ji 
\end{align}
$$


The final problem is that rotations in 3D are ordered. Therefore the mathematics describing them must be ordered. This is the same as saying it must be non-commutative: $ab\neq ba$.


## Quaternion foundations

<figure class="post-figure">
<img class="img-60"
    src="/assets/posts/quaternions/Broom_Bridge_Quaternions_Hamilton.jpg"
	alt="Quaternion Plaque on Broom Bridge"
	>
<figcaption>Quaternion plaque on Broom Bridge, Dublin. Source: <a href="https://en.wikipedia.org/wiki/File:Inscription_on_Broom_Bridge_(Dublin)_regarding_the_discovery_of_Quaternions_multiplication_by_Sir_William_Rowan_Hamilton.jpg">Wikipedia</a></figcaption>
</figure>

Owing to the above problems, Sir William Rowan Hamilton had difficulties multiplying triples, as he called them.
But on 16 October 1843, as he was walking past a canal in Dublin, Hamilton had the great insight that he needed a fourth dimension to describe 3D rotations. 
Three independent axes and a fourth to describe the size of the scaling.
Overcome with euphoria, Hamilton carved the fundamental equation $ijk=-1$ into the Broom bridge (also called Brougham Bridge). The engraving faded but a plaque has been put in its place.

[Voight2021]: https://link.springer.com/book/10.1007/978-3-030-56694-4

Here is a definition of these quaternions:

<div class="card">
  <div class="card-body">
    <h5 class="card-title">Definition: Quaternion</h5>
	<h6 class="card-subtitle mb-2 text-muted">/kwəˈtəːnɪən/</h6>
    <p class="card-text">
		A quaternion is a number of the form $$s + xi+yj + zk \; ; \; s, x, y, z \in 	\mathbb{R}$$
		where the basis elements $i$, $j$ and $k$ obey the following rules of multiplication: 
		$$i^2 = j^2 = k^2 = ijk = -1 \;,\; ij=k \;,\; ji=-k$$
	</p>
  </div>
</div>

<figure class="post-figure">
<img class="img-60"
    src="/assets/posts/quaternions/ijk.png"
	alt="Relationship between the basis units i, j and k."
	>
<figcaption>Relationship between the basis units i, j and k.</figcaption>
</figure>

Like with the imaginary number $i$, each of the basis units corresponds to a 90&deg; rotation about an axis.
Consider $ij=k$. This has two interpretations:
- The point $i$ post-multiplied by $j$, resulting in a clockwise rotation about the $j$ axis to the point $k$ (red solid line).
- The point $j$ pre-multiplied by $i$, resulting in an anti-clockwise rotation about the $i$ axis to the point $k$ (green dashed line).

Similarly for $ji=-k$:
- The point $j$ post-multiplied by $i$, resulting in a clockwise rotation about the $i$ axis to the point $-k$ (green solid line).
- The point $i$ pre-multiplied by $j$, resulting in an anti-clockwise rotation about the $j$ axis to the point $-k$ (red dashed line).

In general, pre-multiplication results in anti-clockwise rotations and post-multiplication in clockwise rotations. 

I like to use the following mnemonic to remember the rules:

<figure class="post-figure">
<img class="img-20"
    src="/assets/posts/quaternions/mnemonic.png"
	alt="Mnemonic for rules for i, j and k."
	>
<figcaption></figcaption>
</figure>

Going anti-clockwise with the arrows results in postive products, such as $ jk = i$, but going clockwise against the arrows results in negative products, like $kj = -i$.

Unlike complex numbers, $i^2=j^2=k^2=-1$ has no direct interpretation because these numbers lie in the fourth dimension.[^irony]

## Properties of quaternions


### Addition and subtraction

Addition and subtraction are done using normal algebra rules:

$$
\begin{align}
	q_1 + q_2 &= (s_1 + x_1i+y_1j + z_1k) +  (s_2 + x_2i+y_2j + z_2k) \\
			  &= (s_1 + s_2) + (x_1 + x_2)i + (y_1 + y_2)j + (z_1 + z_2) k \\
	q_1 - q_2 &= (s_1 + x_1i+y_1j + z_1k) -  (s_2 + x_2i+y_2j + z_2k) \\
			  &= (s_1 - s_2) + (x_1 - x_2)i + (y_1 - y_2)j + (z_1 - z_2) k
\end{align}
$$

### Multiplication

Quaternions obey the distributive law, so multiplication by a scaler can be applied to each element individually. In this case, the multiplication is commutative:

$$
\begin{align}
	q r &= rq = rs + rxi + ryj + rzk
\end{align}
$$

Multiplication by a quaternion is also done with the distributive law:

$$
\begin{align}
	q_1   q_2 =& (s_1 + x_1i+y_1j + z_1k) (s_2 + x_2i+y_2j + z_2k) \\
			  =&  s_1 (s_2 + x_2i+y_2j + z_2k) + x_1i (s_2 + x_2i+y_2j + z_2k) + \\
			   &  y_1j (s_2 + x_2i+y_2j + z_2k) + z_1k (s_2 + x_2i+y_2j + z_2k)  \\
			  =&  s_1s_2 + s_1x_2i+s_1y_2j + s_1z_2k + x_1s_2i - x_1x_2 + x_1y_2 k - x_1z_2j + \\
			   &  y_1s_2j - y_1x_2k - y_1y_2 + y_1z_2 i + z_1s_2k + z_1x_2j - z_1y_2i - z_1z_2 \\ 
			  =&  (s_1s_2 - x_1x_2 - y_1y_2 - z_1z_2 ) + (s_1x_2 + x_1s_2 + y_1x_z - z_1y_2 )i + \\
			   &  (s_1y_2 - x_1z_2 + y_1s_2 + z_1x_2 )j + (s_1z_2 + x_1y_2 - y_1x_2 + z_1s_2 )k
\end{align}
$$

In general, $q_1q_2 \neq q_2 q_1$. This is only true if $q_1=q_2$ or if either quaternion is a scalar.

When writing code for quaternions it is easiest to hard code this 16 term expression as the result of a multiplication of two quaternions rather than writing the rules for $i$, $j$, $k$.
This was done in the [code][Quaternion.js] for the quaternion in [part 1][part1].

[part1]:  {{ "mathematics/2021/11/05/quaternion-1-intro#2-an-axis-and-an-angle" | relative_url }}
[Quaternion.js]: {{ "assets/posts/quaternions/Quaternion.js" | relative_url }}

### Inner product

The [inner product][wiki_inner] is another type of multiplication that is useful.
It is defined as the sum of the product of corresponding co-ordinates:

$$ q_1 \cdot q_2 = s_1s_2 + x_1x_2 + y_1y_2 + z_1z_2 $$

The inner product of a quaternion with itself is its magnitude:

$$ q \cdot q = \| q \| ^2 = s^2 + x^2 + y^2 + z^2 $$

In 2 or 3 dimensions the inner product can be directly related to the angle $\Omega$ between two vectors.
We can simply extend this formula to quaternions even though we cannot visualise an angle between 4D vectors:

$$ q_1 \cdot q_2 =  \| q_1 \|  \| q_2 \| cos\Omega $$

In [part 4][interpolation] this will be used to make a smooth function for interpolation.

[wiki_inner]: https://en.wikipedia.org/wiki/Dot_product


### Conjugation

For convenience, the conjugate $q^{*}$ of a quaternion is defined as:

$$ q^{*} = (s + xi + yj + zk)^{*} = s - xi - yj - zk $$

The conjugate can be used to calculate the magnitude of a quaternion:

$$ \| q \| ^2 = q q^{*} = s^2 + x^2 + y^2 + z^2 $$

### Inverse and division

The inverse of a quaternion $q^{-1}$ is the unique quaternion that satisfies $qq^{-1} = 1 + 0i + 0j +0k $.
This quaternion is given by:

$$ q^{-1} = \frac{q^{*}}{\| q \| ^2 }$$

Proof:

$$
qq^{-1} = q\frac{q^{*}}{\| q \| ^2 } = \frac{\| q \| ^2}{\| q \| ^2} = 1
$$

Division is equivalent to multiplication by the inverse of a quaternion: 

$$ q_1/q_2 = q_1q_2^{-1} $$

### Unit quaternions

A quaternion with magnitude 1 is called a unit quaternion. 
Unit quaternions lie on a hypersphere of 4 dimensions, which is not possible to visualise.
However we can get a feel of it through values:

| $ \|q\| < 1     $         | $\|q\| = 1     $                | $ \|q\| > 1     $        |
|---------------------------|---------------------------------|--------------------------|
|$ 0 $                      | $  1    $                       | $10   $                  |
| $0.8i  $                  | $1i $                           | $    1i + 2j + 3k      $ |
|$ 0.1 + 0.1i + 0.1j + 0.1k $ | $0.5 + 0.5i + 0.5j + 0.5k  $  | $0.9 + 0.9i + 0.9j +0.9k$|
|$ 0.1 + 0.2i + 0.3j + 0.4k$| $0.4 + 0.5i + 0.6j+\sqrt{0.23}k$| $0.5 + 0.6i + 0.7j +0.8k$|



For a unit quaternion, the inverse is $q^{-1} = q^{*}$.

Using a normal vector $\hat{n}=n_xi + n_yj + n_zk$ with magnitude $ \| \hat{n} \| = 1$, the following unit quaternion can always be constructed:

$$ q = cos(\theta) + \hat{n} sin(\theta) $$

### Exponential, logarithm, and power functions

We can define an exponential-like function for unit quaternions similar to the complex number exponential, which will have similar properties:

$$ e^{\hat{n}\theta} = cos(\theta) + \hat{n} sin(\theta) $$

With an inverse logarithm-like function:

$$ log(cos(\theta) + \hat{n} sin(\theta)) = log(e^{\hat{n}\theta}) = \hat{n}\theta $$

We can then find a similar formula to de Moivre's formula:

$$ q^t = (e^{\hat{n}\theta})^t = e^{\hat{n}\theta t} = cos(\theta t) + \hat{n} sin(\theta t) \; , \; \| q \| = 1 $$

Please see this [Wikipedia][wiki_exp] article for formulas with non-unit quaternions.
Please see [Quaternions, Interpolation and Animation by  Dam, Koch and Lillholm][Dam1998]
for more formal proofs of powers with quaternions.

[wiki_exp]: https://en.wikipedia.org/wiki/Quaternion#Exponential,_logarithm,_and_power_functions
[Dam1998]: https://web.mit.edu/2.998/www/QuaternionReport1.pdf

## Rotations with quaternions

Finally, we have enough knowledge of quaternions to calculate rotations.

Multiplying a 3D vector by a 4D quaternion will result in a rotation and scaling in 4 dimensions.
To avoid scaling, we use only unit quaternions.
To avoid a 4th dimension, we pre-multiply by the quaternion and post-multiply by its conjugate:

$$ v_r = qvq^{*} $$

This type of operation which uses an operator and then its inverse is called a commutator. 

This formula works because 
1. Pre-multiplying rotates a vector anti-clockwise and post-multiplying rotates it clockwise.
2. Rotating about a negative axis reverses the direction of rotation. 

When we combine these effects, post-multiplying with a negative axes (because of the conjugate) rotates the vector anti-clockwise, but the 4th dimension part is still rotated clockwise and hence cancels out with the pre-multiplication.

Here is a more formal proof. This is a nice proof because working through it gives further insight into problem:

<div class="card">
  <div class="card-body">
    <h5 class="card-title">Proof of rotation formula</h5>
    <p class="card-text">
		Consider a vector $v=xi+yj+zk$ rotated by a quaternion $q = cos\theta + \hat{n}sin\theta$.
		The co-ordinate axes is arbitrary, so define a new axis in line with the normal axis $\hat{n}$
		so that $k' = \hat{n}$. Then $v$ can be represented in these new co-ordinates as $v=x'i'+y'j'+z'k'$.
		<p>
	<a class="btn" data-toggle="collapse" href="#transform" role="button" aria-expanded="false" aria-controls="collapseExample">
		More detail: transformation of co-ordinates &#8681;
	</a>
	</p>
		<div class="collapse" id="transform">
		<div class="card card-body ">
		 This proof relies on a co-ordinate transform which is done in the abstract.
			Here is a concrete derivation.
			Define:
			$$ 
			\begin{align}
			k' &= \hat{n} = n_x i + n_y  j + n_z k \\
						&= cos(\beta) cos(\alpha) i + cos(\beta) sin(\alpha) j + \sin(\beta) k
			\end{align}
			$$
			For the $i'$ axis we can choose any vector on the unit circle perpendicular to $k'$. 
			This vector will satisfy the inner product $i' \cdot k' = 0$. 
			Choose $i' = ai +bj + 0k$. Solving for $a$ and $b$:
			$$ 
			\begin{align}
			i' \cdot k'  &= a n_x + b n_y + 0 \\
			\therefore \; a &= -sin(\alpha) \;, \; b = cos(\alpha)
			\end{align}
			$$
			Then using the definiton of a quaternion, $j'=k'i'$:
			$$ j' = -sin(\beta)cos(\alpha)i - sin(\beta)sin(\alpha)j + cos(\beta)k$$
			As verfication, $i'j'=k'$. The transformation is then:
			$$
			\begin{bmatrix}
			i' \\
			j' \\
			k' 
			\end{bmatrix}
			=
			\begin{bmatrix}
			-sin\alpha & cos\alpha & 0 \\
			-sin\beta cos\alpha & -sin\beta sin\alpha & cos\beta \\
			\phantom{+}cos\beta cos\alpha & \phantom{+}cos\beta sin\alpha & sin \beta
			\end{bmatrix}
			\begin{bmatrix}
			i \\
			j \\
			k
			\end{bmatrix}
			$$
			Reversing the transform is done with $R^{-1} = R^T$:
			$$
			\begin{bmatrix}
			i \\
			j \\
			k 
			\end{bmatrix}
			=
			\begin{bmatrix}
			-sin\alpha & -sin\beta cos\alpha & cos\beta cos\alpha \\
			\phantom{+}cos\alpha  & -sin\beta sin\alpha & cos\beta sin\alpha  \\
			0       & cos\beta           & sin \beta
			\end{bmatrix}
			\begin{bmatrix}
			i' \\
			j' \\
			k'
			\end{bmatrix}
			$$
			Transforming the co-ordinates of the original vector:
			$$
			\begin{align}
				v =& xi + yj + zk \\
				=& x(-sin\alpha i' -sin\beta cos\alpha j' + cos\beta cos\alpha k' ) +\\
				& y(cos\alpha  i' -sin\beta sin\alpha j' + cos\beta sin\alpha k')  + \\
				& z(0 + cos\beta j' + sin \beta k') \\
				=& (-xsin\alpha + ycos\alpha) i' + \\
				& (-xsin\beta cos\alpha + ysin\beta sin\alpha + z cos\beta) j' + \\
				& (+xcos\beta cos\alpha + y cos\beta sin\alpha + z sin\beta) k' \\
				=& x'i' + y'j' + z'k' 
			\end{align}
			$$
			It's much easier to keep this abstract!
		</div>
		</div>
		Therefore without loss of generalization, we consider only $\hat{n}=k$:
		$$ 
		\begin{align}
			qvq^{*} =& (cos\theta + k\sin\theta)(xi+yj+zk)(cos\theta - k\sin\theta) \\
					=& ((xcos\theta - y sin\theta)i + (xsin\theta + ycos\theta)j + zcos\theta k -zsin\theta)  (cos\theta - k\sin\theta) \\
					=& -z cos\theta sin\theta + z cos\theta sin\theta + (x(cos^2\theta - sin^2\theta ) -2ysin\theta cos\theta) i +\\
					 & (2xsin\theta cos\theta + y(cos^2\theta - sin^2\theta ) )j + z(cos^2\theta + sin^2\theta) k \\
					=& 0 + (xcos2\theta - ysin2\theta)i + (xsin2\theta + ycos2\theta)j + zk		 
		\end{align}
		$$
		This formula should be recognisable from <a href="/mathematics/2021/11/28/quaternion-2-2d">part 2</a>. It is the vector $v$ rotated by the angle $2\theta$ about the axis $\hat{n} =k$. Note that $z$ which is parallel to $k$ is unchanged.
		Also note that on line 2 the rotation is correct for $i$ and $j$, but the parallel part is partly rotated into the 4th dimension. Hence we need to multiply by $q^{*}$ to correct this.
	</p>
  </div>
</div>

Because the vector is rotated by twice the angle, $q$ is usually chosen to rotate half an angle, so that $qvq^{*}$ rotates one angle:

$$ q= cos(\tfrac{\theta}{2}) + \hat{n}sin(\tfrac{\theta}{2})$$

## Conclusion

Congratulations for reaching this far. 
I hope you now know the basics of quaternions.
There is enough detail here to write a fully functioning quaternion implementation.
Most of these functions can be found in the [code][Quaternion.js] for the graph in [part 1][part1].
The next and last section focuses on using interpolations for animations with quaternions.

---

[^ambiguities]: Some ambiguities do exist through multi-valued functions. For example if $x^2=1$ , then $x$ can be $1$ or $-1$. But these ambiguities can be dealt with if the underlying algebra is consistent.

[^3squares]: A solution can be constructed if we allow non-integers, for example $4^2 + 6^2 + \sqrt{8}^2 = 60$. However then the solution is not closed under addition and multiplication (because we have use a square root), which hinders efforts to make a consistent algebra.

[^irony]: An irony of quaternions is that the real part is in the 4th dimension and is therefore more "imaginary" than the three imaginary parts $i$, $j$ and $k$.