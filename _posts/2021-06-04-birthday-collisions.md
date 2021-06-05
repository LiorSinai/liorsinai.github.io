---
layout: post
title:  "Intuitive explanations for non-intuitive problems: the Birthday Problem"
date:   2021-06-04
author: Lior Sinai
categories: coding
background: ''
categories: mathematics
tags: mathematics probability
---

_My take on the famous birthday problem._ 


**intuitive** [ɪnˈtjuːɪtɪv] <br>
_Using or based on what one feels to be true even without conscious reasoning; instinctive._ <br>
[Definition from Oxford Languages](https://languages.oup.com/google-dictionary-en/)

This is part of a short series on probability, a field rife with problems that are easy to understand but results that are decidedly not intuitive for people. This post is harder than last week's on the [Monty Hall][MontyHall] problem. Some background in probability will help.

[MontyHall]: {{ "mathematics/2021/05/26/monty-hall" | relative_url }}

# Introduction 

If you've ever encountered two unrelated people who share a birthday, you might have wondered what the chances of it were happening. We can formalise this and ask how many people do you need to have a 50% chance of at least two people sharing a birthday? A 99% chance? A 100% chance? This is the famous [birthday problem][betterExplained_birthdays]. Besides for being an amusing maths problem, it also has important consequences for hashing functions which are used in cryptography and key-value data stores.

[betterExplained_birthdays]: https://betterexplained.com/articles/understanding-the-birthday-paradox/

The answers often surprise people. Given 365 evenly distributed birthdays, the number of people you require for a 50%, 99% and 100% chance are 23, 57 and 366 people respectively. You need more people than possible birthdays to be dead certain that at least the 366th person shares a birthday with someone before him. But with just 1/6th of 365, you're almost guaranteed to have a match. With only 23 people - 1/16th of 365 - you have a 50% chance of at least two people sharing a birthday. 

In general, for $n$ unique items you need $\sqrt{n}$ instances for a 50% chance of collision, and $3\sqrt{n}$ for a 99% chance.

In high school, I was part of two classes of about 25 people each. In the first I shared a birthday with my twin brother, but also with someone else, and two others shared a birthday on another day. In the second, I only shared a birthday with my twin, which doesn't count here because we are related. So in 50% of my classes, there was a 50% match. 

<figure class="post-figure" id="probability_graphs">
<img class="img-80" 
    src="/assets/posts/birthday-collisions/cdfs.png"
	alt="probability graphs"
	>
	<figcaption></figcaption>
</figure>

This trend holds for larger samples.
I have over 400 Facebook friends, and of these 379 have publically available birthdays. When I sample from these people at random, I find collisions with a probability distribution following the blue curve above. This is very similar to the red curve, which is the exact mathematical probability that can be calculated for a large population of people. The green curve is an approximate probability.

So why is the number of people required so low? The short answer is the large amount of combinations of pairs that can share birthdays. This number rises exponentially but most people apply a linear heuristic to this problem, which leads them to massively overestimate the answer.

Let's investigate further.

## The probability that you share a birthday with someone else

Imagine you are in a circle with 23 other people. What is the probability you share a birthday with someone else? This is specific to you. It does not matter if two other people share a birthday; it only matters if one other person shares a birthday with you.

<figure class="post-figure" id="pairs1">
<img class="img-80" 
    src="/assets/posts/birthday-collisions/pairs1.png"
	alt="1 person paired with 23 others"
	>
	<figcaption></figcaption>
</figure>

To simplify calculations, it is easiest to assume that sharing a birthday with someone is independent of sharing it with anyone else. This is not entirely accurate. For example, if person A and person B share a birthday, then the probability of sharing a birthday with person B depends on the probability of sharing it with person A. In the highly unlikely case of everyone else having the same birthday, then the probablity of sharing with anyone depends on everyone. But the assumption of independence is good because (1) most people will not share birthdays and (2) the events of multiple people sharing birthdays are very unlikely.

Under the assumption sharing is independent, you can imagine going to each and every person and asking them if they have the same birthday as you. There is a 1/365 chance they will say yes, and a 364/365 chance they will say no. For exactly one other person sharing a birthday with you, this is a standard binomial probability where there are 22 ways where 1 person will say yes and 21 will say no:

$$ \begin{align} P(x=1) &= 22 \left(\frac{1}{365}\right)^1 \left(\frac{364}{365}\right)^{21} \\
                            &=0.0567 \end{align}
							$$ 

But you could also share a birthday with two others, or three or four or everyone. These events are more unlikely than sharing with just one other person, but they will add to the total. Instead of calculating them all, it's easier to work backwards and calculate the probability no-one shares a birthday with you and subtract it from 1. There is 1 way 22 people will say no:

$$ \begin{align} P(x\geq 1) &= 1 - \left(\frac{364}{365}\right)^{22} \\
                            &=0.0586 \end{align}
							$$ 

This is close to 23/365 which is about 6%. So far, the numbers are close to our expectations.

## The probability that at least some people share a birthday

Your friend is sitting next to you in this circle. You now want to know the probability that either you or him share a birthday with someone. Intuitively this probability should be higher.

<figure class="post-figure" id="pairs2">
<img class="img-80" 
    src="/assets/posts/birthday-collisions/pairs2.png"
	alt="2 people paired with 23 others"
	>
	<figcaption></figcaption>
</figure>

You have already asked 22 people, including your friend. So he now only needs to ask 21 people. Note that you have not compared your friend's birthday with anyone else. Since you both "win" this game if one of you wins, losing is the equivalent of getting another 21 negatives, or $22+21=43$ negatives in total:

$$ \begin{align} P(X\geq 1) &= 1 - \left(\frac{364}{365}\right)^{43} \\
                            &=0.1113 \end{align}
							$$ 

Your friend does get all negatives. So you bring in a second friend. She only needs to ask 20 people if they share a birthday with her:

$$ \begin{align} P(X\geq 1) &= 1 - \left(\frac{364}{365}\right)^{43+20} \\
                            &=0.1587 \end{align}
							$$ 

You've more than doubled your original probability of 6% to 16%. Now let us go and compare with everyone else. 

<figure class="post-figure" id="pairs23">
<img class="img-80" 
    src="/assets/posts/birthday-collisions/pairs23.png"
	alt="23 people paired with 23 others"
	>
	<figcaption></figcaption>
</figure>

In total you will compare $22+21+...+2+1=253$ pairs of people. The final probability is:

$$ \begin{align} P(X\geq 1) &= 1 - \left(\frac{364}{365}\right)^{253} \\
                            &=0.5005 \end{align}
							$$ 

The fraction 253/365 is 69%, so it is an overestimate of the real probability of 50%. But this does make sense because the chance of collision does not increase linearly. You could still compare pairs 1000 times and fail.

In general, the number of pairs that can be formed is ${k \choose 2} = \frac{k(k-1)}{2}$. This is $k$ groups of pairs with $k-1$ other people divided by 2 because you count every pair twice.
So the number of combinations increases by $k^2$, which is why the number of people required for a collision with $n$ distinct items is $\sqrt{n}$.


## The probability that two of my Facebook's friends share a birthday

My 374 Facebook friends are real people. I have added them on Facebook over 14 years. They range from people I have only met once to my closest friends and family. Here is the distribution of the 374 birthdays per month:[^twins]

<figure class="post-figure" id="birthday_months">
<img class="img-80" 
    src="/assets/posts/birthday-collisions/birthdays_month.png"
	alt="birthdays per month of my FB friends"
	>
	<figcaption></figcaption>
</figure>

There are approximately 30 birthdays a month. Some months have more and some less. It can be considered a rough approximation of a uniform distribution of birthdays.

65% of my friends share a birthday with at least one other unrelated person. On one day there is as much as seven people sharing a birthday. This is the distribution of people per birthday: 

<figure class="post-figure" id="shared_birthdays">
<img class="img-80" 
    src="/assets/posts/birthday-collisions/shared_birthdays.png"
	alt="shared birthdays frequency"
	>
	<figcaption></figcaption>
</figure>

I calculated the probability of picking at least two people who shared a birthday with Monte Carlo simulations.
That is, I ran many simulations where I randomly picked a given number of people, say 20, and checked if any shared a birthday. 
I did this 10,000 times for each number of people and saved the mean number of trials with collisions as the probability.
Here is the pseudo code:

{%highlight C# %}
num_trials = 10,000
samples = ListOfBirthdays
prob = Array(size=60)
for num_people in 1 to 60:
	num_collided = 0
	for trial in 1 to num_trials
		subset = randomUniqueFrom(samples, n=num_people)
		if hasCollisons(subset):
			num_collided += 1		
	prob[num_people] = num_collided/num_trials
{% endhighlight %}


Many of my friends share a birthday. But with picking any random 30 you're only slightly more likely to get the other people who have the same birthday.
Hence the [blue curve](#probability_graphs) is only slightly above the theoretical curve.

# Conclusion

I hope you enjoyed this article and now believe the chances, even if it still feels unintuitive.
I suggest trying it out with your own Facebook friends or LinkedIn connections.

If you want more in-depth mathematics please see my next part on [Advanced mathematics for the birthday problem][BirthdayCollisionsAdvanced].

[BirthdayCollisionsAdvanced]: {{ "mathematics/2021/06/05/advanced-collisions" | relative_url }}

---

[^twins]: I am friends with five pairs of twins (excluding my twin). I have only counted one of each twin to avoid modelling this dependency relationship.
