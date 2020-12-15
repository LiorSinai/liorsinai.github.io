---
layout: post
title:  "Thoughts on Julia after 2 weeks"
date:   2020-12-15
author: Lior Sinai
categories: coding
tags:	'julia'
---

_Julia is a fast, flexible and robust language. Having used Julia for 2 weeks, and Python for 7 years, I can already say I prefer Julia. It is not as mature as Python, but I believe it has the potential to far exceed it._ 


# Introduction

It was Ars Technica's "[The unreasonable effectiveness of the Julia programming language][arstechnica]" that finally convinced me to learn the Julia programming language.
For years, I've heard whispers and glowing praises about Julia. 
It's a dynamic programming language, but was said to be much faster than Python, and even as fast as C.
The terse, clean syntax is supposed to allow very generic but also robust code creation, reducing friction to collaboration. 
On top of that, it has math friendly syntax like in Matlab or R but also with real maths symbols. 
All of this has been driving adoption of Julia in academia, a highly scientific, collaborative environment with very high computational needs.
The adoption in industry has been predictably slower.

[arstechnica]: https://arstechnica.com/science/2020/10/the-unreasonable-effectiveness-of-the-julia-programming-language/
[julia_academy_comp]: https://juliaacademy.com/p/computational-modeling-in-julia-with-applications-to-the-covid-19-pandemic

<figure class="post-figure">
<img class="img-95"
    src="/assets/posts/julia-review/SIR epidemic model.png"
	alt="SIR model"
	>
</figure>

So does Julia live up to the hype? I decided to find out.
To this end, I completed the recently released Julia Academy course "[Computational Modeling in Julia with Applications to the COVID-19 Pandemic][julia_academy_comp]".
This is a 16 hour course which works with real Covid-19 pandemic data, and teaches you how to implement SIR epidemiology models as well. 
The above picture is a snapshot of agents moving around a grid with an infection spreading when the touch, and the resulting infection outbreak graphs.[^SIR]
After completing this course I challenged myself to rewrite my Random Forest Python [code][Python_repo] in Julia and also my corresponding blog [post][Python_post].
You can see the Julia code [here][Julia_repo] and the twin blog post [here][Julia_post]. 
This resulted in code of similar length, but that was 9 times faster and that felt much more robust.

The table below has a quick comparison of the Python and Julia Random Forests fitting times.
This was on the Universal Bank Loan data, with 4000 training samples and 1000 test samples.
The random forest had 20 trees, with each tree having 40 to 120 leaves. 
Tests were run from the Anaconda CMD and Julia REPL.
<table>
<caption style="caption-side:bottom">*after first compile run for Julia. Time to first compile was 4.6s</caption>
<thead>
  <tr>
    <th> </th>
    <th>Number of runs</th>
    <th>Scikit-learn</th>
    <th>Python</th>
    <th>Julia</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Fitting time (s)*</td>
    <td style="text-align:right">10</td>
    <td style="text-align:right">0.04810 ± 0.00951</td>
    <td style="text-align:right">6.87132 ± 0.31106</td>
    <td style="text-align:right">0.73991 ± 0.04208</td>
  </tr>
  <tr>
    <td>ratio mean times</td>
    <td> </td>
    <td style="text-align:right">1</td>
    <td style="text-align:right">142.85</td>
    <td style="text-align:right">15.38</td>
  </tr>
  <tr>
    <td>test accuracy (%)</td>
    <td style="text-align:right">10</td>
    <td style="text-align:right">98.43 ± 0.58</td>
    <td style="text-align:right">98.56 ±  0.46</td>
    <td style="text-align:right">98.66 ± 0.37</td>
  </tr>
</tbody>
</table>
Scikit-learn is still the fastest, but that code is more heavily optimised than mine, and also it use parallel processing.

[Python_post]: /coding/2020/09/29/random-forests.html
[Python_repo]: https://github.com/LiorSinai/randomForests
[Julia_post]: /coding/2020/12/14/random-forests-jl.html
[Julia_repo]: https://github.com/LiorSinai/RandomForest-jl


In general, my experience with Julia was very positive. 
It is indeed fast and powerful, and the syntax is very nice.
It is not an upgrade to Python, but I am going to frame much of this article as such.
It operates under some very different paradigms - for example, Julia is very much a functional programming language, whereas Python is object-orientated.
The creators themselves tried to integrate the best of several different languages into Julia including Python, R, Matlab, Ruby, C and Lisp
(see their 2012 [release statement][why_julia]). But my experience is mostly with Matlab, C++ and Python. 
Of these, Julia is most likely to replace the code I write with Python. 

C++ is a complex and powerful static typed language with memory management capabilities, and I don't see Julia replacing mission critical software written with it.
Matlab is proprietary software which has great support for specialised scientific computing.
Its Simulink control software and image processing toolbox are the nicest of their kind that I've used.
Python, however, is a different story. Python is easy to learn and lovely to tinker with, but as soon as your project expands, that joy dissipates.
It's noticeably slower than other languages. It has no type checking and very generous scoping rules.
This makes you think less when writing your own code, which is great, but it makes you think _more_ when reviewing someone else's.
Did they intend this variable to be an array? A string? A custom type?
Popular packages go out of their way to _not_ use Python for core processing, including Numpy, Scikit-learn and TensorFlow (they use, C, C++ and Cython).
Julia promises to fix these many issues, and it's a relief.

I want to state upfront that my main frustration with Julia is that it is not as mature as Python.
The release of Julia 1.0 was just over two years ago; Python 1.0 was released 25 years ago.
The Julia community is playing catch-up with Python, and has the second-mover advantage of knowing what works and what doesn't.
But there simply are not as many packages, features, tutorials or videos as Python has. 
Fewer people ask questions on StackOverflow or Discourse.
Like many opensource projects, the documentation is often lacking.[^pie]
The language itself is changing fast, and some code on Julia Academy's own online tutorials is already out of date.[^out_of_date]
Then there is all the massive amounts of legacy code in companies and institutions.
So if you're a beginner programmer, you can stop reading now. My advice is focus on Python. It has more resources and will get you further.
But if you can relate to my frustrations with Python or have more of your own, read on.


[why_julia]: https://julialang.org/blog/2012/02/why-we-created-julia/



# Overview

So what makes Julia special? A good summary is given at [www.infoworld.com][infoworld], which I'll briefly repeat here:

[infoworld]: https://www.infoworld.com/article/3241107/julia-vs-python-which-is-best-for-data-science.html

- Julia is just-in-time (JIT) compiled, not interpreted. Furthermore, it has type-specific compiled for each datatype. 
For example, the same function will be compiled differently if integers are passed to it instead of floats. 
This means compiled Julia code can be heavily optimised and therefore is fast.
A downside is that the first call where JIT compiling takes place is slow.
A more detailed explanation as well as tips to take advantage of this can be found [here][julia_tips].
- Julia uses multiple-dispatch to combine the benefits of dynamic typing and static typing. 
In short, multiple methods with different argument types can exist for a single function and Julia will choose the best one to use at compile time.
To facilitate  this, Julia has a comprehensive type system which can easily be extended.
This [video][multidispatch_video] by one of the founders explains this concept the best.
- Julia has a terse and straightforward syntax. The main language has 32 keywords (compared to Python's 35 and C++'s original 63, now 97). 
Of these, 16 have a direct parallel with a Python keyword.
- Julia comes with expansive Base and Core modules. A further 12 of the Python keywords are provided by functions and operators in these modules.
These modules are mostly written in Julia which makes them easily accessible and extendable.
- Julia has full Unicode support and mathematical-like notation. The following is a fully working piece of code which can be copied directly into the Julia REPL: 
<span style="white-space: nowrap;">`gaussian(x, μ, σ) = 1/(σ*√(2π))*exp(-(x-μ)^2/(2σ^2))`</span>. A disadvantage is that string indexing sometimes breaks because Unicode characters can take two or more bytes. 
So Base functions like `isascii`, `nextind()` and `eachind()` should be used to handle strings properly. 
- Julia supports metaprogramming, such as macros (like in C++) and creating expression objects with the keyword `quote`. So Julia programs can generate other Julia programs.
- Julia can call Python, C, and Fortran libraries. This is presumably to facilitate  crossover to Julia.

[julia_tips]: http://www.stochasticlifestyle.com/7-julia-gotchas-handle/
[multidispatch_video]: https://www.youtube.com/watch?v=kc9HwsxE1OY


The next three sections are [Things I like about Julia](#Things I like about Julia), [Neutral issues about Julia](#Neutral issues about Julia), 
and [Things I dislike about Julia](#Things I dislike about Julia).
Lasly there are small [Annoyances](#Annoyances) I would like to vent on after using it for 2 weeks.
I guess no programming language is perfect.
 
 
## Things I like about Julia<a name="Things I like about Julia"></a>

Like all programming languages, there was a learning curve to Julia. This was made more difficult by the lack of resources. 
However overall it was easy to learn coming from a Python background. There were somethings I definitely  prefer about Julia.

#### The type system and multiple-dispatch
This really does balance controlling types with giving the user flexibility.
For my random forest [code][Julia_repo], it essentially provided a way to make object specific functions, even though Julia is a functional language.
For example, the `DecisionTreeClassifier` and `RandomForestClassifier` have different `fit!` methods associated with each of them.
I had no problem calling `DecisionTreeClassifier` `fit!()` method from within the the `RandomForestClassifier` `fit!()` method.

A mistake I made at first is to use `AbstractFloat` inside the struct, whereas the recommendation is to always have concrete types in definitions.
This definitely slowed down my code, which is why I added the type to the struct definition: `DecisionTreeClassifier{T}`.
I also add an outer constructor to set this to `Float64` as a default: `DecisionTreeClassifier{Float64}`.

Multiple dispatch can easily be abused because this problem grows exponentially with the number of different arguments.
However it is not meant to be used for many different arguments, and most functions have a very low number of methods associated with them.[^exceptions]
Another fault is that sometimes it is not fully unambiguous which method should be called, especially with `Union` data types. 
But this is being worked on and clearer rules should be published in the future.

A prime advantage of the type system is on display with my `score` function. 
For my Python code, I wrote a separate score function inside the `DecisionTreeClassifier` and `RandomForestClassifier` classes.
They are however essentially identical functions.
For my Julia code, I wrote a single function which takes in a type of `AbstractClassifier`. 
Since I defined both my classifiers to be subtypes of `AbstractClassifier`, calling score on those objects dispatches to the this `score` function.

Of course you can argue I could have made a super class in Python which implements `score`, and then have both `DecisionTreeClassifier` and `RandomForestClassifier` inherit from it.
However this adds complexity without much benefit.
In Julia, there is a robust type system and it makes sense to follow those design patterns.

#### The keywords and syntax
> The Julia keywords: `abstract type, baremodule, begin, break, catch, const, continue, do, else, elseif, end, export, false, finally, for, function, global, if, import, let, local, macro, module, mutable struct, primitive type, quote, return, struct, true, try, using, while`

The keywords and syntax of Julia is very nice. I think these words are well chosen and make the language very flexible.
I specifically like the module and namespace control keywords: `module`, `export`, `import` and `using`. 
The keyword `export` allows you to define which variable names and functions in a module you want to expose to another namespace with `using`. 
Furthermore, with `using`, the functions will not be extensible - the new namespace will not be able to edit them or accidently overwrite them.
This can be overridden with `import`, but only if it is directly imported. 

On this topic, a minor annoyance is that during debugging it is nicer to `import` so you can update the functions without restarting the code.
However there is no import all option like with Python's `import *`. So you have to import everything, which is annoying.
I'm not sure if there is a better way around this.

Julia requires you to end all functions and control loops with the `end` keyword. 
This is the standard for most languages, and I prefer this to Python where the 'end' of a function is inferred by indents.
In fact I would prefer if `end` was used in more cases. For example, `end` is not required for `export`. This enables you 
to write code like the following:
{%highlight julia %}
module TestModule

y = 9

export x    # export x
       y    # load y, do nothing, move on
x = 5

end
{% endhighlight %}

This is a contrived example, but shows a real error I had. The correct code should be `export x, y` where the ',' tells the compiler that there is more to export.
But I forgot this comma (it was a long list of exports) and therefore everything after it was ignored and I got "undefined" errors.

#### Fast loops

Unlike Python where vectorisation is recommended as much as possible, but this is not the case in Julia.
I prefer for loops because they are easier to understand.
Indeed I found this to be the case with my random forest classifier. 
In Python, I was able to get significant speed increases by predicting samples in batches. 
All samples which have similar values are evaluated in the same recursive call.
In Julia, I found that these were roughly equivalent, with the batch prediction method being about 10% slower.
This makes more since to me: both functions have to test all samples, but the batch method has to create new arrays of varying lengths along the way.
Obviously though this logic doesn't hold in Python.

If you do want to do broadcasting, there is a nice syntax using '.' which broadcasts most functions e.g. `f.(x)` will apply `f` individually to each value in the array `x`.  

## Neutral issues about Julia <a name="Neutral issues about Julia"></a>

Before I go on to the negatives, there are a whole bunch of issues I don't feel strongly about either way.
- Julia use one-based indexing. So does Matlab. Meanwhile C, C++ and Python use zero-based indexing. It's a choice. Either way I get off-by-one errors. 
But really, if you really want me to go with a preference, I prefer one-based indexing. 
- Functional vs object-orientated programming (OOP). Whole books have been written on the merits and cons of both.
Julia is most certainly a functional orientated language. 
But with  multiple dispatch, it provides some features of OOP, so for the most part it bridges this divide (more on where it fails in the next section).
- My personal opinion is Julia is more difficult than Python. Maybe you've got that sense too with my talk of type management. 
Julia gives the user more control than Python, and I think a necessary trade-off is that is more complexity.
But C++ is harder still, and C is even harder.
Python just has set the bar very low here.
Overall, I think Julia is easy to learn, and is suitable as a beginner language.

## Things I dislike about Julia <a name="Things I dislike about Julia"></a>

Finally, the dislikes. I've certainly passed my enchantment phase with Julia. There are some things I really do not like about Julia.

- Slow debugger:
The debugger was seriously slow for me, on both VS Code and Juno (Atom).
This was by a factor of several hundreds. So instead of waiting seconds, I waited minutes for my program. 
I am not sure if this is a specific fault with my installation or a general fault.
Eventually I relied on manual code edits, printing and returning extra variables to do my debugging. 
I really hope this is improved in the future.
- Package precompile times: be warned, the first time you use a package it takes long to load! This is because Julia precompiles code the first time it is used.
However thankfully you only have to put up with this once per Julia installation/environment.
- Compiler latency, or the infamous First Time to Plot problem. 
This is one of the most common complaints, and indeed was at the top of the list in a recent [video][julia_wrong] by a founder of "What's wrong with Julia". 
Every time you use Julia, the first time you use any function it will be slow because of the JIT compiling.
This is why I excluded the first run for my random forest benchmark at the start of this post. For Julia, it took 4.6s, or about 6 times longer than the mean time for 10 subsequent runs.
- There is no easy way to access all the methods of a "class". This is a necessary trade-off with using a functional language instead of an OOP language.
For example, in my Python code you can clearly query a random forest object to see that it contains the `predict`, `fit` and `score` methods.
But in the Julia code, these are all external to the random forest struct, and worse, `score` is called on an `AbstractClassifier`, not a `RandomForestClassifier`.
How does one pull these three functions out of all the other external methods (printing, appending to vectors etc) that can act on the random forest struct?
An obvious fix, without reading through the code of every single package you use, is to have good documentation.
Unfortunately this is lacking for many packages, for example the Plots package.
- Extending the previous complaint, there is no clear way to propagate types. That is, Julia has no concept of inheritance other than defining subtypes and supertypes.
Once again, this feels like a necessary trade-off of using a functional approach over OOP. 
In my random forest code I defined dummy `predict` and `fit!` methods for my `AbstractClassifier` struct. 
But there is nothing forcing a subtype of `AbstractClassifier`  to implement concrete versions of these methods (e.g. like the `virtual` keyword would in C++).

[julia_wrong]: https://youtu.be/TPuJsgyu87U?t=155

## Annoyances<a name="Annoyances"></a>

These are minor issues I don't like. They may be changed in future updates or are not important issues.
- Scoping issues: scoping is dealt better than Python but still there are some inconsistences.
For example, in Julia you have to write the `global` keyword if you want to modify an external variable to a function or control loop (I understand this is a recent change).
However, you don't have to write it if you just want to read the value.
Here is an example:
{%highlight julia %}
x = 5
y = 10
function f()
	global x += 1
	z = x * y
	return z
end
{% endhighlight %}
This function will pull in the global variable `y` even though no `global` keyword was used for y. Hence running `f()` will return 60.
I had a few errors where I accidently used the name of a global variable and hence my values were wrong.
I would prefer this to require a call to load the variable `y` in the function, e.g. `global y`.
- Constructors for structs look horrible. Have a look at my random forest [code][Julia_repo]. They require writing out default variables in order as defined in the struct list.
- There is no concept of header guards like there is in C++. This means you can include multiple files or multiple modules. 
As far as I know, each duplicate block of code will be loaded and write over the previous code. 
This should only slow down compiling and not run times.
- Returns are optional in function definitions. If you leave out the `return` keyword, a function will return the last calculated value.
To return nothing, you have to explicitly write `return` or `return nothing`. I would prefer if the latter was the default. 
- I think the name of this language was very badly chosen. "Julia" is in the top 100 female names in the world ([reference][julia_name]).
There are multiple variants of it in different languages (Giulia, Džūlija, Yulia etc). See Wikipedia's entry on [Julia][julia_wiki] for a full list.
It is the name of famous [actresses](https://en.wikipedia.org/wiki/Julia_Roberts) and [singers](https://en.wikipedia.org/wiki/Julia_Michaels).
There is precedent: in 1901 Emil Jellinek named a car after after his daughter Mercedez, 
dooming every Spanish woman with that name since to explain that they are not named after a car; the car is named after them ([reference](https://en.wikipedia.org/wiki/Mercedes-Benz)).
But really, in this day and age I think the founders should have known better.
Worse, the Julia language is named after no-one ([reference](https://docs.julialang.org/en/v1/manual/faq/)).
Could they not have chosen a less inconvenient name?

[julia_name]: https://www.babycenter.com/baby-names-julia-2368.htm
[julia_wiki]: https://en.wikipedia.org/wiki/Julia

# Conclusions

So I may have a few complaints about Julia (the programming language) but overall I really like it.
I mean it when I say I will use it for all personal programming where I would have used Python before.
The extra control, flexibility and robustness is a big reason to switch to Julia.
If none of this convinces you, I think achieving a 9 times increase in speed for code of similar length and effort should.

This is an awesome language and I hope the community grows.
I also hope industry begins to adopt it like academia already has.

[julialang_tips]: https://docs.julialang.org/en/v1/manual/performance-tips/
[julialang_diffs]: https://docs.julialang.org/en/v1/manual/noteworthy-differences/
[diffs]: https://cheatsheets.quantecon.org/




---

[^SIR]: SIR stands for Susceptible, Infected and Recovered/Removed. I have not published my code because it is a solution for an assignment in the online course. 
[^pie]: A typical example. This is the following documentation for the pie chart function `pie` in the Plots package: "Plot a pie diagram". What? What are the keyword arguments? How do I rotate the chart? How do I add annotations? How do I set colours?
[^out_of_date]: For example, the `CSV.read(file_name)` syntax in the 6 months old Computational Modelling course doesn't work anymore. The new syntax requires you to specify the data sink type - it's a good change. So the new syntax is: `CSV.read(file_name, DataFrame)`.
[^exceptions]: Exceptions to this are the operator functions such as + and \*, which have 184 and 364 methods respectively. If one loads the DifferentialEquations package, this grows to 420 and 833 respectively. 
