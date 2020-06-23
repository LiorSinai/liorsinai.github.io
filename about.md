---
layout: page
title: About Me
description:
background:  #'/img/bg-about.jpg'
permalink: /about/
---
![alt text]({{ site.baseurl }}/assets/profile_graduationMSc.JPG){:.profile}

Hi. Welcome to my blog. I'm an engineer from South Africa and I'm curious about the world. 
My hobbies include reading, long distance running, and complex origami (all the origami on this site is folded by me).
I also occasionally work on my own engineering projects. My favourite place to be is in the middle of the wild bushveld in Africa.

I recently completed a masters in systems and control engineering at the [University of Twente](https://www.utwente.nl/en)
in the Netherlands. I also hold a bachelors in mechanical engineering from the [University of the Witwatersrand](https://www.wits.ac.za/) in South Africa.
In between my bachelors and masters, I worked for the national telecommunications company of South Africa, Telkom.

This blog will focus on engineering topics that I find interesting. Most of it will be on applications of mathematics in coding and in physics. 
My interests are broader than this. After graduating, I took a break from engineering, and my reading list consisted mainly of psychology books[^1]. 
I am also interested in the natural world and climate and climate change.
However, I want to use this space to focus only on engineering, because this is where I have professional expertise.

Here is a list of topics I want to cover:  
+ **Scramble puzzle solver**: this post will describe an algorithm from this [paper][scramble_algorithm] that I implemented for solving scramble puzzles. 
As a child I used to spend many hours trying to solve a  piece scramble puzzle. I was never able to do this. Much later I found out there is no logical
way to solve these unlike, for example, with Rubik's cubes. Computers however can make short work of these puzzles. 
+ **Sudoku solver**: this is a recent project I took on. Sudoku solvers are a well reserached topic (see [1][sudoku_ali], [2][sudoku_norvig] and [3][sudoku_stuart]).  
+ **100 years to solve an integral**: the surprising history and (still relevant) importance of the integral of the secant. I will try make this 
story accessible to everyone, but I will also include all the mathematical details for those who can follow. Of course, others have told this story: links here. 
This will also address the controversy and moral issues surrounding the Mercator projection (which really should not be used in representative/artistic/non-GPS related maps!).  
+ **Electromagnetic waves and the internet**: working in telecommunications taught me much about the infrastructure behind the internet. I'd like to explain that here. 
It always amuses me how people know that wi-fi doesn't work several meters away from the router, but at the same time, are surprised to learn that most of the internet is not
transmitted via wi-fi all the way from satellites in space! (Short explanation: it is mainly transmitted through lots of land and undersea cables).
+ **Quaternions**: these strange 4-dimensional objects were never part of any of my engineering courses. However, they are incredibly useful for 3D rotations, and are used in most
3D renders (for games and graphics). I'm self-taught in them, but I had to rely on several sources. I'd like to explain them in a full detail here, along with 
comparisons to other 3D rotation systems.
+ **Counting routes for the multiple travelling salesman problem**: this is a side problem from my thesis (I really should not have spent so much time on it).
+ **Listing k integer partitions**: another ~~distraction~~ side problem from my thesis.

_Lior Sinai_, {{ "2020-06-22" | date: '%d %B %Y' }}

[^1]: I highly recommend these three: Emotional Intelligence by Daniel Goleman, Blueprint by Nicholas A. Christakis and Quiet by Susan Cain.

[scramble_algorithm]: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.953.6583&rep=rep1&type=pdf
[sudoku_norvig]: https://norvig.com/sudoku.html 
[sudoku_stuart]: https://www.sudokuwiki.org/sudoku.htm
[sudoku_ali]: https://dev.to/aspittel/how-i-finally-wrote-a-sudoku-solver-177g
