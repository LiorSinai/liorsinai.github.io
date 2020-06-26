---
layout: post
title:  "Scramble puzzle solver"
date:   2020-06-26
author: Lior Sinai
categories: coding
tags:	coding backtracking scramble puzzle
---

_This post details a backtracking algorithm for solving scramble square puzzles. This is a deceptively simple looking puzzle, with only 9 pieces, but it
 can be very challenging to solve, with over 23.8 billion possible arrangements._


# Scramble Squares

<figure class="post-figure">
<img 
    src="/assets/posts/2020-06-26-scramble-puzzle/ScrambleSquare_serengeti.jpg"
	style="width:50%"
	alt="Serengeti scramble square"
	>
  <figcaption>Serengeti scramble square puzzle by <a href="https://www.scramblesquares.com/shop/geography/serengeti-scramble-squares/">Scramble Squares</a> </figcaption>
</figure>

When I was a child, I used to spend many hours trying to solve a scramble square puzzle. An example is shown above.
My puzzle was similar to this, except instead of African animals, it had monkeys wearing colourful t-shirts. 
The puzzle consists of nine square pieces and four images spread across the squares. 
On each side on each square is half of one of the four images. For example the top half of an elephant.
Some of the other pieces have the corresponding half, like the bottom half of the elephant.
The aim is arrange the pieces in a 3x3 grid such that all inner images are aligned - all top and bottom halves are matched.
This is actually very difficult, and I was not able solve it.


Much later, when I was in university, I returned to this puzzle. I wondered if there was a procedural method to solve it. Such methods exist for Rubik's cubes.
I was not able to find any, but I did find a depth first search backtracking algorithm for it described in this [paper][scramble_algorithm].
This algorithm is rather painful to do by hand, but easy to implement with a computer.
I wrote up the algorithm in Python, and within a second, I found the two solutions for my puzzle.
So much for hours of frustation when I was younger!

The rest of this post will describe this algorithm in my detail.

## The algorithm

The search space for this puzzle is surprisingly large. 
There are $9!$ permutations for placing each puzzle piece, and each piece has four orientations, except for the middle piece
(because rotating this piece just rotates the entire puzzle).
This gives $4^8 9! \approx 23.8$ billion arrangements of the puzzle. 
Therefore, just trying every puzzle arrangement is too slow. 

<figure class="post-figure">
<img 
    src="/assets/posts/2020-06-26-scramble-puzzle/order.png"
	style="width:30%"
	alt="backtracking order"
	>
</figure>

The backtracking algorithm very efficiently cuts down this search space. Each piece is placed one at a time in the order shown above.
Then either the piece fits, and the algorithm moves onto the next position, or 
it does not and the algorithm tries a new piece.
If the algorithm runs out of pieces, it backtracks to the last position where a piece was placed, and tries a new piece.
If it reaches position $k=8$ and the last piece fits successfully, a solution is registered. Then it backtracks again. 

Note that if a piece does not fit in the *k*th position, then all arrangements with the piece in that position are skipped.
If for example, a piece does not fit in position $k=1$, then all remaining $4(4^7 7!)$ arrangements with the piece in this position - 1.4% of the total search space - are skipped.

## The Python code

<figure class="post-figure">
<img 
    src="/assets/posts/2020-06-26-scramble-puzzle/Example.png"
	style="width:50%"
	alt="scramble mockup"
	>
</figure>

The Serengeti puzzle I posted above has no solution[^1]. So instead I'll use a mock-up of the Monkey puzzle I had as a child. This is shown above. 
I've labelled each piece from 0 to 8. 
Each card is encoded as an array using the following rules:
- Colours are mapped to a number: blue &rarr; 1, green &rarr;  2, red &rarr; 3  and purple &rarr; 4. 
- Triangles are positive and blocks are negative.
- Sides are labelled starting from the top and going clockwise.

For example, piece 0 is encoded as `[-2, -3, +1, +4]`. 
The rest of the cards look like:
{% highlight python %}
blue, green, red, purple = 1, 2, 3, 4
cards=[
	[-green,-red,+blue,+purple],
	[-purple,+blue,+purple,-green],
	[-blue,-green,+blue,+red],
	[-purple,+green,+red,-blue],
	[+red,-purple,-green,+purple],
	[-blue,-red,+green,+purple],
	[-red,+green,+red,-blue],
	[-green,-blue,+purple,+red],
	[-purple,-green,+red,+purple]
]
{% endhighlight %}

The state of the puzzle can be summarised in two variables: `order`, a list of the order of placements of pieces, and `rot`, the current rotation/orientation of each card.
A rotation is encoded as a number from 0 to 3.
I created a simple class to store this state, and also coded a nice representation for the `print` function.  

{% highlight python %}
class ScrambleSquare():
    def __init__(self, cards):
        self.cards = cards
        self.order = [-1, -1, -1, -1, -1, -1, -1, -1, -1]
        self.rot = [0, 0, 0, 0, 0, 0, 0, 0, 0] 

    def __repr__(self):
        repr = ''
        order = self.order
        repr += ' '.join(map(str, [order[6], order[7], order[8]])) + '\n'
        repr += ' '.join(map(str, [order[5], order[0], order[1]])) + '\n'
        repr += ' '.join(map(str, [order[4], order[3], order[2]]))
        return repr
{% endhighlight %}

Two cards "fit" if the sum of the touching sides is 0. So a simple `fit_2cards()` function can be written:
{% highlight python %}
def fit_2cards(card1, rot1, side1, card2, rot2, side2):
    if card1[side1 - rot1] + card2[side2 - rot2] == 0:
        fits = True
    else: 
        fits = False
    return fits
{% endhighlight %}

Cards are "rotated" 90&deg; counter-clockwise by subtracting a value from the index. The indexing behaviour of Python conveniently wraps around with negative numbers,
so something like `card1[0 - 1]` is evaluated as `card1[-1]=card1[3]`. 
Sometimes cards need to fit with two or three cards simultaneously, so we'll still need another function `fit_position()`. 
For now, I'll kept it abstract and present the framework for the whole algorithm:
{% highlight python %}
def solveScramble(cards):
    def solve(k, puzzle, stack):
        calls[k] += 1
        if k == size:
            print('Solution found!!')
            print(puzzle)
            return
        for idx in range(len(stack)):            
            new = stack[idx] #select a new card that hasn't been used
            for r in range(num_orientations): #try different orientations
                if puzzle.fit_position(k, new, r): # backtracking checkpoint
                    puzzle_next = deepcopy(puzzle)
                    puzzle_next.order[k] = new
                    puzzle_next.rot[k] = r
                    stack_next = stack[:idx] + stack[idx + 1:]
                    solve(k + 1, puzzle_next, stack_next)
                if k == 0: 
                    break #don't rotate the first piece
	
    num_orientations = 4
    size = 9  
    stack = [0, 1, 2, 3, 4, 5, 6, 7, 8]       
    calls = [0] * (size + 1)  

    puzzle = ScrambleSquare(cards)

    solve(0, puzzle, stack)
{% endhighlight %}

It is always good practice to write the terminating condition for the recursive function at the top of the function.
The `calls` variable is useful for analysing the results.

For the `fit_position()` function, the sides which need to be compared are different at each position. Also, at positions 3, 5, 7 and 8, two sides on the card need to be checked.
I found it was easiest to hardcode all this:
{% highlight python %}
def fit_position(self, k, used_k, rot_k):
        if k == 0:
            fits = True
        else: #Each card must fit with the previous card:
            card_k = self.cards[used_k]
            side_k = [1, 3, 0, 1, 1, 2, 2, 3, 3][k]         
            card_j = self.cards[self.order[k - 1]]
            rot_j = self.rot[k-1]        
            side_j = [0, 1, 2, 3][side_k - 2] #picks the opposite side
            fits = self.fit_2cards(card_k, rot_k, side_k, card_j, rot_j, side_j)
        
        #Extra fitting criteria for particular positions:
        if k in [3, 5, 7, 8]:
            cards, order, rot = self.cards, self.order, self.rot
            if k == 3:
                side_k = 0
                card_other, rot_other, side_other = cards[order[0]], rot[0], 2
            elif k == 5:
                side_k = 1
                card_other, rot_other, side_other = cards[order[0]], rot[0], 3
            elif k == 7:
                side_k = 2
                card_other, rot_other, side_other = cards[order[0]], rot[0], 0
            elif k == 8:
                side_k = 2
                card_other, rot_other, side_other = cards[order[1]], rot[1], 0
            fits = fits and self.fit_2cards(card_k, rot_k, side_k, card_other, rot_other, side_other)
        return fits 
{% endhighlight %}

## Results

This code retrieves the following solutions in 0.17 seconds:
<div id="unique-name" class="row">
<img class="mx-auto"
    src="/assets/posts/2020-06-26-scramble-puzzle/Solution1.png"
	style="width:40%"
	alt="solution 1"
	>
<img class="mx-auto"
    src="/assets/posts/2020-06-26-scramble-puzzle/Solution2.png"
	style="width:40%"
	alt="solution 2"
	>
</div>
These are the only two solutions for this puzzle.
In total, it calls `solve` 588 times. The final `calls` list looks like: 
<span style="white-space: nowrap"> `[1, 9, 43, 165, 70, 151, 68, 60, 19, 2]`. </span>
So, 19 times the code placed 8 pieces, but only 2 of those times was it able to progress to a solution.

## Conclusion

This is tough puzzle to give to someone. Certainly, as an engineer and as a programmer, I would approach this problem with an algorithmic way of mind.
However, when I was younger, I was not thinking along those lines at all. I wonder though, if with guidance, it is worthwhile showing this to a young student.
It would teach them the value of algorithms. 
It is certainly possible to implement this algorithm by hand. I was able to do this recently although it took me 1.5 hours. 
Part of the reason it took so long was because I was trying to be clever and skip steps.
But once I followed the simple rules with the physical stack in my hand, I got into a rhythm and it went quickly. 
In any case, it is definitely a worthwhile exercise for teaching older students about backtracking and recursion.

-----

[scramble_algorithm]: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.953.6583&rep=rep1&type=pdf

[^1]: At first, I thought this was a mistake. However after failing to find solutions for several other puzzles on that <a href="https://www.scramblesquares.com">website</a>, I think this was delibrate. This is a crude form of copyright protection -  it prevents you from copying the puzzles (from this website at least).
