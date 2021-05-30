---
layout: post
title:  "Intuitive explanations for non-intuitive problems: the Monty Hall problem"
date:   2021-04-11
author: Lior Sinai
categories: coding
background: '/assets/posts/monty-hall/doors.jpg'
categories: mathematics
tags: mathematics probability
---

_My take on the famous Monty Hall problem._ 


**intuitive** [ɪnˈtjuːɪtɪv] <br>
_Using or based on what one feels to be true even without conscious reasoning; instinctive._ <br>
[Definition from Oxford Languages](https://languages.oup.com/google-dictionary-en/)

This is part of a short series on probability, a field rife with problems that are easy to understand but results that are decidedly not intuitive for people. That said, many of these problems have answers that are comprehensibly logical, and it is a wonder that they are non-intuitive at all.

# Introduction 

Last week I watched the movie [21][imdb_21]. It's loosely based on the true story of MIT students who learn to card count in Blackjack.[^blackjack]
The students quickly transform from nerds to high rolling gamblers at Las Vegas. 
Drama ensues, and the script follows a charming but predictable path.

[imdb_21]: https://www.imdb.com/title/tt0478087/
[article_savant]: https://priceonomics.com/the-time-everyone-corrected-the-worlds-smartest/
[wiki_MontyHall]: https://en.wikipedia.org/wiki/Monty_Hall_problem
[betterExplained_MontyHall]: https://betterexplained.com/articles/understanding-the-monty-hall-problem/

There's a  scene early on in the film where the professor quizzes the star of the movie, Ben, on the [Monty Hall problem][betterExplained_MontyHall].
It's a famous and deceptively simple brain teaser.
It has had its share of controversy, like that time in the 90s when everyone corrected the [world's smartest woman's take on the problem][article_savant]. 

It is as follows: You are on a game show and you're given the choice of 3 doors. Behind one door is a car, and behind two are goats. You are asked to pick a door at random, and you will win whatever is behind it. You pick a door. The game show host keeps this door closed, and he opens another door which he knows has a goat behind it. There are now only two closed doors in the game. The game show host then asks you if you want to switch doors.

The big question is, should you? Does it matter? Will it improve your chances of winning a car?

I made a simulation of this game in JavaScript, and I highly recommend playing it before moving on:[^cheating]

<link rel="stylesheet" href="/assets/posts/monty-hall/style.css">
<div class="simulator-container">
  <h2>Monty Hall Game</h2>
  <div class="grid-container">
      <span class="grid-row-1">switched:</span>
      <span class="grid-row-1" id="switched_count">0</span>
      <span class="grid-row-1">won:</span>
      <span class="grid-row-1" id="switched_won">0.0%</span>
      <span class="grid-row-2">stayed:</span>
      <span class="grid-row-2" id="stayed_count">0</span>
      <span class="grid-row-2">won:</span>
      <span class="grid-row-2" id="stayed_won">0.0%</span>
  </div>
  <div class="banner">
      <p id="message" class="banner-text">Pick a door!</p>
  </div>
  <div class="flex-container door-container">
      <ul id="doors" class="door-list">
      </ul>
  </div>
  <form>
      <span class="num-doors-banner">
            number of doors: <input id="num-doors" type="number" class="num-doors" min="3">
      </span>
    <button class="button" type="button" onclick="init()">change</button>
  </form>
</div>

<script src="/assets/posts/monty-hall/MontyHall.js"></script>
    

# The controversy

The answer is it is most certainly better to switch. Play enough rounds following both strategies (about 20 rounds each) and you should see that switching is definitely the better strategy. (Coders can inspect the JavaScript with their browser. There is nothing funny going I swear.)
Most people do not grasp this at first, including me.

Of course the hero in the movie sees it immediately. Here is his answer:
>"Well, when I was originally asked to choose a door, I had a 33.3% chance of choosing right.
But after he opens one of the doors and then re-offers me the choice, it's now 66.7% if I choose to switch. 
So, yeah, I'll take door number two, and thank you for that extra 33.3%."

This is technically correct, but hard to follow, especially while  watching a movie. 
Why did the probability change to 66.7%, instead of being shared equally 50% between the two remaining doors?
This answer offers no explanation, but it need not be so obtuse. 

On a high level, it is important to realise that the host knows more than you, and by opening the door, he leaks this information to you. If he were to shuffle the goat and the car behind the remaining doors, then no information would be leaked and the 50/50 assumption would be valid. 

You can get a good feel of this information leakage by increasing the number of doors. The host will open all but two doors in the second step. Go back and try the again game but with 100 doors and 99 goats.

With the number of doors set to 100, I hope you follow this line of thinking:
1. There are 100 doors, and my chance of guessing correctly is very low (1/100). I probably got a goat with my first guess (99/100).
2. The host opens 98 doors. He has most likely revealed the other 98 goats, leaving the car behind the final unopened door.
3. So if I switch I should almost certainly get the car (99/100).

Here the cognitive biases from the three door problem disappears. There's no "my first guess is as good as the second". The second guess is better. 

With our first guess there is a 1% chance the car is behind that door, and a 99% chance the car is behind one of the other 99 doors. Then 98 doors are opened and there is _still_ a 99% chance it is behind one of the other 99 doors, but obviously you're not going to pick one of the 98 open doors. So therefore all that 99% probability shifts to the one remaining closed door.

# The solution

The strange thing is, is this is the exact same logic you should be following with the three door problem. 
Here is what I call the pessimist strategy:
1. There are 2 goats and 1 car. This game is stacked against me. My first guess is more likely to be a goat than not. So I'm going to assume it's a goat.
2. The host opens a door and reveals a goat. Great, he's just shown me where the second goat is.
3. So if I switch I should probably get a car.

Or the more optimistic approach: After my first guess, there's a 2/3 chance the car is behind one of the others doors, and after one is opened, I can't choose it. So that 2/3 chance is all shifted to one door, which I am going to choose. This was essentially Ben's answer.

Now that I know this logic, I cannot unknow it. It's like an illusion that after being broken cannot be hidden again, or a riddle whose answer renders the riddle itself silly. 

# Conclusion

I think the most fascinating part of this problem is not the solution - the maths is simple.
It's rather that we humans tend to get it wrong the first time. 
It should be intuitive but it's not. 
There's been an endless debate on why. 
Here is my 2 cents on the matter: with small numbers, there is a cognitive bias towards positive outcomes. That is, we think we are right even when we should know that the odds are stacked against us. This bias shortcuts our logical reasoning process, because it once was more beneficial to make small decisions quickly.
So it's not a flaw of our brains; it's a natural response to an unnatural question.

Do you agree? Why do you think this is the case?
 
[complex_variable]: https://en.wikipedia.org/wiki/Complex_analysis
[wiki_counting]: https://en.wikipedia.org/wiki/Card_counting

---

[^blackjack]:
    The film's portrayal of [card counting][wiki_counting] is wildly unrealistic. It focus solely on keeping the count, and completely neglects the different strategies that must be followed based on the count and the cards in play. 
    It also presents this method as almost full proof, whereas in reality the edge gained by practitioners is 1% at most.
    Personally I think this low margin would have injected suspense into the film, not taken away from it.
    I suggest reading "A Man for All Markets" by the person who invented card counting, Edward Thorp, for a real explanation of card counting.

[^cheating]:
    If you know JavaScript you can easily cheat. But please keep to the spirit of the game.