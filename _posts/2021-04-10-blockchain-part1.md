---
layout: post
title:  "A private blockchain for files: part 1"
date:   2021-04-10
author: Lior Sinai
categories: coding
background: '/assets/posts/blockchain/Blockchain.jpeg'
tags:	'blockchain'
---

_A program for storing files in your own personal blockchain (because you can)._ 


This series is split into two parts. 
Part 1 describes the high levels details of the program I made and what it can do.
[Part 2][part2] goes into the technical details of the C# program.

[repo]: https://github.com/LiorSinai/BlockchainFileSystemCs
[verge-nft]: https://www.theverge.com/22310188/nft-explainer-what-is-blockchain-crypto-art-faq
[thecorrespondent]: https://thecorrespondent.com/655/blockchain-the-amazing-solution-for-almost-nothing/86649455475-f933fe63
[cryptoart-enviromental-issues]: https://everestpipkin.medium.com/but-the-environmental-issues-with-cryptoart-1128ef72e6a3
[bitcoin-scam]: https://techcentral.co.za/south-africas-mti-was-worlds-biggest-crypto-scam-in-2020/105021/
[bitcoin-time]: https://zipmex.com/au/learn/how-long-does-it-take-to-transfer-bitcoin/
[bitcoin-fees]: https://www.coindesk.com/saving-bitcoin-high-transaction-fees
[bitcoin-energy]: https://www.statista.com/statistics/881541/bitcoin-energy-consumption-transaction-comparison-visa/
[mining-pools]: https://www.blockchain.com/pools 
[bitcoin-exchanges]: https://www.bitcoin.com/bitcoin-exchange-directory/

[wiki-cheque]: https://en.wikipedia.org/wiki/Cheque

# Introduction<a id="Introduction"></a>

Several years ago during my masters my friend and I made a Bitcoin simulater for our final project in a software course.
It was a watered down version of the real Bitcoin, but it worked well enough.
After I started this blog I'd thought about doing a blog post on it, but I had become very disillusioned with blockchain.
As this [article][thecorrespondent] so wonderfully put it, blockchain is the amazing solution for almost nothing.
Bitcoin creates more problems than the banks it was meant to replace.
It is very [slow][bitcoin-time] for transferring money, has very high transaction [fees][bitcoin-fees] (more than $20 at the time of writing),
it's easy to [scam][bitcoin-scam] people with no consequences and it uses 550,000 times more [energy][bitcoin-energy] than an equivalent VISA transaction. 
It's not even that decentralised because large [mining pools][mining-pools] and [exchanges][bitcoin-exchanges] control most of it anyway. 

I was also unimpressed with the actual software. 
Despite all the metaphors that crypto-currencies are like "digital gold", the underlying technology feels more like a convoluted cheque management system.[^cheques]
For the record, I have not touched a cheque in 15 years.
In short, it's a pain to manage transactions on a blockchain. 

However, the new onset of [Non-fungible Token][verge-nft] (NFT) mania reignited my interest in blockchain technology.
I still thought it was a bad idea. But at least an NFT blockchain would be an _easy_ bad idea.
So I made a private blockchain for storing NFTs in C#. You can find it at my [GitHub repository][repo].
It requires almost no programming experience to use, other than basic knowledge of a command line interface (e.g. the "Command Prompt" in Windows or the "Terminal" on a Mac).

My blockchain program currently implements these features (I've crossed out features that are part of most cryptos but are not in my blockchain.):
1. An immutable blockchain: any change to any file registered inside a block will make that block and all subsequent blocks after it invalid.
2. <s>Decentralized network management.</s> 
	1. <s>Consenesus mechanism. </s>
	2. <s>Propagation mechanism. </s>
	2. <s>Central Waiting list (mempool). </s>
3. Proof of work.
	1. Manual setting of difficulty from level 0 to 256. Default is 0.
	2. <s> Automatic difficulty adjustment mechanism. </s>
4. Storage: the blockchain can be saved in JSON format and reloaded from JSON format. Files are stored in a standard OS directory.
5. <s>Cryptographic security. </s>

I may or may not add to this feature list.

# Ok, so what did I actually make?

Firstly, I've made a Command Line Interface for the program. It looks like this:

<figure class="post-figure" id="Screenshot CLI">
<img class="img-80" 
    src="/assets/posts/blockchain/Screenshot_cli.png"
	alt="Screenshot_cli"
	>
	<figcaption></figcaption>
</figure>

You can enter commands such `--load MyBlockchain/MyBlockchain.json` or `--stage-token` or `--print-blockchain`.

This program makes and edits a managed folder. 
Here is what such a folder looks like:

<figure class="post-figure" id="Screenshot MyBlockchain">
<img class="img-80" 
    src="/assets/posts/blockchain/Screenshot_MyBlockchain.png"
	alt="Screenshot_MyBlockchain"
	>
	<figcaption></figcaption>
</figure>

This is what the inside of a "Block" looks like:
<figure class="post-figure" id="Screenshot MyBlockchain Block 0">
<img class="img-80" 
    src="/assets/posts/blockchain/Screenshot_MyBlockchain_Block0.png"
	alt="Screenshot_MyBlockchain_Block0"
	>
	<figcaption></figcaption>
</figure>

So nothing too special so far. You can easily make these folders and name them yourself.

What is more important is the "MyBlockchain.json" file in the main folder. This holds the blockchain metadata.
Here's an example of its contents:
{%highlight json %}
{
  "Name": "MyBlockchain",
  "TimeStamp": 1617743586,
  "Blocks": [
    {
      "Index": 0,
      "Target": 0,
      "Nonce": 0,
      "PreviousHash": "0000000000000000000000000000000000000000000000000000000000000000",
      "MerkleRoot": beb4d23bb916a16e32cd3b91229057213bf15357e3e16789b5edc3b6abc4f8bb,
      "TimeStamp": 1617743586,
      "Tokens": {
        "f841bb6dcfcf6e4322334c323218e765fd3598b0cf4173b42588199bc0a428bd": {
          "TimeStamp": 1617743587,
          "Author": "Brian Kernighan",
          "UserName": "@admin",
          "FileName": "Hello.txt",
          "FileHash": "315f5bdb76d078c43b8ac0064e4a0164612b1fce77c869345bfc94c75894edd3"
        },
        "85bef1763810a371e930891dfd8956b4a2832602fe5a049af4df0fd38062df88": {
          "TimeStamp": 1617743587,
          "Author": "Lior Sinai",
          "UserName": "@admin",
          "FileName": "FumiakiKawahata_Yoda_LiorSinai.jpg",
          "FileHash": "bad4a80e7e3a60bbb5932dd508ec6b568c79bf58781943e58538272a36a7d6c7"
        },
        "10684a859d87a00a2f1a47c20344b33c976c039784a3ab7006c45a4f9d8427cc": {
          "TimeStamp": 1617743587,
          "Author": "Lior Sinai",
          "UserName": "@admin",
          "FileName": "fox.png",
          "FileHash": "b2b18c67011f4f246eb54ad88d885c6abb9ac09858fdee7c7b13065aeabfa2eb"
        },
        "44fc148f5ca4fd869a300c74f2be9ae46ada2eacc0f43795bd102ff717085241": {
          "TimeStamp": 1617743587,
          "Author": "Marcus Tullius Cicero",
          "UserName": "@admin",
          "FileName": "LoremIpsum.txt",
          "FileHash": "2d8c2f6d978ca21712b5f6de36c9d31fa8e96a4fa5d8ff8b0188dfb9e7c171bb"
        }
      }
    },
    {
      "Index": 1,
      "Target": 0,
      "Nonce": 0,
      "PreviousHash": "73c15e57339d5c4ea732f33136165a49ef86590b12b9551bd2cd0343f6367ab2",
      "MerkleRoot": "58c94931603a0ba52fe13e4c18f0e6189349b4761ac017d667f0a286ecd1f04d",
      "TimeStamp": 1618056323,
      "Tokens": {
        "58c94931603a0ba52fe13e4c18f0e6189349b4761ac017d667f0a286ecd1f04d": {
          "TimeStamp": 1618056356,
          "Author": "unknown",
          "UserName": "@admin",
          "FileName": "Dragonsunset.jpg",
          "FileHash": "d3f768fc0e147ed17f285cae433d8260c949c2fafa6f0831f3d9e2c86d2410b0"
        }
      }
    }
  ]
}
{% endhighlight %}

Here is a simplified outline of this file:
```
MyBlockchain
|___Block0
|	|___token f8 - Hello.txt
|	|___token 85 - FumiakiKawahata_Yoda_LiorSinai.jpg
| 	|___token 10 - fox.png
|	|___token 44 - LoremIpsum.txt
|___Block1
	|___toke 58 - Dragonsunset.jpg

```
Most of the JSON file should be human readable. 
The only pieces which might not be are the timestamps, which are the number of seconds since 1 January 1970 (it's very easy for a [computer][epochconverter] to convert to an actual date)
and the plethora of 64 character length hexadecimal numbers. The latter are essential for checking if any of the data has been tampered with.
More on them later.

[epochconverter]: https://www.epochconverter.com/

I said one of the features of this blockchain is that it is immutable. However, it is clearly not. 
You can open all these files like normal files and edit them.
For example, I can change the text in "Hello.txt" from "Hello, world!" to "Hello!". But then if I try to verify the blockchain or add any blocks, I get this error:
<figure class="post-figure" id="Screenshot MyBlockchain invalid">
<img class="img-80" 
    src="/assets/posts/blockchain/Screenshot_invalid.png"
	alt="Screenshot_invalid"
	>
	<figcaption></figcaption>
</figure>

My blockchain is now in an invalid state and I cannot do anything with it. That is, until I restore the "Hello.txt" file to its original state or delete everything and start over again.

Furthermore, I cannot edit the metadata in the JSON file as this will also put it into an invalid state (except for the MerkleRoot because this is recalculated each time).

And that is a fully functional blockchain.

# What's with those hexadecimal numbers?

Cryptographic hashes are the secret sauce behind the blockchain. 
A hash function maps any sort of data to a number. 
Most blockchains use the SHA-256 hash, which maps data to a number between $0$ and $2^{256} - 1$:

<figure class="post-figure" id="SHA256_simple">
<img class="img-80" 
    src="/assets/posts/blockchain/sha256_simple.png"
	alt="Sha256 simple"
	>
	<figcaption></figcaption>
</figure>
By convention that number is represented as a hexadecimal number with 64 characters.

Let's dive a bit deeper.
Data in a computer is always stored as 1s and 0s. 
What the SHA-256 hash does is take these 1s and 0s in and performs the mathematical equivalent of shaking them up and shuffling them around. 
It then spits out 256 1s and 0s, which can be written as a hexadecimal number with 64 characters, or a decimal number between $0$ and $2^{256} - 1$:

<figure class="post-figure" id="SHA256_detail">
<img class="img-80" 
    src="/assets/posts/blockchain/sha256_detail.png"
	alt="Sha256 detail"
	>
	<figcaption></figcaption>
</figure>

The SHA-256 algorithm  has been carefully designed to have the following properties:
1. It is deterministic. The same result always gives the same output (It's a computer, so we can recreate the shakes and shuffles exactly).
2. It is chaotic. A slight change to the input results in a big change to the output. (For an interactive demonstration, go to [https://emn178.github.io/online-tools/sha256.html](https://emn178.github.io/online-tools/sha256.html).)
3. It is computationally infeasible to reverse.

The last property is counter intuitive. 
The SHA-256 algorithm is opensource and anyone can follow the [instructions][sha256-pdf] to write the code (I have done such a thing).
But even so, it is very, very hard to reverse.
There are many steps to it and several steps flip (XOR) bits so that working backwards requires making many guesses to whether bits were 1 or 0.

If I give you the hash `1478186d3ebe3201be94ffeee6945603b601c22fe289ffa620e8a3ff2b44ede4`, 
it would take you a very long time to work out (if ever) that "Welcome to my blog" was the input. 

So now that we have this magic hash function, it's very easy to see why it is so useful for the blockchain.
We can hash files and save the result. If anything in the file changes, it will change the hash and we will know.
Furthermore we can hash whole blocks, and if anything in the block changes, we will know.

<figure class="post-figure" id="bitcoin_fig2">
<img class="img-80" 
    src="/assets/posts/blockchain/bitcoin_fig2.png"
	alt="blockchain diagram"
	>
	<figcaption>Source: <a href="https://bitcoin.org/bitcoin.pdf">Bitcoin whitepaper</a></figcaption>
</figure>


Each block also has the previous block's hash inside of it. Hence we have a chain. 
If we break this block's hash, the next block will have the wrong `previousHash`, and then its hash will be wrong so the block afterwards will have the wrong hash, and so on.

[sha256-pdf]: http://www.iwar.org.uk/comsec/resources/cipher/sha256-384-512.pdf

# Proof of work

The proof of work is used to make inserting blocks into the blockchain difficult.
This serves two main purposes:
1. It prevents people spamming the networks with millions of blocks committed at once.
2. It functions as a lottery so the person claiming the reward for mining changes each time.

The first is so that the network has time to verify blocks, and the second is to prevent a single entity from controlling all the mined coins
(but [mining pools][mining-pools] dominate it anyway). Together this prevents [double-spending][bitcoin-whitepaper].


[bitcoin-whitepaper]: https://bitcoin.org/bitcoin.pdf#page=2

 
But this is a private blockchain, so there are no rewards and you _can_ spam it if you want to.

That said, I have included a proof of work algorithm for completeness. 
I have implemented the _exact_ same algorithm as Bitcoin.
It is: the block hash must be less than a certain number.
Equivalently, the block hash must start with $n$ zeros in binary form, or $n/4$ zeros in hexadecimal form.
To get it to this, we can set a free value called `Nonce`, add dummy files to the block, or make it at a different time.
Essentially, we shoot in the dark and hope for the best.[^PoW]

The target $n$ for blockchain ranges as an integer from 0 to 256 (Bitcoin basically allows decimal points as well).
The default is 0 and it is strongly recommended you keep it at 0.

To understand why, lets compare it to the Bitcoin [difficulty](https://www.blockchain.com/charts/difficulty). 
This is usually given as a number $D$ which can be calculated from $n$ as follows:

$$D = 2^{n-48}(2^{16}-1)$$

Increasing the target by 1 halves the amount of valid numbers, so it doubles the difficulty and average time taken.
A standard PC has 2GHz $\approx 2^{30} $ calc/sec of processing power.
For my CPU, this means it can do about 1.4 million hashes/sec $\approx 2^{20.4} $ hashes/sec (since it requires approximately $2^{10}$ calculations for a single hash).[^miners]
This means that on average:
* a target of 20 takes 1 second.
* a target of 21 takes 2 seconds.
* a target of 22 takes 4 seconds.
*    .... 
* a target of 30  will take 1024 seconds ~= 17 min.
* a target of 40 will take 12 days.
* a target of 45 will take 1 year.
* a target of 76 will take 2.3 billion years (that is billion with a B).

A target of 76 was the Bitcoin difficulty at the time of this commit (so all the Bitcoin block hashes have 19 zeros).
As has been well-documented, a Bitcoin block is mined every 10 minutes instead of billions of years because an army of 
industrial factories with GPUs which combined use more [electricity than Argentina][bitcoin-energy-comparison]
are all dedicated on solving this one abstract problem while creating an ecological disaster that we will never be able to justify to future generations.

[bitcoin-energy-comparison]: https://www.bbc.com/news/technology-56012952


# How useful is this program?

This program is essentially a very basic piece of version control software.
There is much better version control software out there like [Git](https://git-scm.com/), which ironically I'm using to version my blockchain software.
In addition to tracking if a file changed, Git can track what changed and can restore a file to any time point.
It also has many more features such as branching, security, online hosting, rebasing, commit messages ...

The blockchain really is the amazing solution for almost nothing.

# Technical details

Join me in [part 2][part2].

[part2]: {{ "coding/2021/04/11/blockchain-part2" | relative_url }}

---

[^cheques]: Here is a "digital cheque" metaphor for a cryptocurrency: Imagine you find an unsigned cheque for \\$100. 
	You don't cash it immediately. Later, a friend asks for \\$70. 
	You say, I have an unsigned cheque which we can share. But we can't tear it up.
	So I am going to write a new cheque for you for \\$70, which says you are owed \\$70 of my \\$100 cheque (when I cash it in) and I'll make a note for myself to say I can't spend those \\$70.
	That friend then has to pay for car repairs of \\$50. He says to the mechanic, I have a cheque for \\$70 of a \\$100 cheque.
	We can't tear it up, but I can make a new cheque which says you are owed \\$50 of that cheque (when I cash it in) and I'll make a note for myself to say I can't spend those \\$50.
	And this is essentially how cryptos works except they uses "blocks" instead of "cheques" and the original amount is "found" by a "miner".
	Note that unlike a normal, sane bank the blockchain doesn't store account balances. 
 
[^PoW]: If you know a better strategy, please let me know. Nevermind improving my program, we could make lots of money with Bitcoin.

[^miners]: Real miners use specialised [computers][bitcoin-miners] which do  $10^{14}\approx 2^{46.5}$ hashes/sec. A proof of work on one of those will take 24 years at the current Bitcoin difficulty.

[bitcoin-miners]: https://www.buybitcoinworldwide.com/mining/hardware/
	
	

