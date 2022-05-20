---
layout: post
title:  "Building a transformer in Julia"
date:   2022-05-18
author: Lior Sinai
background: '/assets/posts/transformers/transformer.png'
categories: coding
tags: mathematics transformers 'machine learning' 'deep learning'
---

_Building a transformer in Julia._ 


### Table of Contents
- [Introduction](#introduction)
- [Architecture](#architecture)
	- [Design considerations](#design-considerations)
	- [Inputs and outputs](#inputs-and-outputs)
	- [Description](#description)
- [Julia implementation](#julia-implementation)
	- [Project setup](#project-setup)
	- [Tokenizers](#tokenizers)
	- [Word embeddings](#word-embeddings)
	- [Position encodings](#position-encodings)
	- [Working with higher order arrays](#working-with-higher-order-arrays)
	- [Multi-head attention](#multi-head-attention)
	- [Encoder blocks](#encoder-blocks)
	- [Classifier](#classifier)
- [Use case: Amazon reviews](#use-case-amazon-reviews)
- [Conclusion](#conclusion)

## Introduction

[AttentionIsAllYouNeed]: https://arxiv.org/abs/1706.03762
[NvideaGTC]: https://youtu.be/39ubNuxnrK8?t=772
[BERT]: https://arxiv.org/abs/1810.04805
[GPT]: https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf
[GPT2]: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
[GPT3]: https://arxiv.org/pdf/2005.14165.pdf
[GPT3_api]: https://openai.com/api/
[Gato]: https://www.deepmind.com/publications/a-generalist-agent
[Image16words]: https://arxiv.org/abs/2010.11929
[NvideaMegatron]: https://arxiv.org/abs/1909.08053
[NvideaLM]: https://github.com/NVIDIA/Megatron-LM

In December 2017 Google AI released their transformer architecture in the paper [Attention is all you need][AttentionIsAllYouNeed]. 
They had achieved state of the art results on an English to German translation task using a mostly linear model that could be easily scaled up and parallelized. 
Since then it has come to dominate the machine learning space.
Many of the state of the art natural language processing (NLP) models today are transformer models.
Most of them have an incredibly similar architecture to the original and differ only on training regimes, datasets and sizes. 

<figure class="post-figure">
<img class="img-80"
    src="/assets/posts/transformers/transformer_model_sizes_annotated.png"
	alt="architecture"
	>
<figcaption></figcaption>
</figure>

Transformers lend themselves to large models. The original transformer model was no light weight: it had 65 million parameters and could be scaled up to 213 million for marginal improvements.
But this is tiny in comparison to OpenAI's [GPT-3][GPT3] with 175 billion parameters.
It is licensed over an [API][GPT3_api] and is responsible for generating text that has made its way around the internet and popular science articles. 
These large models are even shaping hardware.
The CEO of chip manufacturer NVIDEA, Jensen Huang, focused a segment of his [2022 keynote][NvideaGTC] speech on transformers and the impact on his industry.
NVIDEA have also released their own large transformer model with [530 billon parameters][NvideaMegatron] and have conducted tests with a [1 trillion parameter model][NvideaLM].

Despite being originally developed for NLP, they have come for computer vision too. 
This [2020 paper][Image16words] showed that they can compete with top convolutional neural networks (CNNs). 
More recently, DeepMind released an impressive model [Gato][Gato] that can perform multiple tasks such as "play Atari, caption images, chat, stack blocks with a real robot arm and much more". It has 1.8 billion parameters.

### Goals

[Bloem]: http://peterbloem.nl/blog/transformers
[IllustratedTransformer]: https://jalammar.github.io/illustrated-transformer/
[AnnotatedTransformer]: https://nlp.seas.harvard.edu/2018/04/03/attention.html
[YouTubeTransformer]: https://www.youtube.com/watch?v=XSSTuhyAmnI
[Transformersjl]: https://github.com/chengchingwen/Transformers.jl
[TransformersLite]: https://github.com/LiorSinai/TransformersLite.jl

All this development in transformers has been over the past 4 years.
In that time they have mostly replaced the old favourite for NLP in academia and industry, recurrent neural networks (RNNs).
However that has not been long enough for pedagogy to adapt.
As of today, machine learning courses still teach RNNs for NLP.
This has created a gap and many blogs have sprung up to full it explaining transformers to the every day student.
This blog post aims to be one of those.

I have two goals here:
1. Build a small working transformer in Julia code and train it on one use case.
2. Detail the mathematics of the transformer.

Julia has a small but growing userbase. It is an elegant and fast language and I highly recommend it.
But even if you don't know it well, I hope you will find this post easily readible and it will help improve your Julia. 
Mathematics on the other hand is a universal language and this should be accessible to anyone with university level maths. 

[AmazonReviews]: https://huggingface.co/datasets/amazon_reviews_multi

The use case is a dataset of [Amazon reviews from HuggingFace][AmazonReviews][^amazon_multi]. Only the English subset of the dataset was used with 200,000 training samples and 5,000 test samples. The models were trained on two tasks:
1. Predict the star rating given the review text. 
2. Predict a positive or negative sentiment with 1-2 stars labelled negative, 4-5 stars labelled positive and 3 stars removed.

[TFIDF]: https://github.com/LiorSinai/TFIDF.jl

Using a transformer for this task can be seen as excessive because it can be solved with simpler models e.g. a term frequency inverse document infrequency (TFIDF) model with 10,000 parameters. (You can see my Julia TFIDF model [here][TFIDF].) 
However because the task is simple it means we can limit the transformer model to around 250,000 parameters and we have a good baseline of the accuracy we can achieve.

I will not focus on the intuition behind transformers. 
For that I recommend Peter Bloem's excellent post [Transformers from scratch][Bloem].
For code in a more popular framework I recommend Alexander Rush's  [The annotated transformer][AnnotatedTransformer] written in PyTorch.
Many transformer blog posts focus on another universal language, pictures. 
Amongst the most notable are Jay Alammar's [the illustrated transformer][IllustratedTransformer] 
and a [video][YouTubeTransformer] by Ari Seff. I'll use pictures too but it won't be the primary medium.

This is not meant to be a full scale working Julia solution.
For that, please see the [Transformers.jl][Transformersjl] package. 
It has better optimizations, CUDA support, APIs for HuggingFace and more. 
My own repository with the code in this blog post can be accessed at [github.com/LiorSinai/TransformersLite.jl](https://github.com/LiorSinai/TransformersLite.jl).

Lastly, transformers are built on top of research and ideas of the last decade of machine learning research.
It is recommended you have a background in simpler models.
For my part, I'll briefly explain the ideas behind techniques like word embeddings, skip connections, regularization and so on
and I'll link to papers and blog posts that go into much more detail for each.
It is also worth keeping in mind that machine learning is at its heart an empirical science, and a sufficient if maybe unsatisfactory answer for why most of these techniques are used is that they have given good results in the past.

## Architecture
### Design considerations

First it is important to look at the design considerations that the Google AI team prioritised.

In the current era of machine learning, the best outcomes have been achieved with bigger models and more data.
To facilitate this, a new architecture should have fast execution times and allow for parallel execution.

<figure class="post-figure">
<img class="img-95"
    src="/assets/posts/transformers/scale_linear.png"
	alt="scale linear - non-linear"
	>
<figcaption></figcaption>
</figure>

Let's consider a scale of linear to non-linear.
In computer terms, fast means simple and simple means linear. On the other side is non-linear which is complex and slow.
Machine learning as used today is mostly linear. 
Anyone who first studies it is a little overwhelmed with all the linear algebra and matrices.
Some non-linearity is needed but research has found that one can get away with very little.
For example, only non-linearity in activation functions.
CNNs are more non-linear, mostly because of the strides of the kernels across images.

<figure class="post-figure">
<img class="img-95"
    src="/assets/posts/transformers/scale_parallel.png"
	alt="scale parallel - sequential"
	>
<figcaption></figcaption>
</figure>

The other scale is parallel to sequential. The previous favourite for NLP, RNNs, were very sequential.
This makes sense for language where words themselves are sequential and the location and order of words in a sentence is important.
Yet transformers are mostly parallel. This enables computers to distribute computation across GPU cores and across clusters.
To reconcile this with sequential language two techniques are used: position encoding and masking. 
In practice these "hacks" perform well enough to justify using a parallel model for sequential tasks.

### Inputs and outputs

The input to a transformer is a sentence. For example "This camera works great!". That is then processed into tokens.
Depending on the complexity desired, these tokens can represent words, punctuation, and subword pieces like suffixes. For examples: 
`[This, camera, work, ##s, great, !]`. A simpler approach is to remove case and punctuation: `[this, camera, works, great]`.

[WordEmbeddings1]: https://lena-voita.github.io/nlp_course/word_embeddings.html
[WordEmbeddings2]: https://neptune.ai/blog/word-embeddings-guide

<div class="message-container info-message">
	<div class="message-icon fa fa-fw fa-2x fa-exclamation-circle">
	</div>
	<div class="content-container">
		<div class="message-body">
		Julia uses column major format whereas Python uses row major format. In Julia word vectors are columns while in Python they are rows.
		Equations between the two formats will look backwards to each other.
		They need to be transposed and definitions also need to be transposed. 
		E.g. $K^TQ \rightarrow (K_c^TQ_c)^T=Q_c^TK_c= Q_r K_r^T$
		</div>
	</div>
</div>

Each token is then associated with a series of weights called a word vector or a word embedding. For example, for our sentence:
<figure class="post-figure">
<img class="img-50"
    src="/assets/posts/transformers/inputs.png"
	alt="architecture"
	>
<figcaption></figcaption>
</figure>
Each word has $d_m$ weights associated with it. For typical transformer models $d_m$ ranges from 64 to 2048.
The main idea is that each weight represents some concept.
For example if we were to assign them manually we could go according to this scheme:
- the first row is for how positive or negative the word is.
- the second row is for how common it is.
- the third row is for if it is a noun or not.

In practice these weights will be assigned during training and it can be hard to interpret them.

Here are two good references on word embeddings: [1][WordEmbeddings1] and [2][WordEmbeddings2].

The output of the transformer is another set of weights usually of the same size:
<figure class="post-figure">
<img class="img-50"
    src="/assets/posts/transformers/outputs.png"
	alt="architecture"
	>
<figcaption></figcaption>
</figure>
But now these weights have a different meaning. They are how each word relates to each other in the sentence and to the task at hand.
Where as the previous weights were unique to each word/token, these are particular to a sentence.
Change one word in the sentence and we might get a completely different matrix, depending on how important that word is.
The name transformer comes from the fact that it _transforms_ a set of word embeddings to another set of embeddings.

The transformer takes advantage of linear algebra to calculate all the weights for all words at the same time.
Hence it is mostly a parallel computation. 
Without additional measures we could shuffle the words (columns) in the sentence and we would get shuffled versions of the same results.
To parallelize further, we can stack the embedding matrices of sentences of $N$ words into $B$ batches of $d_m\times N \times B$ dimensional arrays.

Because the output looks like the input we can feed it back into another transformer. 
Transformer models tend to have stacks of 6 to 24 of these layers.
With each layer it gets more difficult to interpret what the embedding weights actually mean. 
But stacking them definitely does improve accuracy.

The final output matrix is usually fed to a small neural network to output the final result.
This could be a single number in the case of sentiment analysis or a probability of each word in a vocabulary for a translation task.

### Description

This is the famous schematic from the paper [Attention is all you need][AttentionIsAllYouNeed]:

<figure class="post-figure">
<img class="img-50"
    src="/assets/posts/transformers/architecture.png"
	alt="architecture"
	>
<figcaption></figcaption>
</figure>

The left side is called an encoder and the right side a decoder. 
You don't need both sides. [BERT][BERT] is an encoder-only model by Google. [Gato][Gato] is a decoder only transformer.
In this post I'll focus only on the encoder which is sufficient for the classification task.

Each block in the schematic is associated with two sets of equations: the forward equations and the backwards equations.
I have compiled the encoder block equations into the table below.
The task for the remainder of the blog post will be to translate this maths into code.
<figure class="post-figure">
<img class="img-95"
    src="/assets/posts/transformers/transformer_equations.png"
	alt="architecture"
	>
<figcaption></figcaption>
</figure>
Please see this table only as a guideline.
Some of the backpropagation equations are incomplete so that they fit in this table.
Equations will be presented properly in each subsection.
$J_\alpha(Z)\equiv	\frac{\partial Z}{\partial \alpha}$.
There are 4 non-linear steps (softmax, 2&times;layer norm, RELU) and the other 8 are linear.

You'll notice that the inputs are either 3D or 4D arrays. 
These are not standard in linear algebra so a [section](#working-with-higher-order-arrays) below is dedicated to getting comfortable with them.
In particular, in any programming language multiplying two higher order arrays will not work e.g. `A*B`.
Multiplication simply isn't defined for them. So we'll have to write our own function to handle the multiplication here and also for the backpropagation. We'll do this as a simple extension to 2D matrix multiplication.

[wiki_tensor]: https://en.wikipedia.org/wiki/Tensor

There is a set of multidimensional algebraic objects called [tensors][wiki_tensor] where multiplication is defined for higher orders. 
Confusingly, Google named their machine learning framework TensorFlow and calls higher order arrays tensors.
So one should differentiate between machine learning tensors and geometric tensors.
They are not the same.
To give a simple explanation: one can think of geometric tensors as higher order arrays with severe constraints on their entries and operations. These constraints make it harder, not easier, to code higher order arrays as geometric tensors.
We'll try anyway.

## Julia implementation

### Project setup
### Tokenizers
### Word embeddings
### Position encodings
### Working with higher order arrays
### Multi-head attention

[CosineSimilarity]: https://www.machinelearningplus.com/nlp/cosine-similarity/
### Encoder blocks
### Classifier

## Use case: Amazon Reviews 

## Conclusion

Thank you for following me along this journey into the fascinating mathematics of quaternions.
I hope you've enjoyed it and are fully comfortable with using them in animations now.

Please explore the references in [part 1][refs] for more information.

[refs]:{{ "mathematics/2021/11/05/quaternion-1-intro#references" | relative_url }}

---

[^amazon_multi]: HuggingFace would prefer you use their Python API to download the dataset. This is as simple as 
	```
	from datasets import load_dataset
	dataset = load_dataset('amazon_reviews_multi', 'en')
	```

	You can also use curl in a terminal:

	```
	curl https://amazon-reviews-ml.s3-us-west-2.amazonaws.com/json/train/dataset_en_train.json --output amazon_reviews_en_train.json
	curl https://amazon-reviews-ml.s3-us-west-2.amazonaws.com/json/test/dataset_en_test.json --output amazon_reviews_en_test.json
	```

