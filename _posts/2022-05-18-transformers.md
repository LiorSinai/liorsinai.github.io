---
layout: post
title:  "Building a transformer in Julia"
date:   2022-05-18
author: Lior Sinai
background: '/assets/posts/transformers/transformer.png'
categories: coding
tags: mathematics transformers 'machine learning' 'deep learning'
---

_Building a transformer in Julia. This a very long post on the full process behind making a transformer work in Julia._ 


### Table of Contents
- [Introduction](#introduction)
- [Design](#design)
	- [Design considerations](#design-considerations)
	- [Inputs and outputs](#inputs-and-outputs)
	- [Architecture](#architecture)
    - [Attention](#attention)
- [Julia implementation](#julia-implementation)
	- [Project setup](#project-setup)
	- [Tokenizers](#tokenizers)
	- [Word embeddings](#word-embeddings)
	- [Position encodings](#position-encodings)
	- [Multiplication with higher order arrays](#multiplication-with-higher-order-arrays)
	- [Multi-head attention](#multi-head-attention)
	- [Encoder blocks](#encoder-blocks)
	- [Classifier](#classifier)
- [Use case: Amazon reviews](#use-case-amazon-reviews)
- [Conclusion](#conclusion)

## Introduction

[Attention]: https://arxiv.org/abs/1706.03762
[NvideaGTC]: https://youtu.be/39ubNuxnrK8?t=772
[BERT]: https://arxiv.org/abs/1810.04805
[GPT]: https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf
[GPT2]: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
[GPT3]: https://arxiv.org/abs/2005.14165
[GPT3_api]: https://openai.com/api/
[Gato]: https://www.deepmind.com/publications/a-generalist-agent
[Image16words]: https://arxiv.org/abs/2010.11929
[NvideaMegatron]: https://arxiv.org/abs/1909.08053
[NvideaLM]: https://github.com/NVIDIA/Megatron-LM

In December 2017 Google AI released their transformer architecture in the paper [Attention is all you need][Attention]. 
(It is a highly recommended read.)
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
The CEO of chip manufacturer NVIDEA, Jensen Huang, focused a segment of his [2022 keynote][NvideaGTC] speech on transformers and their impact on his industry.
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

All this development in transformers has been over the past 5 years.
In that time they have mostly replaced the old favourite for NLP in academia and industry, recurrent neural networks (RNNs).
However that has not been long enough for pedagogy to adapt.
As of today, machine learning courses still teach RNNs for NLP.
This has created a gap and many blogs have sprung up to full it.
This blog post aims to be one of those.

I have two goals here:
1. Build a small working transformer in Julia code and train it on one use case.
2. Detail the mathematics of the transformer for both forward equations and backpropagation.

Julia has a small but growing user base. It is an elegant and fast language and I highly recommend it.
But even if you don't know it well, I hope you will find this post accessible and that it will help improve your Julia. 
Mathematics on the other hand is a universal language and this should be accessible to anyone with university level maths. 

[AmazonReviews]: https://huggingface.co/datasets/amazon_reviews_multi

The use case is a dataset of [Amazon reviews from HuggingFace][AmazonReviews][^amazon_multi]. Only the English subset of the dataset was used with 200,000 training samples and 5,000 test samples. The models were trained on two tasks:
1. Predict the star rating given the review text. 
2. Predict a positive or negative sentiment with 1-2 stars labelled negative, 4-5 stars labelled positive and 3 stars removed.

[TFIDF]: https://github.com/LiorSinai/TFIDF.jl

Using a transformer for this task can be seen as excessive because it can be solved with simpler models e.g. a term frequency inverse document infrequency (TFIDF) model with 10,000 parameters. (You can see my Julia TFIDF model [here][TFIDF].) 
However because the task is simple it means we can limit the transformer model to around 250,000 parameters and we have a good baseline of the accuracy we can achieve.

For intuition and history behind transformers I recommend Peter Bloem's excellent post [Transformers from scratch][Bloem].
For code in a more popular framework I recommend Alexander Rush's  [The annotated transformer][AnnotatedTransformer] written in PyTorch.
Many transformer posts focus on another universal language, pictures. 
Amongst the most notable are Jay Alammar's [the illustrated transformer][IllustratedTransformer] 
and a [video][YouTubeTransformer] by Ari Seff. I'll use pictures too but it won't be the primary medium.

This is not meant to be a full scale Julia solution.
For that, please see the [Transformers.jl][Transformersjl] package. 
It has better optimizations, CUDA support, APIs for HuggingFace and more. 
My own repository with the code in this blog post can be accessed at [github.com/LiorSinai/TransformersLite.jl](https://github.com/LiorSinai/TransformersLite.jl).

Lastly, transformers are built on top of research and ideas of the last decade of machine learning research.
A background in neural networks is needed for this post.
For my part, I'll briefly explain the ideas behind techniques like word embeddings, skip connections, regularization and so on
and provide some references.
I also encourage you to research more on your own.
It is also worth keeping in mind that machine learning is at its heart an empirical science, and a sufficient if maybe unsatisfactory answer for why most of these techniques are used is that they have given good results in the past.

## Design
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
Anyone who first studies it is surely a little overwhelmed with all the linear algebra and matrices.
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

The output of the transformer is another set of weights usually of the same size:
<figure class="post-figure">
<img class="img-50"
    src="/assets/posts/transformers/outputs.png"
	alt="architecture"
	>
<figcaption></figcaption>
</figure>
But now these weights have a different meaning. They are how each word relates to each other in the sentence and to the task at hand.
Where as the previous weights were unique to each word, these are particular to a sentence.
Change one word in the sentence and we might get a completely different matrix, depending on how important that word is.
The name transformer comes from the fact that it _transforms_ a set of word embeddings to another set of embeddings.

The transformer takes advantage of linear algebra to calculate all the weights for all words at the same time.
Hence it is mostly a parallel computation. 
Without additional measures we could shuffle the words (columns) in the sentence and we would get shuffled versions of the same results.
To parallelize further, we can stack the embedding matrices of sentences of $N$ words into $B$ batches of $d_m\times N \times B$ dimensional arrays.

Because the output looks like the input we can feed it back into another transformer. 
Transformer models tend to have stacks of 6 to 24 of these layers.
With each layer it gets more difficult to interpret what the embedding weights actually mean. 
But stacking them has been shown to improve results.

The final output matrix is usually fed to a small neural network to output the final result.
This could be a single number in the case of sentiment analysis or a probability of each word in a vocabulary for a translation task.

### Architecture

This is the famous schematic from the paper [Attention is all you need][Attention]:

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
<img class="img-95" id="equations_table"
    src="/assets/posts/transformers/transformer_equations.png"
	alt="architecture"
	>
<figcaption></figcaption>
</figure>
Please see this table only as a guideline.
Some of the equations are incomplete so that they fit in this table.
Equations will be presented properly in each subsection.
$J_\alpha(Z)\equiv	\frac{\partial Z}{\partial \alpha}$.
There are 4 non-linear steps (softmax, 2&times;layer norm, RELU) and the other 8 are linear.

You'll notice that the inputs are either 3D or 4D arrays. 
These are not standard in linear algebra so a [section](#multiplication-with-higher-order-arrays) below is dedicated to getting comfortable with them.
In particular, in any programming language multiplying two higher order arrays will not work e.g. `A*B`.
Multiplication simply isn't defined for them. So we'll have to write our own function to handle the multiplication here and also for the backpropagation. We'll do this as a simple extension to 2D matrix multiplication.

There is a set of multidimensional algebraic objects called [tensors][wiki_tensor] where multiplication is defined for higher orders. 
Confusingly, Google named their machine learning framework TensorFlow and calls higher order arrays tensors.
So one should differentiate between machine learning tensors and geometric tensors.
They are not the same.
To give a simple explanation: one can think of geometric tensors as higher order arrays with severe constraints on their entries and operations because they represent geometric objects. These constraints make it harder - not easier - to code higher order arrays as geometric tensors.

### Attention

The most important steps in the above table are the attention steps.
Combing them all into one and working with only 2D matrices, we get the definition for the scaled dot product attention:

$$
    A = \text{softmax}\left(\frac{1}{\sqrt{d_h}}K^T Q\right)V
$$

For both the encoder and decoder $K$ and $Q$ are calculated as the outputs of dense layers with the embedding matrix $X$.
Substituting it into $K^TQ$ and ignoring the bias we get:

$$
    K^T Q = (W_KX)(W_QX)^T = W_K XX^T W_Q^T
$$

[CosineSimilarity]: https://www.machinelearningplus.com/nlp/cosine-similarity/

I hope this is recognisable as a dot product/inner product. 
The Google authors call it a different name, attention, and it is apparently all you need. 
It is very closely related to an older machine learning technique called [cosine similarity][CosineSimilarity].

Every word is multiplied with the embeddings of every other word, resulting in a small $N \times N$ matrix.
The hope is the output looks something like:
<figure class="post-figure">
<img 
    src="/assets/posts/transformers/attention.png"
	alt="architecture"
	>
<figcaption></figcaption>
</figure>
Reading down the columns we have an approximate weighting of how much every word thinks every other word is important.
Here "camera" thinks "great" is the most important word in the sentence.
Because of the weights this matrix is not symmetrical.
So "great" actually places less importance on "camera" than "camera" places on it.

This matrix is at the heart of transformer.
All the other layers work towards it or aim to use weights output by it. 


## Julia implementation
### Project setup

We'll be making use of the Flux framework along with the NNlib and ChainRulesCore packages.

An example output will be:
{%highlight julia %}
TransformerClassifier(
     Embed(32, 7455),                   # 238_560 parameters
     PositionEncoding(32),
     Dropout(0.1),
     TransformerEncoderBlock(
          MultiheadAttention(num_heads=4, head_size=8, 32=>32)(
               denseQ = Dense(32 => 32),  # 1_056 parameters
               denseK = Dense(32 => 32),  # 1_056 parameters
               denseV = Dense(32 => 32),  # 1_056 parameters
               denseO = Dense(32 => 32),  # 1_056 parameters
          )
          Dropout(0.1),
          LayerNorm(32),                # 64 parameters
          Dense(32 => 128, relu),       # 4_224 parameters
          Dense(128 => 32),             # 4_128 parameters
          Dropout(0.1),
          LayerNorm(32),                # 64 parameters
     )
     Dense(32 => 1),                    # 33 parameters
     FlattenLayer(),
     Dense(50 => 5),                    # 255 parameters
)                  # Total: 21 arrays, 251_552 parameters, 1.083 MiB
{% endhighlight %}

The `Dropout`, `LayerNorm` and `Dense` layers are already part of the Flux package.
We'll be making `Embed`, `PositionEncoding`, `MultiheadAttention`, `TransformerEncoderBlock` and `FlattenLayer`.
We'll also make a small index tokenizer to map tokens to word vectors.

The focus will be on the forward equations because Flux handles the backwards equation through automatic differentiation (AD).
Other than reducing our job in half, AD also means our forward and backwards equations will always be in sync. 
There will be collapsible blocks with backpropagation information &#8681;.

To start, make a package in the Julia REPL:
<figure class="highlight">
    <code class="language-julia-repl hljs" data-lang="julia-repl">
        <span class="hljs-meta">julia&gt;</span><span class="julia"> cd(<span class="hljs-string">"path\\to\\project\\directory"</span>)</span>
        <br>
        <span class="hljs-meta">julia&gt;</span><span class="julia"> ] <span class="hljs-comment"># enter package mode</span></span>
        <br>
        <span class="hljs-meta">(@v1.x) pkg&gt;</span><span class="julia"> generate TransformersLite <span class="hljs-comment"># make a directory structure</span></span>
        <br> 
        <span class="hljs-meta">(@v1.x) pkg&gt;</span><span class="julia"> activate TransformersLite <span class="hljs-comment"># activate package environment</span></span>
        <span class="hljs-meta">(TransformersLite) pkg&gt;</span><span class="julia"> add Flux NNlib ChainRulesCore </span>
        <br> 
        <span class="hljs-meta">(TransformersLite) pkg&gt;</span><span class="julia"> activate </span>
        <br> 
        <span class="hljs-meta">(@v1.x) pkg&gt;</span><span class="julia"> dev "path\\to\\project\\directory\\TransformersLite"</span>
    </code>
</figure>

The goal of using the package manager is that we can now use the super helpful Revise package,
which will dynamically update most changes during development without errors:
{%highlight julia-repl %}
julia> using Revise
julia> using TransformersLite
{% endhighlight %}  

You can see my final code at [github.com/LiorSinai/TransformersLite.jl](https://github.com/LiorSinai/TransformersLite.jl).
This is based loosely on the registered [Transformers.jl][Transformersjl] package. 

### Tokenizers
[HuggingFaceBPE]: https://huggingface.co/course/chapter6/5?fw=pt

The input is a sentence when we need to break up into tokens. 
This preprocessing step is a huge topic itself.
To avoid spending too much time here, I am going to provide functions for cleaning the text.
They put all text in lowercase, normalize unicode to ASCII e.g. "é" to "e" and "don't" to "dont" and split sentences into words.
The regex for the latter is <code style="white-space:nowrap">[A-Za-z][A-Za-z]+\b</code> which finds all words with more than two ASCII letters without numbers.

{%highlight julia %}
using Unicode
function clean(s::AbstractString)
    s = lowercase(s)
    s = Unicode.normalize(s, :NFD)
    s = replace(s, r"['`’\u200d\p{M}]" => "") # contractions, zero width joiner and marks from normalization
    s = replace(s, r"\n" => " ")
end

function preprocess(document, tokenizer; pattern = r"[A-Za-z][A-Za-z]+\b", max_length::Union{Nothing, Int}=nothing)
    document = clean(document)
    words = map(m->string(m.match), eachmatch(pattern, document))
    tokens = tokenizer(words)
    if !isnothing(max_length)
        if length(tokens) > max_length
            tokens = tokens[1:max_length]
        end
    end
    tokens
end
{% endhighlight %}

The `preprocess` function requires a tokenizer for subword tokenization.
I have made simple tokenizers at [github.com/LiorSinai/TokenizersLite.jl](https://github.com/LiorSinai/TokenizersLite).
You can also use the registered BytePairEncoding.jl package.
Or if you do not want subword tokenization use `tokenizer=identity`.
This is sufficient for the Amazon Reviews problem that we will investigate later.

Once we have tokens we do need to map them to word embeddings.
For this we'll make a simple `IndexTokenizer`:
{%highlight julia %}
struct IndexTokenizer{T}
    vocabulary::Vector{T}
    unksym::T
    unkidx::Int
    function IndexTokenizer(vocab::Vector{T}, unksym::T) where T
        if !(unksym ∈ vocab)
            pushfirst!(vocab, unksym)
            unkidx = 1
        else
            unkidx = findfirst(isequal(unksym), vocab)
        end
        new{T}(vocab, unksym, unkidx)
    end
end

Base.length(tokenizer::IndexTokenizer) = length(tokenizer.vocabulary)

function Base.show(io::IO, tokenizer::IndexTokenizer) 
    T = eltype(tokenizer.vocabulary)
    print(io, "IndexTokenizer{$(T)}(length(vocabulary)=$(length(tokenizer)), unksym=$(tokenizer.unksym))")
end
{% endhighlight %}

This `IndexTokenizer` takes in a list of tokens and an unknown symbol. 
The constructor function checks if the unknown symbol is in the list else it adds it to the front.

For the encoding process, we need to to replace a token with an index if it is in the vocabulary list and with the unknown symbol index (by default 1) if it is not:
{%highlight julia %}
encode(tokenizer::IndexTokenizer{T}, x::T) where T = something(
	findfirst(isequal(x), tokenizer.vocabulary), tokenizer.unkidx)
{% endhighlight %}

This assumes we are giving a single token of type `T`. 
We also want to do multiple dispatch on sentences, which are `Vector{T}` and on batches of sentences, or `Vector{Vector{T}}`.
When with working with batches we'll need all sentence to be the same length.
We truncate long sentences (already done in `preprocess`) and we can introduce a padding token for sentence longer than some maximum length.
Here the unknown token is used for padding:
{%highlight julia %}
function encode(tokenizer::IndexTokenizer{T}, seq::AbstractVector{T}) where T
    map(x->encode(tokenizer, x), seq)
end

function encode(tokenizer::IndexTokenizer{T}, batch::AbstractVector{Vector{T}}) where T
    lengths = map(length, batch)
    indices = fill(tokenizer.unkidx, maximum(lengths), length(batch))
    for (i, seq) ∈ enumerate(batch)
        for (j, x) ∈ enumerate(seq)
            @inbounds indices[j, i] = encode(tokenizer, x)
        end
    end
    indices
end
{% endhighlight %}

Lastly we can add a method to do multiple dispatch on the type `IndexTokenizer` itself, 
which turns this struct into a function:
{%highlight julia %}
(tokenizer::IndexTokenizer)(x) = encode(tokenizer, x)
{% endhighlight %}

In practice:
{%highlight julia %}
vocab = load_vocab("amazon_reviews_train_en.txt")
indexer = IndexTokenizer(vocab, "[UNK]")

text = "This coffee from Kenya is really good."
tokens = preprocess(text, identity) # [this,coffee,from,kenya,is,really,good]
indices = indexer(tokens) # [8,534,50,1,6,56,30]"
{% endhighlight %}

The vocabulary file is at this [link](https://github.com/LiorSinai/TransformersLite.jl/blob/main/vocab/amazon_reviews_train_en.txt) and the `load_vocab` function is:
{%highlight julia %}
function load_vocab(filepath)
    vocab = String[]
    open(filepath, "r") do file
        for line in eachline(file)
            push!(vocab, line)
        end
    end
    vocab
end
{% endhighlight %}

The vocabulary is sorted from highest to lowest of the word counts in the original data . 
So if we limit the vocabulary e.g. `vocab[1:1000]` we can still be confident that it will have statistical significance.

### Word embeddings
Word embeddings were already introduced in the [Inputs and Outputs](#inputs-and-outputs) section.
Here we'll make a simple layer to store and retrieve them.

It it worth highlighting that the word embedding is unique to each model 
and will be trained from random values for each model.
This is not how humans work. 
Part of what makes language so useful is that we can have generic connotations and meanings for words and then derive more specific meaning from them in specific contexts. So for example, the word "good" always has the same "embedding" in any context.
But here we learn a different embedding for different models and even different training runs.

There are several justifications for this:
1. Word embeddings are task specific: for example in the Amazon review context "return" is a highly negative word associated with returning 
a defective product to the store. In other tasks it may be far more neutral.
2. The tokenizer strategy from the previous section might change, or we might want to experiment with different tokenizers.
3. We can tune the model dimension $d_m$ as a hyperparameter to make bigger or smaller models.

This is somewhat unfortunate as it forms a massive part of our training. 
For the model I will use later it will be 95% of the trainable parameters.

The embedding layer is a struct that holds a matrix:
{%highlight julia %}
struct Embed{W <: AbstractArray}
    embedding::W
end

Flux.@functor Embed # tell Flux that this struct is trainable

Embed(output_dim::Int, vocab_size::Int) = Embed(randn(Float32, output_dim, vocab_size))

Base.size(e::Embed) = size(e.embedding)

Base.show(io::IO, e::Embed) = print(io, "Embed($(size(e.embedding)))")

function Base.show(io::IO, m::MIME"text/plain", e::Embed)
    Flux._layer_show(io, e)
end
{% endhighlight %}
We use `Float32` to reduce the size of the model and for performance benefits. We don't need the extra accuracy provided by `Float64`.
We have a second show function for multimedia (MIME) types when we went prettier printing e.g. in the REPL and Jupyter notebooks.
The `Flux.@functor Embed` line is essential for Flux to be able to perform backpropagation.

For the forward pass we will use `NNlib.gather`:
{%highlight julia %}
using NNlib: gather
function (e::Embed)(x::AbstractArray{Int})
    gather(e.embedding, x)
end
{% endhighlight %}
This is equivalent to `e.embedding[:, x]`. However using gather means that the `rrule` has already been defined for it.
See [here](https://github.com/FluxML/NNlib.jl/blob/ff3ac6eb807e9b41f46f28f8b3287d19f4b722c7/src/gather.jl#L80).

<div class="message-container info-message">
	<div class="message-icon fa fa-fw fa-2x fa-exclamation-circle">
	</div>
	<div class="content-container">
		<div class="message-body">
		ChainRulesCore uses <code>rrule</code> to define backprogation rules.
		The old standard was the badly named <code>@adjoint</code> meaining the Jacobian adjoint meaning the conjugate transpose
		of the Jacobian. 
		This is different to the <code>adjoint</code> function which is the complex transpose of a matrix and the classical adjoint or adjugate which is the transpose of the cofactors of a matrix.
		</div>
	</div>
</div>

The `rrule` is a reverse (backwards) rule that encodes the derivative for backpropagation. 
It is what makes the magic of automatic differentiation work.

The function `gather` does not have a formal derivative, but scatter is the opposite of it and is what we need to apply when we calculate the loss:
<figure class="post-figure">
<img class="img-60"
    src="/assets/posts/transformers/gather.png"
	alt="architecture"
	>
<figcaption></figcaption>
</figure>

At the end of backpropagation we need to distribute the error matrix amongst the original word embeddings.
This is what `scatter` does. Note that we use the red column twice, so we have two error columns directed towards it.
The `rrule` applies `+` as the reducing function; that is, the two errors are added together and then to the word embedding.

There is a cost to using a predefined function: it is very inefficient.
If we do a small experiment and call scatter we will see it results in a large matrix of mostly zeros:
{%highlight julia %}
NNlib.scatter(+, rand(8, 4), [1, 5, 11, 1]; dstsize=(8, 15))
{% endhighlight %}
This could be improved with a custom scatter function using the `SparseArrays` package.

### Position encodings

As mentioned before, the matrix operations on the embedding matrix are parallel operations.
They do not take order into account.
One of the proposed methods proposed in the [Attention is all you need][Attention] paper to counteract this is to add constant value "position encodings" to the matrix.
The hope is that the model will learn these constant values and hence the relative and absolute positions.
For example a simple choice for each column $j$ is $\frac{j}{n_{max}}$: 

$$
\begin{bmatrix} 
    \frac{1}{n_{max}} & \frac{2}{n_{max}} & \cdots & \frac{n_{max}-1}{n_{max}} &  1 \\
    \vdots            &  \vdots           & \ddots & \vdots  &  1 \\
    \frac{1}{n_{max}} & \frac{2}{n_{max}} & \cdots & \frac{n_{max}-1}{n_{max}}  &  1
\end{bmatrix}
$$

A problem with this encoding is that it is dependent on the parameter $n_{max}$, which fixes the sequence length.
Instead the authors propose a more convoluted solution but one that can be easily scaled to any sequence length:

$$
\begin{align} 
    PE(2i + 1, j) &= \sin(j/(10^4)^{2i/d}) \\
    PE(2i + 2, j) &= \cos(j/(10^4)^{2i/d})
\end{align}
$$

Plotted here on the left is a heatmap of the resultant matrix 
and on the right are the sine waves used for the odd numbered rows:
<style>
    .slider {
    position: relative;
    width: 70%;
    margin-left: auto
    }
    .sliderValue{
    position:relative;
    display:inline;
    margin: 0.5em;
    }
    .sliderText{
    position:relative;
    display:inline;
    }
    .graph{
        width:100%
    }
</style>
<div style="display:flex;max-width:95%">
    <img style="width:55%"
        src="/assets/posts/transformers/position_encodings.png"
        alt="architecture"
        >
    <div style="width:45%"> 
        <img class="graph" id="sineGraph"
            src="/assets/posts/transformers/position_encoding_sin0.png"
            alt="sine graphs"
            >
        <div class="slider">
            <p class="sliderText">i</p>
            <input type="range" min="0" max="15" value="0" id="graphSlider">
            <p class="sliderValue" id="graphSliderValue">0</p>
        </div>
    </div>
</div>
<script>
    const sliderValue = document.getElementById("graphSliderValue")
    const slider = document.getElementById("graphSlider")
    const sineGraph = document.getElementById("sineGraph")
    const update = () => { 
        sliderValue.innerText = +slider.value
        filename = "/assets/posts/transformers/position_encoding_sin" + (2*slider.value) + ".png"
        sineGraph.src = filename
    }
    slider.oninput = () => update();
</script>

Each column has a unique pattern so the encoding does accomplishes its task.
To understand why, lets focus on the first row with $i=0$. 
This sine wave has a wavelength of $2\pi \approx 6.28$ and we sample it every $1$ timestep so it repeats every 6 blocks.
This leads to the 6 alternating colours in the top row: 3 light, then 3 dark, then repeat. 
So this sine wave can only distinguish between sequences of lenght 6 or next.
Now let's move on to $i=1$. This sine wave has a period of $2\pi(10^4)^{1/32} \approx 11.17$ so it repeats approximately every 11 blocks in the 3rd row. 
We can now distinguish between sequences of up to length 11 and we can use the first row for greater precision.
As we add sine waves, we can distinguish between sequences of longer wave lengths.
In general the wavelengths are $2\pi(10^4)^{i/d}$.

[position_encoding]: https://kazemnejad.com/blog/transformer_architecture_positional_encoding/

The remaining question, is why use both sine and cosine waves? 
The answer in the paper is: "We chose this function because we hypothesized it would allow the model to easily learn to attend by
relative positions, since for any fixed offset $k$, $PE_{j+k}$ can be represented as a linear function of
$PE_{j}$." Here they are referring to the identities:

$$
    \sin(\omega k + \omega j) = \sin(\omega k)\cos(\omega j) + \cos(\omega k)\sin(\omega j) \\
    \cos(\omega k + \omega j) = \cos(\omega k)\cos(\omega j) + \sin(\omega k)\sin(\omega j) \\
$$

which are linear for constant $\omega$ and $k$. 
For more detail please see [here][position_encoding].[^linear_pe]

Now let's code the `PositionEncoding` layer.
Since these values are constant, it is easiest to preallocate a matrix:
{%highlight julia %}
struct PositionEncoding{W <: AbstractArray}
    encoding::W
end

function PositionEncoding(dim_embedding::Int, max_length::Int=1000)
    W = make_position_encoding(dim_embedding, max_length)
    PositionEncoding(W)
end

function make_position_encoding(dim_embedding::Int, seq_length::Int, n::Int=10000)
    encoding = Matrix{Float32}(undef, dim_embedding, seq_length)
    for pos in 1:seq_length
        for row in 0:2:(dim_embedding - 1)
            denom = 1/(n^(row/dim_embedding))
            encoding[row + 1, pos] = sin(pos * denom)
            encoding[row + 2, pos] = cos(pos * denom)
        end
    end
    encoding    
end

function Base.show(io::IO, pe::PositionEncoding)
    print(io, "PositionEncoding($(size(pe.encoding, 1)))")
end
{% endhighlight %}

The forward pass then selects the required columns from the pre-allocated array:
{%highlight julia %}
(pe::PositionEncoding)(x::AbstractArray) = (pe::PositionEncoding)(size(x, 2))
function (pe::PositionEncoding)(seq_length)
    max_length = size(pe.encoding, 2)
    if seq_length > max_length
        error("sequence length of $seq_length exceeds maximum position encoding length of $max_length")
    end
    pe.encoding[:, Base.OneTo(seq_length)]
end
{% endhighlight %}

Here an error is raised if the size of the pre-allocated matrix is exceeded.
We could instead calculate the extra columns required or resize the encoding matrix.
For now this only adds extra complexity.

Also note this layer only returns the encoding, so we need to add it separately:
{%highlight julia %}
X = rand(32, 100, 16)
pe = PositionEncoding(32)
Z = X .+ pe(X) # broadcast the 2D encoding matrix to 3D
{% endhighlight %}

If desired we can move the addition into the forward pass e.g. for use in `Flux.chain`.

### Multiplication with higher order arrays

Starting in 2D, matrix multiplication for $A\times B$ is defined as the sum of rows in $A$ multiplied with the columns in $B$:

$$ C_{ij} = \sum_r A_{ir} B_{rj} $$

This can be written as a set of three loops (ignoring checks):
{%highlight julia %}
function mul2d(A<:AbstractMatrix, B<:AbstractMatrix)
	n = size(A, 2) # == size(B, 1)
	C = zeros(size(A, 1), size(B, 2))
	for i in 1:size(A, 1)
		for j in 1:size(B, 2)
			for r in 1:n
				C[i, j] += A[i, r] * B[r, j]
			end
		end
	end
	C
end
{% endhighlight %}
 
Of course many programming languages already have this function built in. 
They have highly optimised this simple function. 
We can do a quick time test:
{%highlight julia %}
    A = randn(100, 100);
    B = randn(100, 100);
    @time mul2d(A, B); # 0.002391s
    @time A * B;       # 0.000265s
{% endhighlight %}
The naive implementation is 9&times; slower.
One of the reasons is our indexing is very naive.
The code will start at the top and count down to the cell needed for each multiplication
when we could take advantage of the fact that the next cell is next door:

<figure class="post-figure">
<img class="img-90"
    src="/assets/posts/transformers/indexing.png"
	alt="indexing"
	>
<figcaption></figcaption>
</figure>

Later in machine learning came the idea of batch multiplication.
This is doing multiplications for a set of independent matrices simultaneously by grouping them into one large 3D array:

$$ C_{ijk} = \sum_r A_{irk} B_{rjk} $$

We could write this as a set of four loops. 
Or since we know the inbuilt `*` is faster we can substitute that for the three inner loops:
{%highlight julia %}
function mul3d(A::AbstractArray{T, 3}, B::AbstractArray{T, 3}) where T
    C = Array{Float64}(undef, size(A, 1), size(B, 2), size(A, 3))
	for k in 1:size(A, 3)
		C[:, :, k] = A[:, :, k] * B[:, :, k]
	end
	C
end
{% endhighlight %}
But this doesn't take advantage of of the fact that we are standardising the size of the matrices (all sequences are of the same length).
It is equivalent to using type `Vector{Matrix}` rather than `Array{T, 3}`.
NNlib has written a more optimised version called `batched_mul`. 
Doing a time test:
{%highlight julia %}
    Using NNlib
    A = randn(100, 100, 32);
    B = randn(100, 100, 32);
    @time mul3d(A, B);       # 0.010918s
    @time batched_mul(A, B); # 0.006588s
{% endhighlight %}
The NNlib function is about 1.5&times; faster.

For transformers we work with 4D arrays but they are sets of sets of independent matrices (repetition intended).
So multiplication is a set of five loops or two outer loops and matrix multiplication:
{%highlight julia %}
function mul4d(A::AbstractArray{T, 4}, B::AbstractArray{T, 4}) where T
    C = Array{Float64, 4}(undef, size(A, 1), size(B, 2), size(A, 3), size(A, 4))
    for l in 1:size(A, 4)
        for k in 1:size(A, 3)
            C[:, :, k, l] = A[:, :, k, l] * B[:, :, k, l]
        end
    end
    C
end
{% endhighlight %}

<p>
  <a class="btn" data-toggle="collapse" href="#EulerInterp" role="button" aria-expanded="false" aria-controls="collapseExample">
    Backpropagation for mul4d &#8681;
  </a>
</p>
<div class="collapse" id="EulerInterp">
  <div class="card card-body ">
    If we try getting gradients for <code>mul4d</code> it will not work:
    <pre><code class="language-julia">
    y, pull = Flux.pullback(mul4d, A, B);
    errors = randn(size(y)...);
    grads = pull(errors)
    </code></pre>
    The error is: "Mutating arrays is not supported". So we will have to make an explicit <code>rrule</code> for it.
    <br><br>
    Backpropagation is well known for a linear layer:
    $$
    \frac{\partial L}{\partial W} = \frac{\partial L}{\partial Z} X^T \\
    \frac{\partial L}{\partial X} = W^T \frac{\partial L}{\partial Z}
    $$
    Here are two good derivations using different techniques:
    <a href="https://towardsdatascience.com/deriving-the-backpropagation-equations-from-scratch-part-2-693d4162e779">1</a> 
    (chain rule) and  
    <a href="https://eli.thegreenplace.net/2018/backpropagation-through-a-fully-connected-layer/">2</a>
    (Jacobian).
    If take away the special meanings of weights $W$ and features $X$ we have a general rule for $A\times B$ where $A=W$ and $B=X$.
    We can extend this to 3D and 4D without going through a formal proof by noting that the matrices are independent of each other.
    Hence the equations are:
    $$
    \frac{\partial L}{\partial A}_{ijkl} = \sum_r \frac{\partial L}{\partial Z}_{irkl} B_{jrkl} \\
    \frac{\partial L}{\partial B}_{ijkl} = \sum_r A_{rikl}  \frac{\partial L}{\partial Z}_{rjkl} 
    $$
    Thankfully there already exists a way to get a view of transposes along the first two dimensions in higher order arrays: <code>PermutedDimsArray</code>. So the <code>rrule</code> is relatively short:
    <pre><code class="language-julia">
    import ChainRulesCore.rrule
    using ChainRulesCore: @thunk, NoTangent
    function rrule(::typeof(mul4d), A::AbstractArray{T, 4}, B::AbstractArray{T, 4}) where T
        C = mul4d(A, B)
        function mul4d_pullBack(C̄)
                Ā = @thunk mul4d(C̄, PermutedDimsArray(B, (2, 1, 3, 4)))
                B̄ = @thunk mul4d(PermutedDimsArray(A, (2, 1, 3, 4)), C̄)
            return NoTangent(), Ā, B̄
        end
        return C, mul4d_pullBack
    end
    </code></pre>
    If we try the pullback again it will work now. (You might have to restart and define the rrule before <code>using Flux</code>.)
  </div>
</div>

Unfortunately making an optimised version of this is beyond me. 
But what we can do is extend `batched_mul` by reshaping 4D $m\times n \times p \times q$ arrays into 3D $m\times n \times pq$ arrays:
{%highlight julia %}
import NNlib.batched_mul
function batched_mul(A::AbstractArray{T, 4}, B::AbstractArray{T, 4}) where {T}
    if (size(A, 2) != size(B, 1)) || (size(A, 3) != size(B, 3)) || (size(A, 4) != size(B, 4))
        message = "A has dimensions $(size(A)) but B has dimensions $(size(B))"
        throw(DimensionMismatch(message))
    end
    new_A = reshape(A, size(A, 1), size(A, 2), :)
    new_B = reshape(B, size(B, 1), size(B, 2), :)
    C = batched_mul(new_A, new_B)
    new_C = reshape(C, (size(C, 1), size(C, 2), size(A, 3), size(A, 4)))
    new_C
end
{% endhighlight %}

Doing a time test
{%highlight julia %}
    A = randn(50, 50, 12, 32)
    B = randn(50, 50, 12, 32)
    @time mul4d(A, B);       # 0.016885
    @time batched_mul(A, B); # 0.005216s
{% endhighlight %}
The `batched_mul` version is about 3&times; faster.

We don't need to write a `rrule` for it because rules already exists for `reshape` and `batched_mul`.

[wiki_tensor]: https://en.wikipedia.org/wiki/Tensor


### Multi-head attention

We are finally at the heart of the transformer: multi-head attention.
At the end of this step we will have a `MultiheadAttention` layer:
{% highlight julia %}
MultiheadAttention(num_heads=4, head_size=8, 32=>32)(
    denseQ = Dense(32 => 32),  # 1_056 parameters
    denseK = Dense(32 => 32),  # 1_056 parameters
    denseV = Dense(32 => 32),  # 1_056 parameters
    denseO = Dense(32 => 32),  # 1_056 parameters
)
{% endhighlight %}

The multi-head attention layer splits up the embedding matrix into multiple heads.
Each head will act on an embedding dimension of $d_h$ instead of the full $d_m$ as if the embedding was only $d_h$ in size. 
If we have $H$ heads then $d_m=Hd_h$.

First define a struct to hold all the dense layers and a parameter for $H$ called `nhead`:
{% highlight julia %}
struct MultiheadAttention{Q<:Dense, K<:Dense, V<:Dense, O<:Dense}
    nhead::Int
    denseQ::Q
    denseK::K
    denseV::V
    denseO::O
end

Flux.@functor MultiheadAttention (denseQ, denseK, denseV, denseO, ) # tell Flux which parameters are trainable
{% endhighlight %}

We would like $d_m$ to be divisible by $H$ but the maths will work if it is not.
So if the user supplies $d_h$ accept it as valid: 
{% highlight julia %}
function MultiheadAttention(nhead::Int, dm::Int, dh::Int, dout::Int)
    MultiheadAttention(
        nhead,
        Dense(dm, dh*nhead),
        Dense(dm, dh*nhead),
        Dense(dm, dh*nhead),
        Dense(dh*nhead, dout),
    )
end

function MultiheadAttention(nhead::Int, dm::Int, dout::Int)
    if dm % nhead != 0 
        error("embedding dimension=$dm is not divisible by number of heads=$nhead")
    end
    MultiheadAttention(nhead, dm, div(dm, nhead), dout)
end
{% endhighlight %}

Define utility functions for printing:
{% highlight julia %}
function Base.show(io::IO, mha::MultiheadAttention)
    dh = div(size(mha.denseQ.weight)[1], mha.nhead)
    dm = size(mha.denseQ.weight)[2]
    dout = size(mha.denseO.weight)[1]
    print(io, "MultiheadAttention(")
    print(io, "num_heads=$(mha.nhead), ")
    print(io, "head_size=$(dh), ")
    print(io, "$(dm)=>$(dout)")
    print(io, ")")
end

function Base.show(io::IO, m::MIME"text/plain", mha::MultiheadAttention)
    _show_multiheadattention(io, mha)
end

function _show_multiheadattention(io::IO, mha::MultiheadAttention; indent=0)
    inner_indent = indent + 5
    print(io, " "^indent, mha)
    print(io,"(")
    print(io, "\n")
    Flux._layer_show(io, mha.denseQ, inner_indent, "denseQ")
    Flux._layer_show(io, mha.denseK, inner_indent, "denseK")
    Flux._layer_show(io, mha.denseV, inner_indent, "denseV")
    Flux._layer_show(io, mha.denseO, inner_indent, "denseO")
    print(io, " "^indent, ")")
    if indent==0
        Flux._big_finale(io, mha)
    else 
        println(io, "")
    end
end
{% endhighlight %}

Now let's start with the forward pass. 
We first calculate a query, key and value from the input matrix.
These terms are kind of archaic.
They refer to a database model where the user makes a query (text in a search box), this is mapped to keys (video titles) 
and a value is returned (video). 
Or for a more direct programming metaphor: a hashmap where the query is hashed to a key to retrieve a value.
But the names aren't so important. 
Here we will be using the input matrix as the query, key and value. 
They are each calculated using the dense matrices we stored in the struct:

{% highlight julia %}
function (mha::MultiheadAttention)(query::A1, key::A2, value::A3) where {
    T, A1 <: AbstractArray{T, 3}, A2 <: AbstractArray{T, 3}, A3 <: AbstractArray{T, 3}}
    # batch multiplication version. Input is dm × N × B
    qs = size(query)
    ks = size(key)
    vs = size(value)

    #size(Q) == (dh*nhead, N, B)
    Q = mha.denseQ(query)
    K = mha.denseK(key)
    V = mha.denseV(value)
{% endhighlight %}

In the above [section](#multiplication-with-higher-order-arrays) I went into great detail about handling multiplication for higher order arrays. How does `Dense` handle the 3D input?
{% highlight julia %}
(a::Dense)(x::AbstractArray) = 
  reshape(a(reshape(x, size(x,1), :)), :, size(x)[2:end]...)
{% endhighlight %}
It turns the 3D $d_m \times N \times B$ input into a 2D $d_m \times NB$ matrix, does the multiplication, then transforms it back again.
This solution is valid because the weights for the dense layer are 2D.

We now need to split $Q$, $K$ and $V$ from $d_m \times N \times B$ to $d_h \times N \times H \times B$ matrices.
This is done in two steps: $d_hH \times N \times B \rightarrow d_h \times H \times N \times B \rightarrow d_h \times N \times H \times B$
(break $d_m$ into $d_h$ and $H$ and then swap the 2nd and 3rd dimensions):
{% highlight julia %}
    Q = permutedims(reshape(Q, dh, mha.nhead, qs[2], qs[3]), [1, 3, 2, 4])
    K = permutedims(reshape(K, dh, mha.nhead, ks[2], ks[3]), [1, 3, 2, 4])
    V = permutedims(reshape(V, dh, mha.nhead, vs[2], vs[3]), [1, 3, 2, 4])
{% endhighlight %}

Then we calculate the scaled dot attention for each head, combine results, and pass it through the output dense  layer:
{% highlight julia %}
    A = scaled_dot_attention(Q, K, V)
    A = permutedims(A, [1, 3, 2, 4])
    A = reshape(A, dm, size(A, 3), size(A, 4))
    mha.denseO(A)
end
{% endhighlight %}

Using the `batched_mul` function from the previous section it is now straightforward to calculate attention:[^permutedims]
{% highlight julia %}
function scaled_dot_attention(query::A1, key::A2, value::A3) where {
    T, A1 <: AbstractArray{T, 4}, A2 <: AbstractArray{T, 4}, A3 <: AbstractArray{T, 4}}
    # Batched version. Input is (dh, N, nhead, B)
    dh = size(query, 1)
    keyT = permutedims(key, (2, 1, 3, 4))
    score = one(T)/convert(T, sqrt(dh)) .* batched_mul(keyT, query)
    score = softmax(score; dims=1) #size(score) == (N, N, nhead, B)
    batched_mul(value, score) #size(attention) == (dh, N, nhead, B)
end
{% endhighlight %}
The softmax function (and its rrule) are provided by NNlib.

The one last thing we need to do is make it work for a single embedding instead of a batch.
For code reuse the best solution is to make a single embedding a batch of one:
{% highlight julia %}
function (mha::MultiheadAttention)(query::A1, key::A2, value::A3) where {
    T, A1 <: AbstractMatrix{T}, A2 <: AbstractMatrix{T}, A3 <: AbstractMatrix{T}}
    # single sample version. Input is dm × N
    query = reshape(query, size(query, 1), size(query, 2), 1)
    key   = reshape(key, size(key, 1), size(key, 2), 1)
    value = reshape(value, size(value, 1), size(value, 2), 1)
    A = mha(query, key, value)
    reshape(A, size(A, 1), size(A, 2))
end
{% endhighlight %}

<p>
  <a class="btn" data-toggle="collapse" href="#EulerInterp" role="button" aria-expanded="false" aria-controls="collapseExample">
    Notes on backpropagation in the attention layer &#8681;
  </a>
</p>
<div class="collapse" id="EulerInterp">
  <div class="card card-body ">
    For a 2D matrix $Z=AB$ we have the backwards functions defined as:
    $$
    \frac{\partial L}{\partial A} = \frac{\partial L}{\partial Z} B^T \\
    \frac{\partial L}{\partial B} = A^T \frac{\partial L}{\partial Z}
    $$
    Here we have a case of $Z=A^TB$.  
    This requires finding transposes of these results:
    $$
    \frac{\partial L}{\partial A} = \left(\frac{\partial L}{\partial A^T}\right)^T = B \frac{\partial L}{\partial Z}^T \\
    \frac{\partial L}{\partial B} = (A^T)^T \frac{\partial L}{\partial Z} = A\frac{\partial L}{\partial Z}
    $$
    We don't need to define the rrule because Flux will combine the rules for <code>permutedims</code> and <code>batched_mul</code> to get the same result.
  </div>
</div>

### Encoder blocks

We still need to complete the rest of the equations in the [table](#equations_table).
Thankfully the rest of the layers are provided by Flux. We wrap them in an `TransformerEncoderBlock`:
{% highlight julia %}
struct TransformerEncoderBlock{MA<:MultiheadAttention, L1<:LayerNorm, D1<:Dense, D2<:Dense, L2<:LayerNorm, DO<:Dropout}
    multihead_attention::MA
    layer_norm_attention::L1
    dense1::D1
    dense2::D2
    layer_norm_feedforward::L2
    dropout::DO
end
Flux.@functor TransformerEncoderBlock # make whole TransformerEncoder trainable
{% endhighlight %}

This layer includes drop out regularization which wasn't in the table but it is part of the original paper.
During training this layer randomly sets some weights to zero.
This interferes with training but makes it less likely to overfit.
Have a look at these graphs from the training for the sentiment analysis task:
<figure class="post-figure">
<img class="img-80"
    src="/assets/posts/transformers/dropout.png"
	alt="dropout history"
	>
<figcaption></figcaption>
</figure>
With 10% of dropout there is a much smaller gap between the validation and training accuracies.

[LayerNorm]: https://arxiv.org/abs/1607.06450
Layer norm is a normalization over each layer. 
In the embedding matrix that is each word, so each column in each batch of matrices.
There are other kinds of normalization like batch normalization, which is a normalization across batches.
Interestingly layer norm was only popularised after batch normalization in this [2016 paper][LayerNorm].

The actual function used for layer norm is:

$$
    a_{b}\frac{X_{b}-\mu_{b}}{\sigma_{b}+\epsilon} + b_{b}
$$

For every column $n$ of every batch $b$. This has two parameters in $a_{b}$ and $b_{b}$. 
They are not so important and you can turn them off with `LayerNorm(d, affine=false)`.
$\epsilon$ is a small constant value for numerical stability.

<p>
  <a class="btn" data-toggle="collapse" href="#EulerInterp" role="button" aria-expanded="false" aria-controls="collapseExample">
    Backpropagation for layer norm &#8681;
  </a>
</p>
<div class="collapse" id="EulerInterp">
  <div class="card card-body ">
    This has already been implemented for us. 
    Out of interest, here is a derivation (non-affine):
    $$
    \begin{align}
        \frac{\partial z}{\partial x_i} &= \frac{1}{\sigma + \epsilon}\frac{\partial }{\partial x_i}(x_r -\mu) + (x_r -\mu)\frac{\partial }{\partial x_i}(\sigma + \epsilon)^{-1} \\
                &= \frac{1}{\sigma + \epsilon}(\delta_{ir} - \frac{\partial \mu}{\partial x_i}) - 
                (x_i -\mu)(\sigma + \epsilon)^{-2}\frac{\partial \sigma}{\partial x_i} \\
        \frac{\partial \mu}{\partial x_i} &= \frac{\partial}{\partial x_i} \frac{1}{d}\sum_k^d x_k \\
                            &= 0 + ... + 0 + \frac{1}{d} + 0 + ... + 0 \\
                            &= \frac{1}{d} \\
        \frac{\partial \sigma}{\partial x_i} &= \frac{\partial}{\partial x_i} \sqrt{\frac{1}{d}\sum^d_k (x_k - \mu)^2} \\
                            &= \frac{1}{d\sigma}\left((x_i - \mu)(1 - \frac{\partial \mu}{\partial x_i}) +
                            \sum^d_{k\neq i}(x_k - \mu)(0 - \frac{\partial \mu}{\partial x_i}) \right) \\
                            &= \frac{x_i -\mu}{d\sigma} + \frac{1}{d\sigma}\frac{\partial \mu}{\partial x_i}\left(\sum^d_k x_k -  \mu \sum^d_k 1 \right) \\
                            &= \frac{x_i -\mu}{d\sigma} + \frac{1}{d^2\sigma}(\mu d -  \mu d) \\
                            &= \frac{x_i -\mu}{d\sigma} \\
    \end{align}
    $$
  </div>
</div>

Because the inputs and outputs are similar we only need four parameters to define the whole block:
{% highlight julia %}
TransformerEncoderBlock(nhead::Int, dm::Int, dhid::Int; pdrop::Float64=0.1) = TransformerEncoderBlock(
    MultiheadAttention(nhead, dm, dm),
    LayerNorm(dm),
    Dense(dm, dhid, relu),
    Dense(dhid, dm),
    LayerNorm(dm),
    Dropout(pdrop)
)
{% endhighlight %}

Printing functions:
{% highlight julia %}
function Base.show(io::IO, te::TransformerEncoderBlock)
    print(io, "TransformerEncoderBlock(")
    print(io, te.multihead_attention)
    print(io, ", ", te.layer_norm_attention)
    print(io, ", ", te.dense1)
    print(io, ", ", te.dense2)
    print(io, ", ", te.layer_norm_feedforward)
    print(io, ")")
end

function Base.show(io::IO, m::MIME"text/plain", te::TransformerEncoderBlock)
    _show_transformer_encoder(io, te)
end

function _show_transformer_encoder(io::IO, t::TransformerEncoderBlock; indent=0)
    inner_indent = indent + 5
    print(io, " "^indent, "TransformerEncoderBlock")
    print(io, "(")
    print(io, "\n")
    _show_multiheadattention(io, t.multihead_attention, indent=inner_indent)
    Flux._layer_show(io, t.dropout, inner_indent)
    Flux._layer_show(io, t.layer_norm_attention, inner_indent)
    Flux._layer_show(io, t.dense1, inner_indent)
    Flux._layer_show(io, t.dense2, inner_indent)
    Flux._layer_show(io, t.dropout, inner_indent)
    Flux._layer_show(io, t.layer_norm_attention, inner_indent)
    print(io, " "^indent, ")")
    if indent==0
        Flux._big_finale(io, t)
    else
        println(io, "")
    end
end
{% endhighlight %}

Lastly, the forward pass:
{% highlight julia %}
function (t::TransformerEncoderBlock)(x::A) where {T, N, A<:AbstractArray{T, N}}
    a = t.multihead_attention(x, x, x)
    a = t.dropout(a)
    res_a = x + a # skip connection
    res_a = t.layer_norm_attention(res_a)
    z_ff = t.dense1(res_a)
    z_ff = t.dense2(z_ff)
    z_ff = t.dropout(z_ff)
    res_ff = res_a + z_ff # skip connection
    res_ff = t.layer_norm_feedforward(res_ff)
    res_ff
end
{% endhighlight %}

Skip connections are short-circuits.
They look like they are undoing all the hard work of the previous layer.
However these have proved very useful for neural networks with many layers
because they carry a strong signal both on the forward pass and with the gradient on the backwards pass.

### Classifier

At last, our model is almost ready for use.
There is just one last question, how to use the output embedding matrix?
We could take a mean across each word embedding and then pass that to a dense layer.
Or we can take a dense layer across each word, reduce it down to one dimension, and pass that to a dense layer.
Or we could flatten the whole array into a $dm N \times 1$ column. 

My preference is to do an aggregation on each word first and then on the sentence.
Here is a simple flatten layer which we will need to put in between:
{% highlight julia %}
struct FlattenLayer end

Flux.@functor FlattenLayer

function (f::FlattenLayer)(x::AbstractArray{T, 3}) where T
  reshape(x, :, size(x, 3)) # same as Flux.flatten
end

function (f::FlattenLayer)(x::AbstractArray{T, 2}) where T
    reshape(x, :, 1) # returns a column vector
end

function Base.show(io::IO, f::FlattenLayer)
  print(io, "FlattenLayer()")
end
{% endhighlight %}

We can now make our model with Flux chain:

{% highlight julia %}
dim_embedding = 32
pdrop = 0.1
add_position_encoding(x) = x .+ position_encoding(x)
model = Chain(
    Embed(dim_embedding, length(indexer)), 
    add_position_encoding, # can also make anonymous
    Dropout(pdrop),
    TransformerEncoderBlock(4, dim_embedding, dim_embedding * 4; pdrop=pdrop),
    Dense(dim_embedding, 1),
    TransformersLite.FlattenLayer(),
    Dense(max_length, nlabels)
     )
{% endhighlight %}

Or here is the whole model wrapped in a struct with nicer printing and names:
{% highlight julia %}
struct TransformerClassifier{
    E<:Embed, 
    PE<:PositionEncoding, 
    DO<:Dropout, 
    TEB<:Vector{TransformerEncoderBlock}, 
    A, 
    f<:FlattenLayer, 
    D<:Dense
    }
    embed::E
    position_encoding::PE
    dropout::DO
    encoder_layers::TEB
    agg_layer::A
    flatten_layer::f
    classifier::D
end

Flux.@functor TransformerClassifier

function Base.show(io::IO, m::MIME"text/plain", t::TransformerClassifier)
    _show_transformer_classifier(io, t)
end

function _show_transformer_classifier(io::IO, t::TransformerClassifier; indent=0)
    inner_indent = 5
    print(io, " "^indent, "TransformerClassifier")
    print(io, "(")
    print(io, "\n")
    Flux._layer_show(io, t.embed, inner_indent)
    Flux._layer_show(io, t.position_encoding, inner_indent)
    Flux._layer_show(io, t.dropout, inner_indent)
    for e in t.encoder_layers
        _show_transformer_encoder(io, e, indent=inner_indent)
    end
    Flux._layer_show(io, t.agg_layer, inner_indent)
    Flux._layer_show(io, t.flatten_layer, inner_indent)
    Flux._layer_show(io, t.classifier, inner_indent)
    print(io, " "^indent, ")")
    if indent==0
        Flux._big_finale(io, t)
    end
end

function (t::TransformerClassifier)(x::A) where {T, N, A<:AbstractArray{T, N}}
    x = t.embed(x)
    x = x .+ t.position_encoding(x)
    x = t.dropout(x)
    for e in t.encoder_layers
        x = e(x)
    end
    x = t.agg_layer(x)
    x = t.flatten_layer(x)
    x = t.classifier(x)
    x
end
{% endhighlight %}

An example of a small model:
{% highlight julia %}
model = TransformersLite.TransformerClassifier(
    Embed(dim_embedding, length(indexer)), 
    PositionEncoding(dim_embedding), 
    Dropout(pdrop),
    TransformerEncoderBlock[
        TransformerEncoderBlock(4, dim_embedding, dim_embedding * 4; pdrop=pdrop)
    ],
    Dense(dim_embedding, 1), 
    FlattenLayer(),
    Dense(max_length, nlabels)
    )
{% endhighlight %}

Finally, we have a working transformer!
## Use case: Amazon Reviews 

## Conclusion

Thank you for following me through this very long post.
I hope you now understand transformers.

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

[^linear_pe]: The position encoding of $\frac{j}{n_{max}}$ is also linear for constant $k$:
    $$ \frac{j+k}{n_{max}}=\frac{1}{n_{max}}j+\frac{k}{n_{max}}$$

[^permutedims]: One may think that it is better to use `PermutedDimsArray` because it provides a view instead of allocating a new array like `permutedims`. In practice the `reshape` in `batched_mul` creates a `ReshapedArray{PermutedDimsArray{Array}}` which the compiler struggles to optimise for, greatly slowing the batched multiplication. So it is better to take a smaller performance hit here with allocating a new array. The `reshape` then simply returns `Array`.