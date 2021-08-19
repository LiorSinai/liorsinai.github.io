---
layout: post
title:  "Sudoku OCR reader in Julia: part 4"
date:   2021-08-10
author: Lior Sinai
categories: coding
tags:	'machine learning'
---

_On convolutional neural networks, overly large models and the importance of understanding your data._ 

This post is part of a series. The other articles are:
- [Part 1: introduction][introduction].
- [Part 2: grid extraction][grid_extraction].
- [Part 3: digit extraction][digit_extraction].
- [Part 5: conclusion][conclusion].

[introduction]: {{ "coding/2021/08/10/sudoku-reader-part1-intro" | relative_url }}
[grid_extraction]: {{ "coding/2021/08/10/sudoku-reader-part2-grid" | relative_url }}
[digit_extraction]: {{ "coding/2021/08/10/sudoku-reader-part3-digits" | relative_url }}
[machine_learning]: {{ "coding/2021/08/10/sudoku-reader-part4-cnn" | relative_url }}
[conclusion]: {{ "coding/2021/08/10/sudoku-reader-part5" | relative_url }}

All code is available online at my repository: [github.com/LiorSinai/SudokuReader.jl](https://github.com/LiorSinai/SudokuReader.jl).

# Part 4 - machine learning
## A mistaken assumption

When I initially envisioned this blog post, I thought I wouldn't have much to talk about with regards to machine learning.
Digit recognition is the entry level problem of machine learning. It is incorporated into almost every beginner course on the topic. 
My assumption was you as the reader have done this before and I would only need to present the code. 
The model would be copied off the first tutorial I found on the internet and would be trained on the MNIST dataset. But I found MNIST ill-suited to the task at hand and the models subpar. In particular:
1. The handwritten MNIST dataset resulted in many false positives with computer font digits. This was most pronounced between 1s, 2s and 7s and between 6s and 0s.
2. The models were unnecessarily large. I used a 44,000 parameter model to solve this problem. Some other models had over 1 million parameters.


I'll now delve further into each of these issues.

## Data

[MNIST]: https://www.kaggle.com/c/digit-recognizer
[LeNet5]: http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
[Char74K]: http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/

The [Modified National Institute of Standards and Technology database (MNIST)][MNIST] is a large set of 70,000 labelled handwritten digits. Here is a sample of the digits:[^7s][^zero]

<figure class="post-figure">
<img class="img-95"
    src="/assets/posts/sudoku-reader/mnist_sample.png"
	alt="mnist_sample"
	>
<figcaption>MNIST dataset.</figcaption>
</figure>

I trained my model and then discovered very quickly that despite a 99% test accuracy, it was not the best dataset for a model that will mostly be aimed at computer printed digits. To understand why, let us look at the [Char74K][Char74K] dataset which is a based off computer fonts:

<figure class="post-figure">
<img class="img-95"
    src="/assets/posts/sudoku-reader/char74k_sample.png"
	alt="char74k_sample"
	>
<figcaption>Char74k numbers.</figcaption>
</figure>

Here is a general comparison of the mean image of each digit for each dataset:

<figure class="post-figure">
<img class="img-95"
    src="/assets/posts/sudoku-reader/digit_means.png"
	alt="digit_means"
	>
<figcaption>Digit means. Top: MNIST. Bottom: Char74k</figcaption>
</figure>

The computer 1s tend to have a hook that handwritten 1s don't have. The handwritten 4s are open at the top whereas computer 4s are closed. The circle in handwritten 6s tends to be smaller than the circle in computer 6s.
All these small differences make a big impact. 

As an experiment, I trained the LeNet5 model (detailed below) on the MNIST dataset and evaluated it on the Char74k dataset. While it had a 98.7% test accuracy for MNIST data, this only translated to a 81.3% accuracy on the whole Char74k dataset. So these datasets are strongly correlated but that is not enough for the required task. Here is the confusion matrix for the Char74k dataset:

<figure class="post-figure">
<img class="img-95"
    src="/assets/posts/sudoku-reader/confusion_matrix.png"
	alt="confusion_matrix"
	>
<figcaption>Confusion matrix for a model trained on MNIST and evaluated on Char74k.</figcaption>
</figure>

As expected, 7, 1 and 2 are often confused. So was 6s with 5s and - to a less a extent - 6s with 0s. 
This sort of confusion also appeared with the few Sudoku images I tried.
Training the model on the Char74k removed this problem.

Going the other way, the a model trained on the Char74k dataset to a 99.1% test accuracy only achieved a 54.1% accuracy on the MNIST data. I found this acceptable because the target is computer fonts not handwritten digits. Its overall performance with the Sudoku images was much better.

Another strategy is to train on both: 10,000 Char74k figures and 10,000 MNIST figures. 
The model has a test accuracy of 98.1%. For the separate datasets (train+test) it is 99.4% on the Char74k data and 98.8% on the MNIST data.
A flaw of this model is that 42 times it confused 2s for 7s. (Out of 4000 2s and 7s that is acceptable.) Otherwise the second largest value in the confusion matrix (off diagonal) was 12. 
Overall, this model seems to perform slighlty worse on the Sudoku images.

## Models

I eventually used the [LeNet5][LeNet5] model originally published in 1998. Its architecture is as follows:
<figure class="post-figure">
<img class="img-95"
    src="/assets/posts/sudoku-reader/LeNet5.png"
	alt="LeNet5"
	>
<figcaption>LeNet5, illustrated by Barnabás Póczos.</figcaption>
</figure>
It is a convolutional neural network. If you're unfamiliar with this type of neural network, please see this [article](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53) for a guide on how they work.

In Julia code, LeNet5 is built as follows:[^flatten]
{% highlight Julia %}
model = Chain(
	Conv((5, 5), 1=>6, relu),
	MaxPool((2, 2)),
	Conv((5, 5), 6=>16, relu),
	MaxPool((2, 2)),
	Flux.flatten,
	Dense(256, 120, relu), 
	Dense(120, 84, relu), 
	Dense(84, 10),
	)
{% endhighlight %}


This was not the first model I encountered. 
Instead that was huge a 1.2 million parameter model on a blog post.
It worked, but I thought it is unnecesary large and slow. It is built as follows:
{% highlight Julia %}
model = Chain(
	Conv((3, 3), 1=>32, stride=1, pad=0, relu),
	Conv((3, 3), 32=>64, stride=1, pad=0, relu),
	MaxPool((2, 2)),
	Dropout(0.25),
	Flux.flatten,
	Dense(9216, 128),
	Dropout(0.5),
	Dense(128, 10),
    )
{% endhighlight %}
This model is similar to LeNet5 but it has a few key differences. Firstly, it does 2&times; the amount of convolutions per `Conv` step. Secondly, and very importantly, it only down samples once with a `MaxPool`. 
Down sampling again by a factor 2 in both directions would have reduced the total pixel size by 4, which in turn reduces
the parameter space by approximately 4. So 225,034 parameters instead of over a million.

I then investigated a few other models.
Here is a summary of my findings for models trained only on the Char74k dataset:
<table>
<thead>
  <tr>
    <th></th>
    <th>name</th>
    <th>train accuracy (%)</th>
    <th>test accuracy (%)</th>
    <th>inference time (ms)</th>
    <th>file size (kB)</th>
    <th>parameters</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>1</td>
    <td>cnn_mastery</td>
    <td>99.99</td>
    <td>98.77</td>
    <td>0.1547</td>
    <td>2119.68</td>
    <td>542230</td>
  </tr>
  <tr>
    <td>2</td>
    <td>LeNet5</td>
    <td>99.51</td>
    <td>98.67</td>
    <td>0.0806</td>
    <td>183.0</td>
    <td>44426</td>
  </tr>
  <tr>
    <td>3</td>
    <td>cnn_medium</td>
    <td>99.37</td>
    <td>98.67</td>
    <td>0.1355</td>
    <td>79.4</td>
    <td>18378</td>
  </tr>
  <tr>
    <td>4</td>
    <td>cnn_huge</td>
    <td>99.26</td>
    <td>98.13</td>
    <td>0.8164</td>
    <td>4689.92</td>
    <td>1199882</td>
  </tr>
  <tr>
    <td>5</td>
    <td>cnn_small</td>
    <td>98.62</td>
    <td>97.74</td>
    <td>0.0576</td>
    <td>27.7</td>
    <td>5142</td>
  </tr>
  <tr>
    <td>6</td>
    <td>nn_fc</td>
    <td>98.43</td>
    <td>96.75</td>
    <td>0.0023</td>
    <td>103.0</td>
    <td>25450</td>
  </tr>
</tbody>
</table>

Th cnn_huge model comes in 4th place despite its large amount of parameters. It is 10 times slower and 25.6 times larger than LeNet5. You may notice it still only has an inference time of 0.8ms and a file size of 4.6MB. It took 10 minutes to train as opposed to 2 minutes for LeNet5. These numbers are still small, so what is the fuss? I just greatly dislike increasing complexity and reducing performance for no reason.

The fastest model is nn_fc. It only has fully connected layers with no convolutions. Here is its architecture:
{% highlight Julia %}
model = Chain(
        Flux.flatten, 
        Dense(784, 32, sigmoid),
        Dense(32, 10),
    )
{% endhighlight %}
This model is prone to errors because it does not take the structure of the image to account. So very weirdly, you can swap rows and columns and not affect the prediction. Clearly this limits the overall accuracy of the model.

LeNet5 is a good balance between slow convolutions with few parameters and large but fast fully connected layers.
I still think it is too large - the smallest model in the table is 1/8th its size. 
But for 183kB against 28kB, I think it's a reasonable tradeoff for a 1% increase in accuracy.


## The code

I followed the tutorials of [Nigel Adams][Adams] and [Clark Fitzgerald][Fitzgerald].
Julia is a young language and is still short on resources and tutorials for Flux, so I am glad these two ventured into the unknown early with their articles.

[Adams]: https://spcman.github.io/getting-to-know-julia/deep-learning/vision/flux-cnn-zoo/
[Fitzgerald]: http://webpages.csus.edu/fitzgerald/julia-convolutional-neural-network-MNIST-explained/

I used the digits from the [Char74k][Char74k] dataset, specifically Sample001-Sample010 in EnglishFnt.tgz.
It is useful to convert the Char74k dataset to the MNIST format so the same model can be used for both datasets.
This script will do that (be sure to have the correct `inpath` for your data):
{% highlight Julia %}
using FileIO
using Images

inpath = "..\\..\\datasets\\74k_numbers"
outpath = inpath *"_28x28"
include("..\\utilities\\invert_image.jl");

if !isdir(outpath)
    mkdir(outpath)
end

for indir in readdir(inpath)
    println("working in $(joinpath(inpath, indir))")
    outdir = joinpath(outpath, string(parse(Int, indir[(end-1):end]) - 1))
    if !isdir(outdir)
        mkdir(outdir)
    end
    num_saved = 0
    for filename in readdir(joinpath(inpath, indir))
        num_saved += 1
        image = load(joinpath(inpath, indir, filename))
        image = imresize(image, (28, 28))
        image = invert_image(image)
        save(joinpath(outdir, filename), image) 
    end
    println("saved $num_saved files to $outdir")
end
{% endhighlight %}

Now we can look into the training.
Here are all the imports we will need:
{% highlight Julia %}
using Flux
using Flux: Data.DataLoader, unsqueeze
using Flux: onehotbatch, onecold, logitcrossentropy
using BSON # for saving models

using StatsBase: mean
using Random

using Printf
{% endhighlight %}

I also wrote a few helper functions at [ml_utils.jl](https://github.com/LiorSinai/SudokuReader.jl/blob/main/DigitDetection/ml_utils.jl). For example, here is the function to load the data:
{% highlight Julia %}
function load_data(inpath)
    # data is images within a folder with name inpath/label/filename.png
    digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    data = Array{Float32, 3}[] # Flux expects Float32 only, else raises a warning
    labels = Int[]
    for digit in digits
        indir = joinpath(inpath, string(digit))
        println("loading data from $indir")
        for filename in readdir(indir)
            image = load(joinpath(inpath, string(digit), filename))
            image = Flux.unsqueeze(Float32.(image), 3)
            push!(data, image)
            push!(labels, digit)
        end
    end
    data, labels
end
{% endhighlight %}
An important caveat is that the data should be of type `Float32`. This does noticeably increase the speed of training the model.

Next split the data into train, validation and test sets.
Here is the `split_data` function:
{% highlight Julia %}
function split_data(X, y; rng=Random.GLOBAL_RNG, test_split=0.2)
    n = length(X)
    n_train = n - round(Int, test_split * n)
    
    idxs = collect(1:n)
    randperm!(rng, idxs)

    X_ = X[idxs]
    y_ = y[idxs]

    x_train = X_[1:n_train]
    y_train = y_[1:n_train]
    x_test  = X_[n_train+1:end]
    y_test  = y_[n_train+1:end]

    x_train, y_train, x_test, y_test
end
{% endhighlight %}

It's best to fix the seed at an arbitrary number so that the data is always "randomly" sorted the same way. 
This makes comparisons between training runs consistent.
{% highlight Julia %}
seed = 227
rng = MersenneTwister(seed)
x_train, y_train, x_test, y_test = split_data(data, labels, rng=rng);
{% endhighlight %}

Now we need to get the data into the required format for Flux.
The sample data needs to be in the form height&times;width&times;channels&times;batch_size.
To prevent using extra space, we can use `Flux.DataLoader` which lazily batches the data. This means it doesn't create a copy of the data (unlike `split_data`) when allocating a batch for a small gradient descent step.
The test data and validation data however are all evaluated at once, and are much smaller, so I eagerly batch them with `Flux.batch`.  This does create duplicates.
{% highlight Julia %}
train_data = Flux.DataLoader((Flux.batch(x_train), y_train), batchsize=128)
n_valid = floor(Int, 0.8*size(y_test, 2))
valid_data = (Flux.batch(x_test[1:n_valid]), y_test[:, 1:n_valid])
test_data = (Flux.batch(x_test[n_valid+1:end]), y_test[:, n_valid+1:end])
{% endhighlight %}

The labels need to be one hot matrices for the loss function:
{% highlight Julia %}
y_train = onehotbatch(y_train, 0:9)
y_test =  onehotbatch(y_test, 0:9)
{% endhighlight %}

Now we can load the LeNet5 model from before:
{% highlight Julia %}
function LeNet5()
	return Chain(
            Conv((5, 5), 1=>6, relu),
            MaxPool((2, 2)),
            Conv((5, 5), 6=>16, relu),
            MaxPool((2, 2)),
            Flux.flatten,
            Dense(256, 120, relu), 
            Dense(120, 84, relu), 
            Dense(84, 10),
          )
end
model = LeNet5()
{% endhighlight %}

<p>
  <a class="btn" data-toggle="collapse" href="#LayerSizes" role="button" aria-expanded="false" aria-controls="collapseExample">
    Calculating layer sizes &#8681;
  </a>
</p>
<div class="collapse" id="LayerSizes">
  <div class="card card-body ">
<p>
Flux assumes you know the input and output sizes for the dense layers.
Therefore we need to know the output size of the last convolution layer.
This can be caluated with the following formula:
$$ \left\lfloor \frac{i+2p-k}{s} \right\rfloor + 1 $$
Where $i$ is the input size, $p$ is the pad, $k$ is the kernel (filter) size and $s$ is the stride length.
For a very thorough explanation, please see this paper: 
<a href="https://arxiv.org/abs/1603.07285">A guide to convolution arithmetic for deep learning</a>.
</p>

<p>In Julia code:
{% highlight julia %}
function calc_output_size(input_size::Int, filter_size::Int, stride::Int=1, pad::Int=0)
	floor(Int, (input_size + 2pad - filter_size)/stride) + 1
end
{% endhighlight %}
Apply this repeatedly to get the final output dimension size.
For example, for LeNet5:
{% highlight julia %}
output_dim = 28
output_dim = calc_output_size(output_dim, 5, 1, 0)
output_dim = calc_output_size(output_dim, 2, 2, 0)
output_dim = calc_output_size(output_dim, 5, 1, 0)
output_dim = calc_output_size(output_dim, 2, 2, 0)
output_size = prod((output_dim, output_dim, 16)) # prod((4, 4, 16))=256
{% endhighlight %}
</p>
  </div>
</div>

We then need to define an accuracy function and a loss function:
{% highlight Julia %}
accuracy(ŷ, y) = mean(onecold(ŷ, 0:9) .== onecold(y, 0:9))
loss(x::Tuple) = Flux.logitcrossentropy(model(x[1]), x[2])
loss(x, y) = Flux.logitcrossentropy(model(x), y)
opt=ADAM()
{% endhighlight %}
The `Flux.logitcrossentropy` calculates the following:

$$ \frac{\sum y \cdot log(\sigma(\hat{y}))}{n}; \;\; \sigma(y_i)=\frac{e^{y_i}}{\sum e^y} $$

The logits are the direct outputs of the model, which can be any real number. 
The softmax function $\sigma$ converts them to values between to 0 and 1 which sum to 1.
That is, it converts them to a probability distribution.
[ADAM][ADAM_gd] is a well known and stable gradient descent algorithm that works well without much fine tuning.

[ADAM_gd]: https://arxiv.org/abs/1412.6980

With these function in place, we can simply call `Flux.train!(loss, params(model), train_data, opt)` and wait for our model to train. But I wanted more. I wanted to create a history of the change in accuracy during training and return it. I wanted to have a print out each time a batch was completed. I wanted to save after every epoch, where an epoch is one full run through the entire training set. 
So I copied the definition for `Flux.train!` and edited it as follows:
{% highlight julia %}
function train!(loss, ps, train_data, opt, acc, valid_data; n_epochs=100)
    history = Dict("train_acc"=>Float64[], "valid_acc"=>Float64[])
    for e in 1:n_epochs
        print("$e ")
        ps = Flux.Params(ps)
        for batch_ in train_data
            gs = gradient(ps) do
                loss(batch_...)
            end
            Flux.update!(opt, ps, gs)
            print('.')
        end
        # update history
        train_acc = 0.0
        n_samples = 0
        for batch_ in train_data
            train_acc += sum(onecold(model(batch_[1])) .== onecold(batch_[2]))
            n_samples += size(batch_[1], 4)
        end
        train_acc = train_acc/n_samples
        valid_acc = acc(model(valid_data[1]), valid_data[2])
        push!(history["train_acc"], train_acc)
        push!(history["valid_acc"], valid_acc)

        @printf "\ntrain_acc=%.4f valid_acc=%.4f\n" train_acc*100 valid_acc*100

        # save model
        save_path = output_path * "_e$e" * ".bson"
        BSON.@save save_path model history
    end
    history
end
start_time = time_ns()
history = train!(
    loss, params(model), train_data, opt, 
    accuracy, valid_data, n_epochs=20
    )
end_time = time_ns() - start_time
println("done training")
@printf "time taken: %.2fs\n" end_time/1e9
{% endhighlight %}

Here is an example of the training history for LeNet5 trained on the Char74k dataset:
<figure class="post-figure">
<img class="img-95"
    src="/assets/posts/sudoku-reader/history.png"
	alt="training history"
	>
<figcaption>training history</figcaption>
</figure>

Finally we calculate the final test accuracy on the small test data we had left behind:
{% highlight julia %}
test_acc = accuracy(model(test_data[1]), test_data[2])
@printf "test accuracy for %d samples: %.4f\n" size(test_data[2], 2) test_acc 
{% endhighlight %}

## Next section

This was long section with unexpected detours. I hope you enjoyed it.
We can now go to the final section where we can see the results of our model: [part_5][conclusion].

---

[^7s]: The 7s and 9s in the MNIST data are lowered with respect to the top of the image. I presume this is because when this data was processed the centroid was used as the centre of image. I believe a better choice would have been to use the centre of the bounding box as the centre of the image. This is what the digit extraction does in part 3.

[^zero]: One can argue that you don't need to train with zero because there should never be a zero in Sudoku. For the user experience I'd argue it is worth the marginal effort. If the user does have a zero in their image, for whatever reason, it will make the model appear very incompotent if it cannot classify zeros. Anyway, this is your personal preference.

[^flatten]: Flatten is exported by both DataFrames and Flux and therefore needs to be qualified.