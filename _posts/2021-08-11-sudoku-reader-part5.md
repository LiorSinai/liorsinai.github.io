---
layout: post
title:  "Sudoku OCR reader in Julia: part 5"
date:   2021-08-10
author: Lior Sinai
categories: coding
tags:	'machine learning'
---

_Wrapping up the Sudoku OCR reader series._ 

This post is part of a series. The other articles are:
- [Part 1: introduction][introduction].
- [Part 2: grid extraction][grid_extraction].
- [Part 3: digit extraction][digit_extraction].
- [Part 4: machine learning][machine_learning].

[introduction]: {{ "coding/2021/08/10/sudoku-reader-part1-intro" | relative_url }}
[grid_extraction]: {{ "coding/2021/08/10/sudoku-reader-part2-grid" | relative_url }}
[digit_extraction]: {{ "coding/2021/08/10/sudoku-reader-part3-digits" | relative_url }}
[machine_learning]: {{ "coding/2021/08/10/sudoku-reader-part4-cnn" | relative_url }}
[conclusion]: {{ "coding/2021/08/10/sudoku-reader-part5" | relative_url }}

All code is available online at my repository: [github.com/LiorSinai/SudokuReader-Julia](https://github.com/LiorSinai/SudokuReader-Julia).

# Part 5

Thank you for following along until now. This final part is split into the following sections:
- [Integrating code from the previous parts](#integrating-code).
- [Presenting the result](#presenting-the-result).
- [Reflection on the approach](#reflection).

## Integrating code

First the required imports:
{% highlight julia %}
using Images
using Plots
using FileIO
using Flux
using BSON

include("GridDetection/GridDetection.jl")
using .GridDetection
include("DigitDetection/DigitExtraction.jl")
using .DigitExtration
include("utilities/Transforms.jl")
using .Transforms
{% endhighlight %}

Now that we have all the pieces assembled, we can pass the outputs from one part as the input to the next:
{% highlight julia %}
image_path = "images/nytimes_20210807.jpg";
image = load(image_path)

blackwhite, quad = detect_grid(
    image; 
    max_size=1024, 
    blur_window_size=5, σ=1.1, 
    threshold_window_size=15, threshold_percentage=7);
warped, invM = four_point_transform(blackwhite, quad)

BSON.@load "DigitDetection\\models\\LeNet5_both_e20.bson" model
grid, centres, probs = read_digits(
    warped, model,
    offset_ratio=0.1, 
    radius_ratio=0.25, 
    detection_threshold=0.1
    );
{% endhighlight %}

`read_digits` uses a function called `prediction`. It provides a wrapper around the output of the model, which are logits.
The softmax probability is a useful proxy for how confident the model is in its prediction. On the training data, the confidence for correct predictions is 100%.

{% highlight julia %}
using Flux: softmax, batch, unsqueeze
using Images: imresize

function prediction(model, image::AbstractArray, pad_ratio=0.2)
    image = pad_image(image, pad_ratio=pad_ratio)
    image = imresize(image, (28, 28))
    x = batch([unsqueeze(Float32.(image), 3)])
    logits = model(x)
    probabilities = softmax(logits)
    idx = argmax(probabilities)
    ŷ = idx[1] - 1
    ŷ, probabilities[idx]
end

function pad_image(image::AbstractArray{T}; pad_ratio=0.2) where T
    height, width = size(image)
    pad = floor(Int, pad_ratio * max(height, width))
    imnew = zeros(T, (height + 2pad, width + 2pad))
    imnew[(pad + 1):(pad + height), (pad + 1):(pad + width)] = image
    imnew
end
{% endhighlight %}

## Presenting the result

The output of `read_digits` is three 9&times;9 matrices: grid, centres and probabilities.
The grid has the numbers, the centres has the co-ordinates of the centres of the bounding boxes in the warped image, and the probabilities has the maximum probability. The latter are zero if no prediction was made.

Drawing text over the original numbers is easy if we use Plots.jl. We will need the `perspective_transform` function from [part 3][digit_extraction] to unwarp the centres back to their positions in the original image.
{% highlight julia %}
pred_threshold = 0.90
image_out = imresize(image, size(blackwhite));
canvas = plot(image_out, ticks=nothing, border=:none, size=(800, 600));
for i in 1:9
    for j in 1:9
        centre = centres[i, j]
        centre_unwarped = perspective_transform(invM)(centre)
        label =  (probs[i, j] > pred_threshold) ? string(grid[i, j]) : "·"
        annotate!(canvas, centre_unwarped[2], centre_unwarped[1], label, :yellow)
    end
end
{% endhighlight %}

Here is the result:
<figure class="post-figure">
<img class="img-95"
    src="/assets/posts/sudoku-reader/6_out.png"
	alt="6_out"
	>
<figcaption></figcaption>
</figure>

There are two things we can do which greatly improve the presentation:
1. Project the grid back on to the original image.
2. Align the points which don't have a prediction.

First a very basic function for making lines which form a grid:
{% highlight julia %}
function construct_grid(height::Int, width::Int; nblocks::Int=3)
    grid = []
    step_i = height/nblocks
    step_j = width/nblocks
    for i in 0:nblocks
        push!(grid, [(step_i * i, 1), (step_i * i, width)])
    end
    for j in 0:nblocks
        push!(grid, [(1, step_j * j), (height, step_j * j)])
    end
    grid
end
{% endhighlight %}

Then here is a loop for projecting those lines onto the original image:
{% highlight julia %}
for line in construct_grid(size(warped, 1), size(warped, 2))
    line_unwarped = map(point -> perspective_transform(invM)(point), line)
    xs = [point[2] for point in line_unwarped]
    ys = [point[1] for point in line_unwarped]
    plot!(canvas, xs, ys, label="", linewidth=2, color=:yellow)
end
{% endhighlight %}

Next the `align_centres` function. We can use the mean of the co-ordinates of the numbers above and below a point to get its $x$ value, and similarly for numbers to the left and right of it for the $y$ value:
{% highlight julia %}
function align_centres(centres::Matrix, guides::BitMatrix)
    centres_aligned = copy(centres)
    if size(centres) != size(guides)
         throw("$(size(centres)) != $(size(guides)), sizes of centres and guides must be the same.")
    end
    for i in 1:size(centres, 1)
        for j in 1:size(centres, 2)
            if !guides[i, j]
                # y is common to row i
                if any(guides[i, :])
                    ys = [point[1] for point in centres[i, :]] .* guides[i, :]
                    Cy = sum(ys) / count(guides[i, :])
                else
                    Cy = centres[i, j][1]
                end
                #  x is common to column j
                if any(guides[:, j])
                    xs = [point[2] for point in centres[:, j]] .* guides[:, j]
                    Cx = sum(xs) / count(guides[:, j])
                else 
                    Cx = centres[i, j][2]
                end
                centres_aligned[i, j] = (Cy, Cx)
            end
        end
    end
    centres_aligned
end
{% endhighlight %}

Applying these two functions makes the result look much more professional:
<figure class="post-figure">
<img class="img-95"
    src="/assets/posts/sudoku-reader/nytimes_20210807_read.png"
	alt="nytimes_20210807_read"
	>
<figcaption></figcaption>
</figure>

The final step is to pass the grid into a Sudoku solver, get those numbers back, and project them on to the grid.
But I'll stop here 🙂. 

## Reflection

This application used several algorithms, some rather complex, to do a task that humans consider trivial. 
This is not to downplay the effort. The task is a complex one, and we only consider it trivial because our brains have exquisitely adapted to it.

We've used several algorithms along the way. It is worth taking stock of all of them and all the parameters that are needed. Some of these parameters are fixed, whether set explicitly or implied. For example, the blurring is done the same in the horizontal and vertical directions and so one parameter is fixed. 
Others are free and may require hand tuning.
Here is a table with an overview of all fixed and free parameters:[^LeNet5]

<table>
<thead>
  <tr>
    <th>Step</th>
    <th>Algorithm</th>
    <th>Fixed parameters</th>
    <th>Free parameters</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="3">preprocess</td>
    <td>imresize</td>
    <td>0</td>
    <td>1</td>
  </tr>
  <tr>
    <td>Guassian Blur</td>
    <td>2</td>
    <td>2</td>
  </tr>
  <tr>
    <td>AdaptiveThreshold</td>
    <td>1</td>
    <td>2</td>
  </tr>
  <tr>
    <td>detect_grid</td>
    <td>find_contours</td>
    <td>0</td>
    <td>0</td>
  </tr>
  <tr>
    <td rowspan="4">extract_digits</td>
    <td>warp</td>
    <td>8</td>
    <td>0</td>
  </tr>
  <tr>
    <td>read_digits</td>
    <td>1</td>
    <td>1</td>
  </tr>
  <tr>
    <td>detect_in_centre</td>
    <td>0</td>
    <td>2</td>
  </tr>
  <tr>
    <td>extract_digit (label_components)</td>
    <td>0</td>
    <td>0</td>
  </tr>
  <tr>
    <td rowspan="3">prediction</td>
    <td>pad_image</td>
    <td>1</td>
    <td>1</td>
  </tr>
  <tr>
    <td>model (LeNet5)</td>
    <td>16</td>
    <td>44426</td>
  </tr>
  <tr>
    <td>threshold</td>
    <td>0</td>
    <td>1</td>
  </tr>
</tbody>
</table>

For the image processing algorithms there are 9 free parameters. 
Some are subsets of more diverse algorithms.
Others are more bespoke and are optimised specifically for one use case. 

For machine learning, there are 44,426 free parameters.
Compared to the hand crafted image processing algorithms, it is more general and can be repurposed (retrained) for other tasks such as recognising alphabet letters.

As with everything, one does not need to understand these algorithms in depth. But you do need sufficient knowledge of each in order to be able to integrate and fine tune them.

## Conclusion

I hope you enjoyed this series and have a working Sudoku OCR reader yourself now.

---

[^LeNet5]: The 16 fixed parameters for LeNet5 are: $k_1$, $k_2$, $s$, $p$, $n_{out}$ for each convolution layer (5&times;2); $k_1$, $k_2$ for each max pool layer (2&times;2) and $n_{out}$ for the hidden dense layers (2&times;1). This count excludes other hyper-parameters such as training parameters, number of layers, number of choices for activation function etc. 
