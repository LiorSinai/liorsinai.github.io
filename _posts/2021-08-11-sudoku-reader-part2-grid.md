---
layout: post
title:  "Sudoku OCR reader in Julia: part 2"
date:   2021-08-10
author: Lior Sinai
categories: coding
tags:	'machine learning'
---

_Extracting the grid for the Sudoku OCR reader._ 

This post is part of a series. The other articles are:
- [Part 1: introduction][introduction].
- [Part 3: digit extraction][digit_extraction].
- [Part 4: machine learning][machine_learning].
- [Part 5: conclusion][conclusion].

[introduction]: {{ "coding/2021/08/10/sudoku-reader-part1-intro" | relative_url }}
[grid_extraction]: {{ "coding/2021/08/10/sudoku-reader-part2-grid" | relative_url }}
[digit_extraction]: {{ "coding/2021/08/10/sudoku-reader-part3-digits" | relative_url }}
[machine_learning]: {{ "coding/2021/08/10/sudoku-reader-part4-cnn" | relative_url }}
[conclusion]: {{ "coding/2021/08/10/sudoku-reader-part5" | relative_url }}

All code is available online at my repository: [github.com/LiorSinai/SudokuReader.jl](https://github.com/LiorSinai/SudokuReader.jl).

# Part 2 - grid extraction
## Overall process

The main goal here is to find contours in the image and assume the largest contour to be the grid.
A "contour" is a line which defines a boundary between the "background" and an "object". 
This assumption works well for images where the Sudoku grid dominates the image.[^contour_assumption]
I found more complicated techniques - e.g. line detections with Hough Transformations - to only add complexity without providing much benefit. 
Machine learning could also be used for this problem - see [MathWork's blog post][MathWorks]. This requires creating a dataset to train on, which can be a time consuming process, and the training itself will require a largish model and will be slow.

After the largest contour has been found, a simple four point quadrilateral is fit to the contour, and that is returned as the grid.

[MathWorks]: https://blogs.mathworks.com/deep-learning/2018/11/15/sudoku-solver-image-processing-and-deep-learning/

## Working with images in Julia

The main packaged used in this code is [JuliaImages.jl][JuliaImages]. 
An image in Julia is an $h\times w$ matrix, where $h$ is the height and $w$ is the width. Its most general type is an `AbstractArray`. 
Each element of the array is a pixel, which can be a native type such as `Int` or `Float64`, or a `Colorant` type such as `Gray` (1 channel), `RGB` (3 channels) or `RGBA` (4 channels). 
At the time of publication, JuliaImages.jl defines 18 `Color` subtypes and 36 `TransparentColor` subtypes.
This format is convenient  because an image can always be expected to be 2D and extra dimensions are handled by the pixel type.

Machine learning often uses a $h\times w \times c$ format, where $c$ is the number of channels.  
The function `channelview` can be used to convert it to $c \times h\times w$ array, and then `permutedims` can convert it to a $h\times w \times c$ array:
 {%highlight julia %}
image_CHW = channelview(img_rgb)
image_HCW = permutedims(img_CHW, (2, 3, 1))
{% endhighlight %}
For grayscale images, calling `Float64.(image)` converts the image directly to a `Float64` array.


With this knowledge in mind, we can start by loading our image.
{%highlight julia %}
using Images
using FileIO  

image_path = "images/nytimes_20210807.jpg"
image = load(image_path)
{% endhighlight %}


[JuliaImages]: https://juliaimages.org/stable/


## Preprocessing
## Resizing 

Resizing the image is useful because:
1. This speeds up all other operations. Most of them are $O(p)$ where $p=hw$ is the number of pixels. Reducing an image size by 2 in both dimensions reduces the number of pixels by 4, which speeds up operations by the same factor.
2. The blurring and adaptive threshold methods that will be described next are not scale invariant. Having a maximum size is insurance that the parameters we pick will always be appropriate.

I found a maximum dimension size of 1024 pixels to be sufficient. We can use it to calculate a ratio and pass this number to our `imresize`:
{%highlight julia %}
ratio = max_size/size(gray, argmax(size(gray)))
if ratio < 1
    gray = imresize(gray, ratio=ratio)
end
{% endhighlight %}    

## Binarize

The next step is to convert the image to black and white. 
This is because it simplifies the calculations greatly for classic image processing techniques.
This is in contrast to machine learning, which can easily handle multiple colour channels and where more information is encouraged.

<figure class="post-figure">
<img class="img-95"
    src="/assets/posts/sudoku-reader/gray.png"
	alt="Grayscale image"
	>
<figcaption>Grayscale image.</figcaption>
</figure>

First convert the image to grayscale. This is done by calling `Gray.(image)`. Out of interest, the underlying code from [Colors.jl][Colors.jl] is :
{%highlight julia %}
cnvt(::Type{G}, x::AbstractRGB) where {G<:AbstractGray} = 
    G(0.299f0*red(x) + 0.587f0*green(x) + 0.114f0*blue(x))
{% endhighlight %}   
This is a standard grayscale conversion where the green spectrum is more heavily weighted because of a natural bias towards green in human eyes. More information here: [www.tutorialspoint.com/dip/grayscale_to_rgb_conversion.htm](https://www.tutorialspoint.com/dip/grayscale_to_rgb_conversion.htm).


[Colors.jl]: https://github.com/JuliaGraphics/Colors.jl/blob/9c55e4a5f787771239eb6bff5c0ec061029c4f00/src/conversions.jl#L777

All the pixels now have a value between 0 and 1. Next we need to threshold values so that the values are either 0 or 1. 
We could apply a simple threshold over the whole picture e.g. `blackwhite = gray .< 0.8`. 
A more intelligent method is to use the adaptive thresholding technique provided by the [ImageBinarization.jl][ImageBinarization.jl] package.
From the documentation:
> If the value of a pixel is $t$ percent less than the average of an $s×s$ window of pixels centered around the pixel, then the pixel is set to black, otherwise it is set to white.

[ImageBinarization.jl]: https://zygmuntszpak.github.io/ImageBinarization.jl/stable/

The code is as follows:
{%highlight julia %}
using ImageBinarization

blackwhite = binarize(gray, AdaptiveThreshold(window_size=15, percentage=7))
{% endhighlight %}    

And the result:
<figure class="post-figure">
<img class="img-95"
    src="/assets/posts/sudoku-reader/blackwhite.png"
	alt="blackwhite image"
	>
<figcaption>Black and white image.</figcaption>
</figure>

This is what we wanted but there are two problems. 
Firstly there is a lot of noise - it would be better if we could remove all those small patches.
Secondly, for the contour detection we need the contours to be white (have a value of 1) and the background to be black (have a value of 0).

To reduce noise, we can apply an image filter with a Gaussian Kernel from [ImageFiltering.jl][ImageFiltering.jl].
In simpler words, every pixel will become a weighted sum of it and its neighbours based on a kernel we give it.
The net effect is a blurring of the image.

[ImageFiltering.jl]: https://juliaimages.org/ImageFiltering.jl/stable/

Here is the code to do this:
{%highlight julia %}
using ImageFiltering

window_size = 5
σ = 1
kernel = Kernel.gaussian((σ, σ), (window_size, window_size))
gray = imfilter(gray, kernel)
{% endhighlight %}  

<figure class="post-figure">
<img class="img-95"
    src="/assets/posts/sudoku-reader/imfilter.png"
	alt="slightly blurred image"
	>
<figcaption>Left: Original image. Middle: blurred. Right: Gaussian kernel</figcaption>
</figure>

As can be seen in the image, the blurring affect is slight. Things do look smoother. Of course if we changed the parameters, say `σ = 5` and 
`window_size=21`, the effects could be more dramatic:


<figure class="post-figure">
<img class="img-80"
    src="/assets/posts/sudoku-reader/blurred.png"
	alt="blurred image"
	>
<figcaption>Heavily blurred image</figcaption>
</figure>


The slight blurring however does have a big effect on the thresholding:
<figure class="post-figure">
<img class="img-80"
    src="/assets/posts/sudoku-reader/blackwhite_blurred.png"
	alt="blackwhite image with blurring"
	>
<figcaption>black and white image with blurring</figcaption>
</figure>

Inverting the image is rather simple. Remember in Julia `for` loops are fast:
{%highlight julia %}
function invert_image(image)
    image_inv = Gray.(image)
    height, width = size(image)
    for i in 1:height
        for j in 1:width
            image_inv[i, j] = 1 - image_inv[i, j]
        end
    end
    return image_inv
end
{% endhighlight %}  

The result:
<figure class="post-figure">
<img class="img-80"
    src="/assets/posts/sudoku-reader/blackwhite_inv.png"
	alt="blackwhite image inverted"
	>
<figcaption>black and white image inverted</figcaption>
</figure>

## Contour Detection

We can finally start looking for contours. The de facto algorithm for doing this is highly specialised, so I will not go into more detail here. For more information, please see the 1987 paper by [Suzuki and Abe][Suzuki1987].

I couldn't find a package with this algorithm (a common occurrence for a young language like Julia)
but thankfully the people at JuliaImages.jl wrote a Julia port of it: [contour_detection][contour_detection].
I copied this code and modified it slightly to (1) better follow Julia Conventions and (2) have the option to only return external contours. The code is available here: [Contours.jl][Contours.jl]. Please note: this is a work in progress. The `fill_contour!` fuction will fail for complex shapes. Also I am very proud of the `point_in_polygon` function, which uses a very robust method from a fairly recent [paper][Galetzka2012]. 

[Suzuki1987]: https://www.sciencedirect.com/science/article/abs/pii/0734189X85900167
[Galetzka2012]: https://arxiv.org/abs/1207.3502
[contour_detection]: https://juliaimages.org/stable/democards/examples/contours/contour_detection/#Contour-Detection-and-Drawing
[Contours.jl]: https://github.com/LiorSinai/SudokuReader.jl/blob/main/utilities/Contours.jl


Here is the result of calling `find_contours(blackwhite)`:
<figure class="post-figure">
<img class="img-80"
    src="/assets/posts/sudoku-reader/2_contours.png"
	alt="contours in the image"
	>
<figcaption>Contours in the image. Blue: external contours. Red: hole contours.</figcaption>
</figure>

Now we're getting somewhere! The grid is clearly the largest contour in the image, which we're going to assume is the general case. With the `calc_area_contour` function that I also provide, it takes three lines to extract it:
{%highlight julia %}
contours = find_contours(blackwhite, external_only=true)
idx_max = argmax(map(calc_area_contour, contours))
contour _max = contours[idx_max]
{% endhighlight %}  


## Quadrilateral fitting

The contour we have is comprised  of many points (in this case, 2163). We want to simplify it to a 4 point quadrilateral.
There are few ways to do this. I found the easiest was to first fit a rectangle to the contour. Its corners can be found as the min and max of the $x$ and $y$ values of the contour points. Then fit a quadrilateral by finding the points on the contour that minimise the rectangular distance to each corner. In a single equation:

$$ min(|x_i - r_{x,j}| + |y_i - r_{y,j}|) \; \forall (x_i, y_i) \in c, \; j \in \{1, 2, 3, 4 \} $$

Where $r$ is the rectangle and $c$ is the contour.

Here is the code:
{%highlight julia %}
function fit_rectangle(points::AbstractVector)
    # return corners in top-left, top-right, bottom-right, bottom-left
    min_x, max_x, min_y, max_y = typemax(Int), typemin(Int), typemax(Int), typemin(Int)
    for point in points
        min_x = min(min_x, point[1])
        max_x = max(max_x, point[1])
        min_y = min(min_y, point[2])
        max_y = max(max_y, point[2])
    end
    
    corners = [
        CartesianIndex(min_x, min_y),
        CartesianIndex(max_x, min_y),
        CartesianIndex(max_x, max_y),
        CartesianIndex(min_x, max_y),
    ]

    return corners
end

function fit_quad(points::AbstractVector) 
    rect = fit_rectangle(points)

    corners = copy(rect)
    distances = [Inf, Inf, Inf, Inf]

    for point in points
        for i in 1:4
            d = abs(point[1] - rect[i][1]) + abs(point[2] - rect[i][2])
            if d < distances[i]
                corners[i] = point
                distances[i] = d
            end
        end
    end
    return corners
end
{% endhighlight %}  

Finally, the result, which we can return as our grid:
<figure class="post-figure">
<img class="img-80"
    src="/assets/posts/sudoku-reader/3_quad.png"
	alt="quadrilateral"
	>
<figcaption>quadrilateral fitted to the grid.</figcaption>
</figure>

# Next section

Now that we have the grid, we can extract the digits inside it. This is explained next at [part 3][digit_extraction].

---

[^contour_assumption]: More robust methods could check for the "squareness" of the contour. For example, $\frac{l_{max}}{l_{min}} - 1 < \epsilon $ where $l$ is the length of a side and $\epsilon$ is some small number.