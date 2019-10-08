## Artificial Neural Networks

<span data-acronym-label="ann" data-acronym-form="singular+short">ann</span>
were originally developed as a mathematical model of the biological
brain (). Although
<span data-acronym-label="ann" data-acronym-form="singular+short">ann</span>
have little resemblance to real biological neurons, they are a powerful
<span data-acronym-label="ml" data-acronym-form="singular+short">ml</span>
tool and one of the most popular research topics in the last years.
Nowadays, most researchers have shifted from the prespective of the
biological neuron model to a more general *function approximator* point
of view; in fact, it was proved that
<span data-acronym-label="ann" data-acronym-form="singular+short">ann</span>
with enough capacity are capable of approximating any measurable
function to any desired degree of accuracy (); this is, however, a
non-constructive proof.

The basic structure of
<span data-acronym-label="ann" data-acronym-form="singular+short">ann</span>
is a network of nodes (usually called neurons) joined to each other by
weighted connections. Many varieties of
<span data-acronym-label="ann" data-acronym-form="singular+short">ann</span>
have appeared over the years with different properties. One important
distinction is between
<span data-acronym-label="ann" data-acronym-form="singular+short">ann</span>
whose connections form feedback loops, and those whose connections are
acyclic.
<span data-acronym-label="ann" data-acronym-form="singular+short">ann</span>
with cycles are typically referred to as recurrent neural networks and
those without cycles are known as
<span data-acronym-label="fnn" data-acronym-form="singular+short">fnn</span>.

In this work, only
<span data-acronym-label="fnn" data-acronym-form="singular+short">fnn</span>
are used, in particular, a special kind that makes use of the
convolution operation called
<span data-acronym-label="cnn" data-acronym-form="singular+short">cnn</span>.
The following section provides an overview of this networks as well as
the basic principles of training them.

### Convolutional Neural Networks (CNN)

<span data-acronym-label="cnn" data-acronym-form="singular+short">cnn</span>
are a kind of
<span data-acronym-label="ann" data-acronym-form="singular+short">ann</span>
particulary well-suited for computer vision tasks. They were first
introduced in  to perform the task of hand-written digit classification
and later popularized by  entry on the ImageNet Large Scale Visual
Recognition Challenge 2012 (ILSVRC2012), which won the classification
task with a large margin of 10%.

<span data-acronym-label="cnn" data-acronym-form="singular+short">cnn</span>
consist of an input and an output layer, as well as multiple hidden
layers (see figure [1](#fig:alexnet)). The input layer contains the data
(e.g., RGB image) with minimal preprocessing (normalization,
cropping...), in contrast to other
<span data-acronym-label="ml" data-acronym-form="singular+short">ml</span>
algorithms that need hand-engineered features. The output layer is
different depending on the task.

Each hidden layer performs the convolution operation with one or more
filters (commonly refered to as kernels by the
<span data-acronym-label="dl" data-acronym-form="singular+short">dl</span>
community) taking the previous layer’s output as the input and then an
element-wise non-linear function is applied to the output. The
non-linearities allow the model to extract hierarchical features (early
layers extract the called low-level features and deeper layers extract
high-level features) from the input data as it is illustrated in figure
[2](#fig:cnn-visualization) on page  extracted from .

Down-sampling (also known as pooling in the
<span data-acronym-label="dl" data-acronym-form="singular+short">dl</span>
literature) is also a very common operation applied after some hidden
layers, aimed to make the model tranlation-invariant and reduce memory
needs. Three main methods can be used to represent the set of \(N\) (or
\(N \times N\)) neighbouring samples with a single number:

Max-pooling uses the maximum value;

The average value is used by the average-pooling method and

Standard decimation “takes” a sample out of every \(N\) samples, it is
usually implemented via \(N\)-strided (skipping \(N - 1\) positions when
sliding the filter) convolution to compute only the used values.
<span id="n-strided-conv" label="n-strided-conv">\[n-strided-conv\]</span>

Other kinds of layers like dropout () and batch-normalization () can be
used for regularization or faster training process.

![Visualization of a CNN with five convolutional layers and two
fully-connected layers (a kind of layer not described in this
work).<span label="fig:alexnet"></span>](../images/nn.png)

![Visualization of features in a fully trained model. For layers 2-5 the
top 9 activations in a random subset of feature maps are shown projected
down to pixel space using the “deconvolutional” network introduced in 
work<span label="fig:cnn-visualization"></span>](../images/CNN_visualization.png)

### Training <span data-acronym-label="ann" data-acronym-form="singular+short">ann</span>

Finding (or learning) the best set of parameters (\(\theta\)) (weights,
filters, ...) of a network for a given problem can be posed as an
optimization problem by defining an appropiate objective function
(\(\mathcal{J}\)). In a supervised setting, where training data composed
by inputs (**X**) and targets (**Y**) is available:

\[\theta^* = \argmin_{\theta} \mathcal{J}
\left( (\tensor{X}, \tensor{Y}), f_{\theta} \right)\]

The complex structure of
<span data-acronym-label="ann" data-acronym-form="singular+short">ann</span>
makes this optimization problem non-convex, hence iterative methods are
used for moving towards an optimum solution; particulary, gradient
descent based are the most common type as the networks are —by
construction— fully-differentiable.

The idea behind gradient descent is simple. Given a (random) initial
value for the input variables (e.g., filter coefficients in the case of
<span data-acronym-label="cnn" data-acronym-form="singular+short">cnn</span>),
these are updated by moving towards the direction with greater slope
i.e., the gradient. Moving towards the direction of the gradient
(gradient ascent) will yield a local maximum, useful when maximizing the
objective function; while moving toward the direction oposite to the
gradient will yield a local minimum: \[\label{eq:gradient-descent}
\iteration{\theta_i}{n} \leftarrow \iteration{\theta_i}{n - 1}
- \mu \frac{\partial \mathcal{J}}{\partial \theta_i}
\rvert_{\iteration{\theta_i}{n - 1}}\] where the superscript \((n)\)
denotes the \(n\)-th iteration step and \(\mu\) (known as learning rate)
controls how much the variables should change between steps. Note that
the length of the “step” is not only governed by the learning rate but
also by the magnitute of the gradient.

In a
<span data-acronym-label="dnn" data-acronym-form="singular+short">dnn</span>
with thousands or millions of parameters, computing the gradient with
respect to each parameter independently is computationally restrictive.
In  work, an algorithm to efficiently train
<span data-acronym-label="ann" data-acronym-form="singular+short">ann</span>
called backpropagation which uses the principles of dynamic programming
and exploits the chain rule was presented and it is how almost all
<span data-acronym-label="ann" data-acronym-form="singular+short">ann</span>
are trained nowadays.

#### Cost function

When training
<span data-acronym-label="ann" data-acronym-form="singular+short">ann</span>,
the objective function is commonly defined as a cost function with two
parts: the expected value of a loss plus a (weighted) regularization
term.

<span id="eq:cost-function" label="eq:cost-function">\[eq:cost-function\]</span>
\[\begin{aligned}
\mathcal{J}\left( (\tensor{X}, \tensor{Y}), f_{\theta} \right) & =
\mathbb{E}\{ L \left( f_{\theta}(\tensor{X}), \tensor{Y} \right) \}
+ \lambda R(f_{\theta}) \tag{\ref{eq:cost-function}}\\
& \approx \frac{1}{N}
\sum_{\tensor{x}_i, \tensor{y}_j \in \tensor{X}, \tensor{Y}}
{L \left( f_{\theta} (\tensor{x}_i), \tensor{y}_j \right)}
+ \lambda R(f_{\theta}) \label{eq:sgd}\end{aligned}\]

Equation [\[eq:sgd\]](#eq:sgd) shows how the expected value is
approximated by taking the mean over \(N\) samples (batch size) of the
dataset. Stochastic gradient descent is the case where \(N=1\), and
mini-batch gradient descent when \(N\) is smaller than the total number
of samples in the dataset; a small batch size is almost mandatory for
large datasets, as computing the loss for every sample at each iteration
step is very time and memory expensive and not only that: a small batch
size helps avoiding bad local optima and improve generalization ().

The loss function will depend on our task. For example, in
classification problems with \(c\) clases the cross entropy can be used:
\(L(\hat{\tensor{y}}_i, \text{y}_i) = -\log [\hat{\tensor{y}}_i]_{\text{y}_i},
\quad \text{y}_i \in \{0\ ..\ c - 1\}, \hat{\tensor{y}}_i \in \R^{c}\);
this loss enforces the model to make a good estimation of the class
probabilities given an input datapoint. A loss function well aligned
with our task is crucial, but defining such mathematical description of
some problems is not always straightforward.

#### Advanced gradient-based optimization methods

The basic (stochastic/mini-batch) gradient descent methods tend to find
bad sub-optima when dealing with the noisy, non-convex landscape of
<span data-acronym-label="ann" data-acronym-form="singular+short">ann</span>;
with a performance very sensitive to the initial values, learning rate
and batch size. Many research work () has focused on this area,
developing algorithms that try to find better optima with fewer
iterations. The one used in this work is the Adam “optimizer” () which
defines the following update rule based on adaptive estimates of
gradient moments:

<span id="eq:adam" label="eq:adam">\[eq:adam\]</span>

\[\iteration{\theta_i}{n} \leftarrow \iteration{\theta_i}{n-1} -
\mu \frac{\iteration{\hat{m}_i}{n}}{\sqrt{\iteration{\hat{v}_i}{n}} + \epsilon}
\tag{\ref{eq:adam}}\]

\[\label{eq:adam-m}
\iteration{\hat{m}}{n} \leftarrow \frac{\iteration{m}{n}}{1 - \beta_1^n},\quad
\iteration{m_i}{n} \leftarrow \beta_1 \iteration{m_i}{n-1} +
(1-\beta_1)\iteration{g_i}{n}\]

\[\label{eq:adam-v}
\iteration{\hat{v}_i}{n} \leftarrow \frac{\iteration{v_i}{n}}{1 - \beta_2^n},\quad
\iteration{v_i}{n} \leftarrow \beta_2 \iteration{v_i}{n-1} +
(1-\beta_2)\left(\iteration{g_i}{n}\right)^2\]

\[\iteration{g_i}{n} \leftarrow
\frac{\partial \mathcal{J}}{\partial \theta_i}
\rvert_{\iteration{\theta_i}{n - 1}}\]

The parameter update equation [\[eq:adam\]](#eq:adam) looks similar to
the one in [\[eq:gradient-descent\]](#eq:gradient-descent) but instead
of directly using the gradient, its based on the exponential moving
average of the gradient (\(\hat{m}\)) with a coefficient of
\(1 - \beta_1\) (equation [\[eq:adam-m\]](#eq:adam-m)) and the
exponential moving average of the squared gradient (\(\hat{v}\)) with a
coefficient of \(1 - \beta_2\) (equation [\[eq:adam-v\]](#eq:adam-v)).
The \(\beta\)’s are refered to as momentum an typical values lie around
0.9. \(\epsilon\) is a very small constant (e.g, \(10^{-8}\)) to avoid
division by zero. The intuition behind this update rule is that the
steps are forced to be of size \(\mu\), as we are dividing by the
magnitude of the gradient; and instead of taking the direction of the
gradient at a given iteration step, the gradient value is passed through
a low-pass filter to avoid making sudden changes that are common in
stochatic optimization (specially with a small batch size).

## Generative Adversarial Networks (GANs)

In
<span data-acronym-label="ml" data-acronym-form="singular+short">ml</span>
generative models are ones which model the distribution of the data.
This models can be used to perform classification through the
conditional distribution (via a prior distribution of the classes) or to
sample/generate data. Multiple generative models have been proposed that
make use of
<span data-acronym-label="dnn" data-acronym-form="singular+short">dnn</span>
(),
<span data-acronym-label="gans" data-acronym-form="singular+short">gans</span>
in particular have gained a lot of attention as they are capable of
generating realistic images.

<span data-acronym-label="gans" data-acronym-form="singular+short">gans</span>
is a framework for estimating generative models via an adversarial
process, in which two models are trained: a generative model (\(G\))
that captures the data distribution, and a discriminative model (\(D\)).

Both models are implemented as
<span data-acronym-label="dnn" data-acronym-form="singular+short">dnn</span>
(usually
<span data-acronym-label="cnn" data-acronym-form="singular+short">cnn</span>
in the case of image data). \(G_{\theta_g}(\tensor{z})\) maps from an
input space of noise variables with distribution \(p_z(\tensor{z})\) to
data space. \(D_{\theta_d}(\tensor{x})\) outputs the probability that
\(\tensor{x}\) came from the data distribution \(p_{data}(\tensor{x})\)
rather than \(G\). See figure [3](#fig:gan-diagram) for a simple
illustration.

\(G\) is trained to minimize the likelihood of \(D\) assigning a low
probability to its samples. While \(D\) is simultaneously trained to
maximize the probability of assigning the correct label to both training
examples and samples from G. Hence, the loss functions (see equation
[\[eq:cost-function\]](#eq:cost-function)) for each model are defined as
(note how they go one against the other):

\[\label{eq:generator}
L_G = \log(1 - D(G(\tensor{z})))\]

\[\label{eq:discriminator}
L_D =
  \begin{cases}
    - \log(D(\tensor{x}), & \tensor{x} \sim p_{data} \\
    - \log(1 - D(\tensor{x})) \equiv - \log(1 - D(G(\tensor{z}))),
    & \tensor{x} \sim p_g
  \end{cases}\]

Equivalently, \(D\) and \(G\) play the following two-player minimax
game: \[\label{eq:gan-minmax}
\min_G \max_D \mathbb{E}_{x \sim p_{data}(x)}\{\log D(x)\} +
\mathbb{E}_{z \sim p_z(z)} \{\log(1 - D(G(z)))\}\]

If \(G\) and \(D\) have enough capacity, they will reach a point at
which both cannot improve because \(p_g = p_{data}\). The discriminator
will be unable to differentiate between the two distributions, i.e.
\(D(x) = 0.5\).

![<span data-acronym-label="gans" data-acronym-form="singular+short">gans</span>
training process
diagram<span label="fig:gan-diagram"></span>](../images/GAN-diagram.png)

This way of traning a generative model is unstable and can fail to
converge due to a number of problems known as *failure modes*:

  - Mode collapse: the generator collapses which produces limited
    varieties of samples,

  - Diminished gradient: the discriminator gets too successful that the
    generator gradient vanishes and learns nothing,

  - Unbalance between the generator and discriminator causing
    overfitting,

A number of publications () tackle this problems with architecture and
loss function modifications, and practical techniques. Successfuly
tranined generators, generate very realistic samples as the constructed
objective function essentially says “generate samples that look
realistic”.

### GANs for image-to-image translation

In this section, two frameworks for image-to-image translation based on
<span data-acronym-label="gans" data-acronym-form="singular+short">gans</span>
are described. The first one, called pix2pix, () uses a conditional
generative adversarial network to learn a mapping from input to output
images using paired data. On the other hand,
<span data-acronym-label="cyclegans" data-acronym-form="singular+short">cyclegans</span>
() mapping is learned without paired data. Paired data consists of
training examples \(\{\tensor{x}_i, \tensor{y}_i\}_{i=1}^N\) where
correspondence between \(\tensor{x}_i\) and \(\tensor{y}_i\) exists,
obtaining this kind of data can be difficult (or impossible) and
expensive; contrarily, unpaired data consists of a source set
\(\tensor{X}\) and a target set \(\tensor{Y}\) with no information
provided as to which \(\tensor{x}_i\) matches which \(\tensor{y}_i\) (if
any).

#### Pix2Pix

work presents a conditional adversarial setting and applies it
successfully to a variety of image-to-image translation problems that
traditionally would require very different loss formulations; proving
that *learned loss functions* are versatile. The losses used are similar
to the ones in equations [\[eq:generator\]](#eq:generator) and
[\[eq:discriminator\]](#eq:discriminator); but the discriminator not
only gets the output of the generator as input, but the corresponing
input as well (see figure [4](#fig:pix2pix) extracted from ). Apart from
that, the generator is tasked to also be near the ground truth in a L1
sense; this is done by modifying the generator’s loss function:
\[L_G = \log(1 - D(\tensor{x}, G(\tensor{x}))) +
\lambda \left\| \tensor{y} - G(\tensor{x}) \right\|_1\]

![Training a conditional GAN to map edges to photo. The discriminator,
\(D\), learns to classify between fake and real <span>edge, photo</span>
tuples. The generator, \(G\), learns to fool the discriminator. Unlike
an unconditional GAN, both the generator and discriminator observe the
input edge map.<span label="fig:pix2pix"></span>](../images/pix2pix-diagram.png)

#### CycleGAN

presents a method that builds on top of the Pix2Pix framework for
capturing special characteristics of one image collection and
transfering these into another image collection in the absence of any
paired training examples. In theory, an adversarially trained generator
can learn to map images from a domain \(X\) to look indistiguishable
from images from a domain \(Y\); in practice, it is difficult to
optimize the adversarial objective in isolation as this often leads to
the mode collapse problem. In  work, this problem is addressed by adding
more structure to the objective; concretely, it “encourages” the mapping
to be cycle-consistent, i.e.: a function \(\cycleGAN{X}{Y}\) that maps
from domain \(X\) to \(Y\) should have and inverse \(\cycleGAN{Y}{X}\)
that maps its output to the original input; as in language translation,
if a sentence is tranlated from Spanish to English and then back to
Spanish we should arrive to a sentence close to the original.
\(\cycleGAN{Y}{X}(\cycleGAN{X}{Y}(\tensor{x})) \approx \tensor{x}\).
This is done by simultaneously training two generators (with
corresponding discriminators \(D_X\), \(D_Y\) —notice how in this case,
these do not take the source image as input—) and tasking them to not
only “fool” their discriminator but to also produce an image that is
close to the original input when translated back to the source domain
(using the complementary generator), this is done by defining the
following losses for the generators (more clearly visualized in figure
[5](#fig:cyclegan)).  
An aditional term called idenity loss can be added to encourage the
mapping to preserve color composition between the input and the output
by making the generator be near an idenity mapping when samples from the
target domain are provided. Note that the input and output domains need
to have the same number of channels.  
The final losses for the generators are the following:

\[\begin{split}
L_{\cycleGAN{X}{Y}} = &\log(1 - D_Y(\cycleGAN{X}{Y}(\tensor{x}))) \\
&+ \lambda_{cycle} \left\| \tensor{x} -
\cycleGAN{Y}{X}(\cycleGAN{X}{Y}(\tensor{x})) \right\|_1 \\
&+ \lambda_{identity} \left\| \tensor{y} - \cycleGAN{X}{Y}(\tensor{y}) \right\|_1
\end{split}\] \[\begin{split}
L_{\cycleGAN{Y}{X}} = &\log(1 - D_X(\cycleGAN{Y}{X}(\tensor{y}))) \\
&+ \lambda_{cycle} \left\| \tensor{y}
- \cycleGAN{X}{Y}(\cycleGAN{Y}{X}(\tensor{y})) \right\|_1 \\
 &+\lambda_{identity} \left\| \tensor{x} - \cycleGAN{Y}{X}(\tensor{x}) \right\|_1
\end{split}\]

![(Extracted from ) The mapping model denoted as \(\cycleGAN{X}{Y}\) in
this work is is denoted in this figure as \(G\) and \(\cycleGAN{Y}{X}\)
as \(F\)<span label="fig:cyclegan"></span>](../images/cyclegan-diagram.png)

Note that neither Pix2Pix nor CycleGAN generators use a noise
distribution to generate samples (in contrast to the original
<span data-acronym-label="gans" data-acronym-form="singular+short">gans</span>
framework).
