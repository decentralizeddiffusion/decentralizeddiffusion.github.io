---
layout: distill
title: Decentralized Diffusion
description: Train diffusion models across GPU clusters without networking bottlenecks.
tags: distill formatting
giscus_comments: true
date: 2021-05-22
featured: true
mermaid:
  enabled: true
  zoomable: true
code_diff: true
map: true
chart:
  chartjs: true
  echarts: true
  vega_lite: true
tikzjax: true
typograms: true

authors:
  - name: David McAllister
    url: "https://en.wikipedia.org/wiki/Albert_Einstein"
    affiliations:
      name: UC Berkeley
  - name: Matthew Tancik
    url: "https://en.wikipedia.org/wiki/Boris_Podolsky"
    affiliations:
      name: Luma AI
  - name: Jiaming Song
    url: "https://en.wikipedia.org/wiki/Nathan_Rosen"
    affiliations:
      name: Luma AI
  - name: Angjoo Kanazawa
    url: "https://en.wikipedia.org/wiki/Nathan_Rosen"
    affiliations:
      name: UC Berkeley

bibliography: 2018-12-22-distill.bib

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
  - name: Motivation
  - name: Simple Intuitions for Diffusion and Flow Models
    # if a section has subsections, you can add them as follows:
    # subsections:
    #   - name: Example Child Subsection 1
    #   - name: Example Child Subsection 2
  - name: Decentralized Diffusion Models
  - name: Why DDMs?

# Below is an example of injecting additional post-specific styles.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---

## Motivation

State of the art image and video diffusion models train on thousands of GPUs that distribute work then communicate results before each optimization step. This means that training clusters must live in centralized facilities with specialized networking hardware and enormous power delivery systems.

This is cost-prohibitive. Academic labs can't build specialized supercomputing clusters with custom networking fabrics. Even companies struggle as they hit fundamental limits on power delivery and networking bandwidth when scaling to many thousands of GPUs. Networking is the critical bottleneck in both cases: GPUs need constant, high-bandwidth communication throughout the system. A lighter network load makes it possible to use compute where it’s available, in different datacenters or across the internet.

<b>Decentralized Diffusion Models</b> tackle this problem. Our method trains a set of expert diffusion models over partitions of the dataset, each in networking isolation from one another. At inference time, they ensemble through a lightweight learned router. We show that this ensemble collectively optimizes the diffusion objective, the same objective as a single model trained over the whole dataset. It even outperforms monolithic diffusion models FLOP-for-FLOP, leveraging sparse computation at train and test time. Crucially, DDMs scale gracefully to billions of parameters and produce great results with small pretraining budgets.

<div class="fake-img l-page">
  <img src="{{ '/assets/img/decentralized_diffusion/teaser_images.jpg' | relative_url }}" alt="DDM Overview" style="width: 100%; height: auto;">
</div>

<div class="caption">
    Some samples from our 8x3B parameter model, trained with just eight independent GPU nodes in less than a week.
</div>

In this post, we present a simple, geometrically intuitive view on diffusion and flow models from which Decentralized Diffusion Models arrive naturally. We also highlight their strong results and implications for training hardware. DDMs make possible simpler training systems that produce better models.

## Simple Intuitions for Diffusion and Flow Models

Diffusion models and rectified flows can be seen as special cases of flow matching, so we use the FM framework for explanation. Most perspectives on diffusion models and flow matching focus on the forward corruption process and the paths it samples for each training example. Let’s instead focus on the regression target of these models: the marginal flow. 

<div class="l-body" style="text-align: center; margin-top: 0px;">
  <img src="{{ '/assets/img/decentralized_diffusion/marginal_flow_int.svg' | relative_url }}" alt="DDM Overview" style="width: 80%; height: auto;">
</div>

The marginal flow, $$u_t$$, represents a vector field at each timestep that transports from $$x_t$$, a noisy latent, to the data distribution ($t=0$). It can be computed analytically over a dataset or regressed in a model (e.g., a Diffusion Transformer) through flow matching. It takes the form of an expectation over $$x_0$$ data samples.

<div class="l-body" style="text-align: center;">
  <img src="{{ '/assets/img/decentralized_diffusion/marginal_flow_sum.svg' | relative_url }}" alt="DDM Overview" style="width: 80%; height: auto;">
</div>

Let’s rewrite the marginal flow as a sum over a discrete dataset for clarity. It’s now easy to see that it’s just a weighted average of the paths from $x_t$ to every data point, $u_t(x_t\|x_0)$. The weights of each path are determined by the normalized probability of drawing $x_t$ from each $x_0$ sample, $p_t(x_t\|x_0)$. Let’s visualize the marginal flow in 2D over a small dataset. 

In the following live plot:
- Data points ($x_0$ samples) are <b style="color: #323083;">dark blue</b>.
- Each path, $u_t(x_t\|x_0)$, is drawn in <b style="color: #2cc779;">turquoise</b> and its opacity represents its weight.
- The noisy latent ($x_t$) is <b style="color: #F84643;">red</b>. Feel free to drag it around and see how the weight of each path is affected.
- The <b style="color: #F84643;">red</b> dotted line shows the denoising path, AKA the marginal flow evaluated at $x_t$.

<div class="l-page">
  <iframe src="{{ '/assets/plotly/plot_one.html' | relative_url }}" frameborder='0' scrolling='no' height="620px" width="100%" style="border: 1px dashed grey;"></iframe>
</div>

<!-- <br style="margin: 6px 0;"> -->

Since the marginal flow is defined at each timestep, the slider above updates the timestep t. $x_t$ will be transported accordingly by Euler integrating the marginal flow. The data points will also scale according to a simple linear schedule, $(1-t)*x_0$, the mean of the Gaussians that determine $p_t(x_t\|x_0)$. At low timesteps, the path weights are much peakier and $x_t$ will be drawn to its nearest neighbor. Play around, this simulates a “perfectly overfit” diffusion model.

This interpretation sets up Decentralized Diffusion Models very naturally. The marginal flow is a linear system, and linear systems are associative. DDMs exploit this associativity to simplify training systems and improve downstream performance.

## Decentralized Diffusion Models

We partition the data into K disjoint clusters  $\{S_1, S_2, \ldots, S_K\}$, and each expert trains on an assigned subset $(x_0 \in S_i)$. Since the marginal flow is a linear combination over data points, we can apply the associative property within each of these data clusters. We therefore rewrite the global marginal flow as a weighted combination of marginal flows over each data partition.

<div class="l-body" style="text-align: center;">
  <img src="{{ '/assets/img/decentralized_diffusion/marginal_flow_associated.svg' | relative_url }}" alt="DDM Overview" style="width: 80%; height: auto;">
</div>

We learn each data-expert flow in a separate model. This is standard flow matching training, so we can reuse popular architectures, hyperparameters and codebases. By linearly combining each model’s prediction at test-time, we sample from the entire distribution with an ensemble that optimizes the global flow matching objective. We must also learn a router to predict the weights at test-time, which ends up being a classifier over data clusters. We discuss this more thoroughly in the paper.

We can visualize the component flows of a Decentralized Diffusion Model in the plot below. By ensembling them at test-time, we recover the global marginal flow. Drag the black $x_t$ circle and time slider around!

<div class="l-page">
  <iframe src="{{ '/assets/plotly/plot_two.html' | relative_url }}" frameborder='0' scrolling='no' height="620px" width="100%" style="border: 1px dashed grey;"></iframe>
</div>

<br>

Our methods figure below outlines the data preprocessing, training and inference stages of Decentralized Diffusion Models. 

<div class="l-body" style="text-align: center;">
  <img src="{{ '/assets/img/decentralized_diffusion/method_wide.jpg' | relative_url }}" alt="DDM Overview" style="width: 100%; height: auto;">
</div>

## Why DDMs?

These are all cute observations, but why does it matter?

Associativity is the key enabler behind many distributed computing algorithms including parallel scans and MapReduce. Decentralized Diffusion Models use the associative property to split diffusion training into many sub-training jobs that proceed independently, with no cross-communication. This means each training job can be assigned to a different cluster in a different location and with different hardware. For example, we train a text-to-image diffusion model on eight independent nodes (8 GPUs each) for around a week. These nodes are readily available to rent from cloud providers, whereas eight nodes with high-bandwidth interconnect must be co-located in one datacenter and are much harder (and more expensive!) to find.

<div class="l-body" style="text-align: center;">
  <img src="{{ '/assets/img/decentralized_diffusion/combined.jpg' | relative_url }}" alt="DDM Overview" style="width: 100%; height: auto;">
</div>
<div class="caption">
    Some nice samples from the eight-node training run.
</div>

What’s the performance hit from this added convenience? There is none. In fact, Decentralized Diffusion Models outperform non-decentralized diffusion models FLOP-for-FLOP.

<div class="l-body" style="text-align: center; margin-top: -240px; margin-bottom: 20px;">
  <img src="{{ '/assets/img/decentralized_diffusion/combined_fid_plots.svg' | relative_url }}" alt="DDM Overview" style="width: 100%; height: auto; clip-path: inset(240px 0 0 0);">
</div>

By selecting only the top data-expert model per step at test-time, the ensemble instantiates a sparse model. This is the same insight behind Mixture-of-Experts. By activating only a subset of parameters at test time, we get better results at the same computational cost. We use the same architectures, datasets and training hyperparameters between monoliths and DDMs in all our evaluations, and we account for the additional cost of training the router (~4%). Serving a sparse model can be inconvenient with less sophisticated infrastructure, so we also demonstrate that we can efficiently distill DDMs into monolith models in the paper. 

<div class="l-body" style="text-align: center; margin-top: 0px; margin-bottom: 20px;">
  <img src="{{ '/assets/img/decentralized_diffusion/laion_scaling_laws.svg' | relative_url }}" alt="DDM Overview" style="width: 80%; height: auto; clip-path: inset(0 0 0 0);">
</div>

Decentralized Diffusion Models also scale gracefully. We saw consistent improvements on evaluations as model size and compute capacity increased. Please see the paper for more detailed analysis of DDMs and how they compare to standard diffusion training.
