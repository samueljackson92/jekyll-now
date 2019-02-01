---
layout: post
title: Machine Learning Benchmarks
---


## MLPerf

![_config.yml]({{ site.baseurl }}/images/mlperf.png){:width="300px" .img-center}

[MLPerf][MLPerf] is a collection of datasets and benchmarks for measuring the performance of machine learning algorithms. The project aims to provide a common basis for measuring the performance both cross problem domains as well as across hardware platforms and is inspired by the SPEC benchmarks for general purpose computing.

MLPerf is currently still in preliminary alpha release (v0.5) at the time of writing. The benchmarks are split into two divisions: Open & Closed.

 - Closed: only a limited number of models and hyper parameters may be used. This is for the case of comparing hardware architectures & software frameworks.

 - Open: For testing state-of-the-art machine learning models. Hardly any restrictions on the hardware and software used.

### Datasets & Problems


 - Image Classification - Resnet-50 v1 applied to Imagenet.
 - object Detection - Mask R-CNN applied to COCO.
 - Single Stage Detector - SSD applied to COCO 2017.
 - Speech Recognition - DeepSpeech2 applied to Librispeech.
 - Translation - Transformer applied to WMT English-German.
 - Recommendation - Neural Collaborative Filtering applied to MovieLens 20 Million (ml-20m).
 - Sentiment Analysis - Seq-CNN applied to IMDB dataset.
 - Reinforcement - Mini-go applied to predicting pro game moves.

### Metrics

The MLPerf benchmarks also measure a number of different performance metics.

 - wall clock time to train a model to target quality
 - power consumption (as a proxy for cost)


## DAWNBench

![_config.yml]({{ site.baseurl }}/images/dawn-bench.png){:width="700px" .img-center}

[DAWNBench][DAWNBench] is part of the [Stanford DAWN project][StanfordDAWN] aiming to make AI tools more effcient and more reliable. The original results of benchmark submissions have now been archived and they are encouraging the community to contribute to MLPerf.


### Datasets & Problems

 - Image classification - ImageNet and CIFAR10
 - Question Answering - SQuAD

### Metrics
 - wall clock time to train a model to target accuracy
 - training cost in USD
 - inference latency
 - inference cost in USD

## Penn Machine Learning Benchmarks

The [Penn Machine Learning Benchmarks][PMLB] (PMLB) are a collection of 165 classification and regression datasets for real world problem, created and curated by the university of University of Pennsylvania. For easy access to the datasets they provide a small [python library][PMLB-lib] that can down load and iterator over the datasets available in the package in a way that inegrates well with `scikit-learn`

### Datasets & Problems
 - Too many to list here but they are split into the following categories
     - Classification - 165 datasets.
     - Regression - 120 datasets.

### Metrics
The metrics a open ended. In the [original paper][PMLB-paper] the only looked at classification problems and the metric reported was accuracy.

## Training Benchmark for DNNs

![_config.yml]({{ site.baseurl }}/images/TBD.png){:width="500px" .img-center}

The [TBD][TBD] benchmarks is a project created by the Univeristy of Toronto and Microsoft Research.

### Datasets & Problems
 - Image Classification - ImageNet1k
 - Machine translation - IWSLT and WMT
 - Object Detection - PASCAL Visual Object Classes
 - Speech Recognition - LibriSpeech ASR
 - Recommendation System
 - Adversarial learning - ImageNet
 - Reinforcement Learning - OpenAI Gym

### Metrics
They published a memory and network profiler for neural networks which support the TensorFlow, MXNet, and CNTK frameworks. The metrics they primarily report for each benchmark are:

 - Training time to accuracy
 - FP32 utilization
 - Memory (GB)
 - Compute utilization (relative time GPU is actively executing kernels).
 - Throughput (samples/s)

## Deep Learning Benchmarking Suite (DLBS)
[DLBS][DLBS] is a collection of machine learning benchmarks created by Hewlett Packard. They provide their suite as a collection of command line utilities for running experiments on a variety of hardware & software.

This benchmarking suite is more of a framework providing a collection of command line utilities and docker images with some model well known to machine learning (e.g. ResNet). They don't appear to publish any performance metrics for results anywhere.

### Datasets & Problems

They don't present any information of results online but are more about providing a framework for running benchmarks.
 - They do provide [this][DLBS-models] collection of models implemented in different frameworks.

### Metrics
The framework looks quite flexiable and can be modified and configured to report different types of info. For a quick search on their website it appears to mostly just support:
 - training time to accuracy
 - inference time.

## DeepBench

### Datasets & Problems

### Metrics


[MLPerf]: https://mlperf.org/
[DAWNBench]: https://dawn.cs.stanford.edu/benchmark/
[StanfordDAWN]: https://dawn.cs.stanford.edu//
[PMLB]: https://github.com/EpistasisLab/penn-ml-benchmarks 
[PMLB-lib]: https://pypi.org/project/pmlb/
[PMLB-paper]: https://doi.org/10.1186/s13040-017-0154-4 
[TBD]: http://tbd-suite.ai/
[DLBS]: https://hewlettpackard.github.io/dlcookbook-dlbs/#/
[DLBS-models]: https://hewlettpackard.github.io/dlcookbook-dlbs/#/models/models?id=supported-models

