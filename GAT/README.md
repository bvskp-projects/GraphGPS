# Graph Attention Transformer (GAT)

This repository contains the implementation of a Graph Attention Transformer (GAT) and Graph Convolution Network (GCN) based model for various purposes.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)

## Introduction

The Graph Attention Transformer (GAT) is a powerful deep learning model that uses attention logic for assigning weights to edges for information propagation. It is particularly effective in tasks that involve graph-structured data, such as node classification, link prediction, and graph classification.

The Graph Convolution Network (GCN) is a simpler implementation of message passing neural network for learning graph based information by propogating information from all the nearby nodes with equal weightage unlike GCNs which weights each connected node and assigns weights to their contribuation on a node.

This is a primitive implementation of GatV2 using the [DLG](https://docs.dgl.ai/en/latest/generated/dgl.nn.pytorch.conv.GATv2Conv.html) implementation of [Graph Attention Convolution v2](https://arxiv.org/pdf/2105.14491.pdf) and [Graph Convolution Network](https://arxiv.org/abs/1609.02907)

## Installation

Tested on
- Python 3.10

To install the necessary dependencies, run the following command:

```bash
pip install --upgrade torch==2.0.1

pip install --upgrade --force-reinstall dgl -f https://data.dgl.ai/wheels/cu117/repo.html
pip install dglgo -f https://data.dgl.ai/wheels-test/repo.html
```

## Usage for Generating resutls on Cora Dataset

Run the script to generate testing results.

```bash
python3 gatv2_cora.py
```

Current script uses first 50% of data for training and last 10% for testing with a 2 layer model.
Changes for training data is on line 35.
Changes for testing data is on line 40.
Changes for number of layers is on line 46. Note that the number of layers passed as param should be 1 less than the total number of layers in the model as the output layer is not considered by the param and is constant.

## Usage for Testing GAT and GCN on High Pass filter

Run the script to generate a graph with the results.

```bash
python3 gat_highpass_test.py
```