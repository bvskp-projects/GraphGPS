# Graph Attention Transformer (GAT)

This repository contains the implementation of a Graph Attention Transformer (GAT) based model for [insert purpose here].

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)

## Introduction

The Graph Attention Transformer (GAT) is a powerful deep learning model that uses attention logic for assigning weights to edges for information propagation. It is particularly effective in tasks that involve graph-structured data, such as node classification, link prediction, and graph classification.

This is a primitive implementation of GatV2 using the [DLG](https://docs.dgl.ai/en/latest/generated/dgl.nn.pytorch.conv.GATv2Conv.html) implementation of [Graph Attention Convolution v2](https://arxiv.org/pdf/2105.14491.pdf) on [CoraDataset](https://docs.dgl.ai/en/latest/generated/dgl.data.CoraGraphDataset.html).

## Installation

Tested on
- Python 3.10

To install the necessary dependencies, run the following command:

```bash
pip install --upgrade torch==2.0.1

pip install --upgrade --force-reinstall dgl -f https://data.dgl.ai/wheels/cu117/repo.html
pip install dglgo -f https://data.dgl.ai/wheels-test/repo.html
```

## Usage

Run the script to generate testing results.

```bash
python3 gatv2_cora.py
```