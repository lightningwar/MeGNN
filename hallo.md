# Datasets

This repository contains three real-world graph datasets: VolumeGenre, AlmaMater, and DrugGroup. They extract facts from well-known knowledge bases, namely Freebase, YAGO, and DrugBank, respectively.

+ **VolumeGenre** is the knowledge graph about volumes from Freebase, released by [[1]](#ref-1). The task of this dataset is to classify the genre of volume entities, which are divided into scholarly work, book character, published work, short story, magazine, newspaper, and journal.
+ **AlmaMater** is the knowledge graph pertaining to alma mater from Yago4, in accordance with the procedure of [[2]](#ref-2). The task of this dataset is to classify the alma mater of person entities, which are top 24 colleges in terms of quantity in the raw knowledge base.
+ **DrugGroup** is the knowledge graph concerning on drugs parsed from DrugBank [[3]](#ref-3). The task of this dataset is to classify drug entities according to FDA, which are grouped as approved, nutraceutical, illicit, investigational, withdrawn, and experimental.

The statistical overview of the three datasets is presented in the table below.

|   Dataset   | #Entities |  #Edges   | #Entity Types | #Edge Types | Target | #Classes |
| :---------: | :-------: | :-------: | :-----------: | :---------: | :----: | :------: |
| VolumeGenre |  135,904  |  618,339  |       8       |     36      | Volume |    7     |
|  AlmaMater  |  75,781   |  325,451  |      18       |     38      | Person |    24    |
|  DrugGroup  |  190,493  | 2,543,506 |      15       |     31      |  Drug  |    6     |

## Usage

* unzip the `datasets.zip` file
* The `raw` folder has the explicit **Entity** and **Relation** information of the three knowledge graphs.
* The `processed` folder has the integrated graph data, which is in the class of HeteroData. They can be read directly with additional Python packages **Pytorch** and **PyTorch Geometric** installed in advance. More details in `details.ipynb`.

## Reference

<div id="ref-1"></div>[1] Heterogeneous Network Representation Learning: A Unified Framework With Survey and Benchmark

<div id="ref-2"></div>[2] Node Classification Meets Link Prediction on Knowledge Graphs

<div id="ref-3"></div>[3] DrugBank 5.0: a major update to the DrugBank database for 2018
