## Introduction to Machine Learning: Coursework 1 (Decision Trees)

### Introduction

This repository contains our decision tree implementation and evaluation code.

### Data

The ``data/`` directory contains the datasets you need for the coursework.

The primary datasets are:
- ``train_full.txt``
- ``train_sub.txt``
- ``train_noisy.txt``
- ``validation.txt``

Some simpler datasets that you may use to help you with implementation or 
debugging:
- ``toy.txt``
- ``simple1.txt``
- ``simple2.txt``

### Codes

- ``classification.py``

	* Contains the skeleton code for a basic decision tree ``DecisionTreeClassifier`` class, a pruned version of our basic decision tree ``DecisionTreeWithPruning`` class, a random forest implementation of our basic decision tree ``RandomForestClassifier`` class, a hyperparameter variable decision tree class ``AdvancedDecisionTreeClassifier`` and a final, random forest and hyperparameter variable decision tree ``AdvancedRandomForestClassifier`` class.

- ``improvement.py``

	* Contains the skeleton code for the ``train_and_predict()`` function (Task 4.1).

- ``evaluation.py``

	* Contains the skeleton code for the evaluation functions (Task 3).

- ``main.py``

	* Invokes the methods/functions defined in ``classification.py``, ``evaluation.py`` and ``improvement.py``.


### Instructions

Run the ``main.py`` file to complete the coursework.



