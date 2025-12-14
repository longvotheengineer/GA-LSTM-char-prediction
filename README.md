# AI in Control Engineering: Comprehensive Algorithms Portfolio
[![MATLAB](https://img.shields.io/badge/Language-MATLAB-orange.svg)](https://www.mathworks.com/products/matlab.html)
[![Python](https://img.shields.io/badge/Language-Python-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()

> **A comprehensive repository containing 9 implementations of fundamental and advanced Artificial Intelligence algorithms. This collection covers Supervised Learning, Unsupervised Learning, Reinforcement Learning, Evolutionary Computation, and Deep Neural Networks, developed for the "AI in Control" course at HCMUT.**

---

## Table of Contents
- [Project Overview](#-project-overview)
- [Repository Structure](#-repository-structure)
- [Module 1: Supervised Learning](#-module-1-supervised-learning)
- [Module 2: Unsupervised Learning](#-module-2-unsupervised-learning)
- [Module 3: Reinforcement Learning](#-module-3-reinforcement-learning)
- [Module 4: Evolutionary Algorithms](#-module-4-evolutionary-algorithms)
- [Module 5: Deep Learning & CNNs](#-module-5-deep-learning--cnns)
- [Authors & Acknowledgments](#-authors--acknowledgments)

---

## Project Overview

This repository serves as a practical implementation guide for core AI concepts. The projects are divided into **9 Homework (HW)** assignments, utilizing a hybrid tech stack of **Python** (for Scikit-learn/Data Science tasks) and **MATLAB** (for Control/Simulation/Math-heavy tasks).

**Key Domains Covered:**
* **Classification & Regression:** Perceptron, SVM, Linear Regression, KNN.
* **Clustering:** K-Means.
* **Decision Making:** Q-Learning.
* **Optimization:** Genetic Algorithms.
* **Sequence Modeling & Vision:** LSTM, AlexNet Architecture.

---

## Repository Structure

| HW | Algorithm | Type | Language | Key Feature |
| :--- | :--- | :--- | :--- | :--- |
| **01** | **Perceptron** | Supervised | Python | Linear binary classification on 2D/3D datasets. |
| **02** | **SVM** | Supervised | Python | Optimal hyperplane finding using Scikit-learn. |
| **03** | **Linear Regression** | Supervised | Python | House price prediction based on Area ($m^2$). |
| **04** | **KNN** | Supervised | Python | Handwritten Digit Recognition (MNIST). |
| **05** | **K-Means** | Unsupervised | MATLAB | Customer segmentation (Age vs. Income). |
| **06** | **Q-Learning** | RL | MATLAB | Pathfinding agent in a multi-room environment. |
| **07** | **AlexNet** | DL Theory | Doc | Manual calculation of CNN parameters. |
| **08** | **Genetic Algorithm** | Optimization | MATLAB | Text prediction using Evolutionary Strategy. |
| **09** | **LSTM** | Deep Learning | MATLAB | RNN built from scratch for sequence modeling. |

---

## Module 1: Supervised Learning

### HW1: Perceptron Algorithm
Implementation of the single-layer Perceptron for binary classification.
* **Task:** Linearly separate Class 0 and Class 1 in both 2D and 3D spaces.
* **Result:** Successfully converged to find the separating weights.

### HW2: Support Vector Machine (SVM)
Utilizing SVM to find the optimal hyperplane that maximizes the margin between classes.
* **Tools:** `scikit-learn`.
* **Outcome:** Visualized the support vectors and the decision boundary.

### HW3: Linear Regression (House Price Prediction)
A classic regression problem predicting housing prices based on area.
* [cite_start]**Formula:** $y = 1.9449x + 107.8733$[cite: 157].
* [cite_start]**Performance:** Achieved an RMSE of **6.53 million VND**[cite: 161].
* **Visualization:** Plotting the "Line of Best Fit" against real data points.

### HW4: K-Nearest Neighbors (KNN) - MNIST
A computer vision system to recognize handwritten digits (0-9).
* **Dataset:** MNIST (28x28 pixel images).
* **Optimization:** Tested $k \in \{1, 3, 5, 7, 9, 11\}$.
* [cite_start]**Result:** Optimal accuracy of **91.60%** achieved at **$k=3$**[cite: 335, 380].

---

## Module 2: Unsupervised Learning

### HW5: K-Means Clustering
Grouping data points without labeled outcomes.
* [cite_start]**Scenario:** Segmentation of population based on **Age** and **Income**[cite: 438].
* **Technique:** Used the **Elbow Method** to determine the optimal number of clusters.
* [cite_start]**Result:** The Elbow plot indicated $K=3$ as the optimal cluster count[cite: 591].

---

## Module 3: Reinforcement Learning

### HW6: Q-Learning (Navigation)
A model-free reinforcement learning algorithm where an agent learns to navigate rooms to reach a goal.
* **Mechanism:** Agent explores states (Rooms 0-5), updates the Q-Matrix based on Rewards (R-Matrix).
* **Parameters:** Learning Rate $\alpha = 1$, Discount Factor $\gamma = 0.8$.
* [cite_start]**Convergence:** The Q-Matrix converged at **Episode 2457**[cite: 687].

---

## Module 4: Evolutionary Algorithms

### HW8: Genetic Algorithm (Text Prediction)
Using the MATLAB **GA Toolbox** to "evolve" random strings into target words.
* **Tasks:**
    1.  **Sequential Typing:** Predicting the next character.
    2.  **Word Suggestion:** Finding full words ("hello", "green", "white").
* **Configuration:** Tournament Selection, Laplace Crossover, Adaptive Mutation.
* **Performance:**
    * "Hello": Converged in ~91 generations.
    * "Green": Converged in ~72 generations.
    * [cite_start]"White": Converged in ~78 generations[cite: 815, 965, 998].

> **Optimization Process:**
> The penalty value drops to 0 as the population evolves to match the target string perfectly.

---

## Module 5: Deep Learning & CNNs

### HW7: AlexNet Architecture Analysis
A theoretical breakdown of the famous AlexNet CNN.
* **Task:** Manual calculation of learnable parameters for every layer.
* [cite_start]**Total Parameters Calculated:** **62,378,344**[cite: 753].

### HW9: Long Short-Term Memory (LSTM) From Scratch
A rigorous MATLAB implementation of LSTM cells to solve the Vanishing Gradient problem.
* **Core Feature:** Implemented purely using matrix operations (no high-level DL toolboxes) to visualize $f_t$, $i_t$, $o_t$, and Cell State $C_t$.
* **Application:** Character-level language model to learn the sequence "h-e-l-l-o".
* **Visualization:** Dynamic dashboard showing gate saturation and memory retention over time.

---

## Authors & Acknowledgments

[cite_start]**Group L01 - AI in Control Engineering** [cite: 8, 13]
* [cite_start]**Instructor:** Dr. Pham Viet Cuong [cite: 7]
* [cite_start]**Institution:** Ho Chi Minh City University of Technology (HCMUT) [cite: 2]

**Team Members:**

| Full Name | Student ID | Responsibility |
| :--- | :---: | :--- |
| **Thanh Le Van** | `2213087` | HW 1, 2 |
| **Thuong Tran Dinh** | `2213424` | HW 3, 4 |
| **An Truong Tuan** | `2210041` | HW 5, 6, 7 |
| **Long Vo** | `2211910` | HW 8 & 9 |
| **Thien Huynh Duc** | `2213249` | Compilation |

---
*Created December 2025*
