# Hybrid Sequence Modeling: LSTM & Genetic Algorithm (MATLAB)

[![MATLAB](https://img.shields.io/badge/Language-MATLAB-orange.svg)](https://www.mathworks.com/products/matlab.html)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Educational-green.svg)]()

> **A comprehensive MATLAB repository implementing Long Short-Term Memory (LSTM) networks from scratch and utilizing Genetic Algorithms (GA) for Character-Level Language Modeling and Sequence Optimization.**

---

## Table of Contents
- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
- [Method 1: LSTM From Scratch](#-method-1-lstm-from-scratch)
- [Method 2: Genetic Algorithm Optimization](#-method-2-genetic-algorithm-optimization)
- [Visual Diagnostics](#-visual-diagnostics)
- [Authors & Acknowledgments](#-authors--acknowledgments)

---

## Project Overview

This project explores two distinct approaches to **Sequence Prediction** and **Text Generation** (specifically solving the "Hello" problem):

1.  **Neural Approach (LSTM):** A rigorous, mathematically explicit implementation of LSTM cells without relying on high-level Deep Learning toolboxes. It focuses on visualizing the internal state (Gates, Cell Memory, Hidden State) transparency.
2.  **Evolutionary Approach (GA):** Leveraging the Global Optimization Toolbox to "evolve" character strings towards a target, demonstrating how heuristic search can solve sequence problems.

---

## Key Features

| Feature | Description |
| :--- | :--- |
| **LSTM "From Scratch"** | Full implementation of forward propagation, including sigmoid/tanh activations and gate logic ($f_t, i_t, o_t, \tilde{C}_t$). |
| **Live Animation** | Dynamic dashboards showing how neurons activate and how the algorithm "thinks" step-by-step. |
| **Genetic Search** | Implementation of sequential typing prediction and whole-word suggestion using evolutionary strategies. |
| **Interactive CLI** | Scripts that accept user input (`h`, `g`, `w`) to trigger specific optimization tasks. |

---

## Repository Structure

The codebase is organized into modular scripts for core logic, visualization, and execution drivers.

### Core LSTM Logic
| File | Functionality |
| :--- | :--- |
| `lstm_cell.m` | **The Engine.** Calculates the state updates for a single time step given input $x_t$ and previous states $h_{t-1}, C_{t-1}$. |
| `lstm_forward.m` | Handles the temporal loop, processing an entire sequence of inputs through the LSTM cell. |

### LSTM Drivers & Animations
| File | Functionality |
| :--- | :--- |
| `lstm_animate_hello.m` | **Main Demo.** Trains an LSTM to learn "hello", then animates the prediction probability (Softmax) for each character. |
| `lstm_animate_single_step.m` | **Deep Dive.** Visualizes the matrix operations of a *single* time step in a dark-mode dashboard. |
| `lstm_animate_process.m` | Animates the gates and cell state response to a specific input pulse pattern. |
| `lstm_animate_state.m` | Visualizes the real-time continuous signal processing (Input vs. Memory vs. Output). |

### Genetic Algorithm Drivers
| File | Functionality |
| :--- | :--- |
| `hello_run_genetic_algorithm.m`| **Sequential Typing.** Solves "hello" character-by-character using GA at every keystroke. |
| `run_genetic_algorithm.m` | **Word Suggestion.** Finds full words ("hello", "green", "white") based on user input. |
| `find_text_fitness.m` | Cost function calculating the distance between the current string and the target. |
| `print_evolution.m` | Helper function to log generation progress to the command window. |

---

## Getting Started

### Prerequisites
* **MATLAB** (R2018b or newer recommended).
* **Global Optimization Toolbox** (Required for the GA scripts).

### Installation
1.  Clone this repository:
    ```bash
    git clone [https://github.com/YourUsername/LSTM-GA-Char-Prediction-Matlab.git](https://github.com/YourUsername/LSTM-GA-Char-Prediction-Matlab.git)
    ```
2.  Open MATLAB and navigate to the repository folder.
3.  Add the folder to your MATLAB path.

---

## Method 1: LSTM 

The LSTM implementation focuses on the mathematical transparency of the cell updates:

$$ f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) $$
$$ C_t = f_t * C_{t-1} + i_t * \tilde{C}_t $$

### How to Run:
1.  **To see the network learn "hello":**
    Run `lstm_animate_hello.m`.
    * *Note:* The script will first train (displaying loss in the command window) and then launch a figure to animate the prediction.
2.  **To understand the internal math:**
    Run `lstm_animate_single_step.m`.

> **LSTM Internal State:**
<img width="975" height="548" alt="image" src="https://github.com/user-attachments/assets/e4a1b06e-331c-41ad-a38f-5bd703a9be61" />

---

## Method 2: Genetic Algorithm Optimization

This module treats text generation as an optimization problem where the "fitness" is the similarity to a target word.

### How to Run:
1.  **Sequential Typing Simulation:**
    Run `hello_run_genetic_algorithm.m`.
    * Follow the prompts to type "hello". The GA will "search" for the next key for you.
2.  **Word Suggestion:**
    Run `run_genetic_algorithm.m`.
    * Enter `h`, `g`, or `w` when prompted.
    * The script will plot the convergence of the penalty value over generations.

> **Gate Activations:**
<img width="700" height="564" alt="image" src="https://github.com/user-attachments/assets/93eb5dc9-d195-4298-a697-874ddafd583f" />

---

## Visual Diagnostics

The repository includes specialized visualization tools to diagnose network behavior:

* **Gate Saturation:** See when the Forget Gate ($f_t$) is open (1) or closed (0).
* **Memory Retention:** Observe how long the Cell State ($C_t$) holds a value after the input signal vanishes.
* **Softmax Confidence:** Real-time bar charts showing the probability distribution of the next predicted character.

---

## Authors & Acknowledgments

* **Author:** Longvo Theengineer
* **Institution:** Ho Chi Minh city University of Technology (HCMUT)
* **Inspiration:** Concepts based on Character-Level RNNs and Evolutionary Computing strategies.

---
