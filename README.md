# Linear Regression Demo

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Short Description
A hands-on Python demo that generates noisy linear data, normalizes it, and uses gradient descent to learn the best-fit line—complete with live plotting and progress updates.

## Long Description

This Python script provides an end-to-end, from-scratch walkthrough of building a simple linear regression model. It’s designed for anyone who wants to see exactly how raw data can be created, prepared, and optimized with basic algorithms—perfect for learners who crave intuition behind high-level ML libraries and for developers seeking a lightweight reference implementation. Every step is clearly commented, making it easy to follow, tweak, and extend.

The process starts with **synthetic data generation**: you choose how many points to create, the true slope and intercept of the underlying line, and the amount of random noise to add. That means you can simulate nearly perfect lines or extremely noisy datasets to observe first-hand how noise affects model training. A random-seed option ensures reproducible results, which is essential for consistent experimentation and benchmarking.

Next comes **feature scaling (normalization)**. We center both x and y at zero mean and scale to unit variance—a step that usually speeds up convergence and prevents numeric issues in gradient-based methods. The script also records the original means and standard deviations so that once the model is learned in normalized space, you can convert the slope and intercept back to the real-world scale without loss of accuracy.

At its core, the demo implements **batch gradient descent with gradient clipping**. In each epoch it calculates the mean squared error and its derivatives with respect to slope and intercept, then caps those gradients to prevent runaway updates. You can easily adjust the learning rate, epoch count, and clipping threshold to see how they influence convergence speed, stability, and final error—giving you hands-on practice in hyperparameter tuning.

Finally, the script uses Matplotlib’s **interactive plotting** mode to visualize training in real time. At set intervals it redraws the scatter of noisy points and overlays the current regression line, with epoch number and loss shown in the title. This dynamic feedback makes it simple to spot issues like oscillations or plateaus, and turns an abstract algorithm into a vivid, educational experience. Use this as a teaching tool, a reference guide, or the starting point for more advanced machine-learning experiments.

1. **Synthetic Data Generation**  
   - Creates `(x, y)` points following `y = m·x + b` with configurable noise and seed for reproducibility.

2. **Feature Normalization**  
   - Scales both features to zero mean and unit variance for faster, more stable learning.

3. **Gradient Descent with Clipping**  
   - Computes derivatives for slope/intercept, applies gradient clipping, and updates parameters by a user-defined learning rate.

4. **Loss Monitoring & Live Visualization**  
   - Calculates mean squared error on original data and uses Matplotlib’s interactive mode to redraw the data + regression line at regular intervals.

5. **Easy Configuration**  
   - Tweak hyperparameters (learning rate, epochs, noise level) directly in the `__main__` section to see how they affect convergence.

## Features
- Customizable data size, slope, intercept, and noise  
- Automatic normalization & denormalization  
- Gradient clipping to prevent runaway updates  
- Interactive plotting of training progress  

## Installation
```bash
git clone https://github.com/YourUsername/linear-regression-demo.git
cd linear-regression-demo
pip install numpy pandas matplotlib
