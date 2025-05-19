#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 19 2025
@author: sohampradhan

This script:
1. Makes up some straight-line data with noise.
2. Scales (normalizes) it so it’s easier to learn.
3. Runs gradient descent to find the best-fit line.
4. Shows the learning progress with printed updates and live plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generate_linear_data(num_points, slope, intercept, noise_level, random_seed):
    """
    Make fake (x, y) points that roughly follow y = slope*x + intercept,
    but with some random wiggle (noise_level) on y.
    """
    # If the user wants reproducible results, set the seed for random numbers
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Create x values from 1 up to num_points
    x = np.arange(1, num_points + 1)
    
    # Make random noise between -noise_level and +noise_level for each point
    noise = np.random.uniform(-noise_level, noise_level, num_points)
    
    # Calculate y using the line equation plus noise
    y = slope * x + intercept + noise
    
    # Round y to two decimal places to keep things neat
    y = np.round(y, 2)
    
    # Put x and y into a pandas DataFrame so we can handle it easily
    df = pd.DataFrame({
        'x': x,
        'y': y
    })
    
    return df


def normalize_features(df):
    """
    Change x and y so they each have a mean of 0 and a standard deviation of 1.
    Returns the new DataFrame and the stats we need to undo the scaling later.
    """
    # Compute the average (mean) and spread (std) for x and y
    x_mean = df['x'].mean()
    x_std = df['x'].std()
    y_mean = df['y'].mean()
    y_std = df['y'].std()
    
    # Apply normalization formula: (value - mean) / std
    df_norm = pd.DataFrame({
        'x': (df['x'] - x_mean) / x_std,
        'y': (df['y'] - y_mean) / y_std
    })
    
    # Save the original means and stds so we can convert back later
    norm_params = {
        'x_mean': x_mean,
        'x_std': x_std,
        'y_mean': y_mean,
        'y_std': y_std
    }
    
    return df_norm, norm_params


def denormalize_parameters(m_norm, b_norm, norm_params):
    """
    Take slope (m_norm) and intercept (b_norm) from normalized data
    and convert them back to the original data scale.
    """
    # Scale slope by the ratio of y spread to x spread
    m = m_norm * (norm_params['y_std'] / norm_params['x_std'])
    # Adjust intercept to account for shifts and scaling
    b = (b_norm * norm_params['y_std'] + norm_params['y_mean']
         - m * norm_params['x_mean'])
    return m, b


def loss_function(m, b, points):
    """
    Compute the average squared distance between actual y and predicted y = m*x + b.
    This tells us how bad (high) the errors are.
    """
    total_error = 0
    n = len(points)
    for i in range(n):
        x = points.iloc[i].x
        y = points.iloc[i].y
        # Sum up (actual - predicted)^2 for every point
        total_error += (y - (m * x + b)) ** 2
    # Return the mean of those squared errors
    return total_error / float(n)
        

def gradient_descent(m_now, b_now, points, learning_rate, clip_value=5.0):
    """
    Do one round of gradient descent:
    1. Find slope and intercept gradients (derivatives of loss).
    2. Clip them so they don’t jump too far.
    3. Step in the opposite direction by learning_rate.
    """
    m_gradient = 0
    b_gradient = 0
    n = len(points)
    
    # Calculate total gradients over all points
    for i in range(n):
        x = points.iloc[i].x 
        y = points.iloc[i].y
        prediction = m_now * x + b_now
        error = y - prediction  # how far off we are
        # derivative wrt m: -2/n * x * error
        m_gradient += -(2/n) * x * error
        # derivative wrt b: -2/n * error
        b_gradient += -(2/n) * error
    
    # Prevent gradients from being too huge (stabilizes learning)
    m_gradient = max(min(m_gradient, clip_value), -clip_value)
    b_gradient = max(min(b_gradient, clip_value), -clip_value)
    
    # Update rules: new = old - learning_rate * gradient
    m = m_now - learning_rate * m_gradient
    b = b_now - learning_rate * b_gradient
    
    return m, b
        

# If we run this file directly, do the following:
if __name__ == "__main__":
    # 1) Create 100 noisy points along y = 2x + 5, seed=100 for repeatability
    df = generate_linear_data(
        num_points=100,
        slope=2,
        intercept=5,
        noise_level=1,
        random_seed=100
    )
    
    # 2) Peek at the first few rows to check the data
    print(df.head())
    
    # 3) Scale the data so each feature has mean 0 and std 1
    df_norm, norm_params = normalize_features(df)
    
    # 4) Turn on interactive mode so plots update as we go
    plt.ion()
    
    # 5) Start with some guess for slope and intercept in normalized space
    m_norm = 0.5  # a reasonable starting point for slope
    b_norm = 0    # start intercept at zero
    learning_rate = 0.001  # controls step size in gradient descent
    epochs = 3000         # how many updates to do
    
    # 6) Training loop: update parameters many times
    for i in range(epochs):
        # Update normalized slope/intercept
        m_norm, b_norm = gradient_descent(m_norm, b_norm, df_norm, learning_rate)
        
        # Convert to original scale to check real performance
        m, b = denormalize_parameters(m_norm, b_norm, norm_params)
        
        # Every 50 epochs, print progress
        if i % 100 == 0:
            loss = loss_function(m, b, df)
            print(f"Epoch: {i}, m: {m:.4f}, b: {b:.4f}, Loss: {loss:.4f}")
        
        # Every 100 epochs (and at the last one), update the live plot
        if i % 100 == 0 or i == epochs - 1:
            plt.clf()  # clear old plot
            # Show data points
            plt.scatter(df['x'], df['y'], alpha=0.7)
            # Draw the current best-fit line
            x_min, x_max = df['x'].min(), df['x'].max()
            x_range = np.linspace(x_min - 10, x_max + 10, 100)
            y_pred = m * x_range + b
            plt.plot(x_range, y_pred, 'r-', linewidth=2)
            # Title with current stats
            loss = loss_function(m, b, df)
            plt.title(f'Epoch: {i}, m: {m:.4f}, b: {b:.4f}, Loss: {loss:.4f}')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.grid(True, alpha=0.3)
            plt.draw()      # redraw the plot
            plt.pause(0.1)  # short pause so the plot shows
    
    # 7) Turn off interactive mode and show final plot
    plt.ioff()
    plt.show()
