# Pearson Correlation Analysis: x vs y

This project analyzes the linear relationship between two variables, `x` and `y`, using the **Pearson correlation coefficient**. It also visualizes the data with a scatter plot and regression line to help interpret the correlation.

---

## Dataset

The dataset is stored in a file named `data.csv` 


## What Is Pearson Correlation?
The Pearson correlation coefficient (r) is a statistical measure used to evaluate the strength and direction of the linear relationship between two continuous variables.

## Why Use It?
Pearson correlation helps determine whether increases in one variable tend to be associated with increases or decreases in another. It’s widely used in data analysis, machine learning, and scientific research.

## Pearson Correlation Formula

The Pearson correlation coefficient \( r \) is calculated as:

$$
r = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}
{\sqrt{\sum_{i=1}^{n} (x_i - \bar{x})^2 \sum_{i=1}^{n} (y_i - \bar{y})^2}}
$$

Where:
- xᵢ and yᵢ are individual data points
- x̄ and ȳ are the means of x and y
- Σ denotes summation across all data points

## Interpretation Guide

| r value | Meaning                      |
|--------:|------------------------------|
|   1     | Perfect positive correlation |
|  -1     | Perfect negative correlation |
|   0     | No linear correlation        |
| ~0.25   | Weak correlation             |
| ~0.75   | Strong correlation           |


## What Each Step Does
Step 1: Loads the CSV file into a pandas DataFrame.

Step 2: Calculates the Pearson correlation using .corr() with method 'pearson'.

Step 3: Uses Seaborn to create a scatter plot and overlays a regression line to visualize the relationship.



## Output Summary
Pearson r value: -0.2597

Interpretation: Weak negative linear relationship

Graph: Scatter plot with regression line showing a slight downward trend