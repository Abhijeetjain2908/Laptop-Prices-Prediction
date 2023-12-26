# Laptop-Prices-Prediction

**Introduction**

Performs data preprocessing and exploratory data analysis (EDA) on a laptop dataset using Python libraries such as Pandas, Matplotlib, NumPy, and Seaborn. It aims to clean the dataset, visualize relationships between laptop attributes, and prepare the data for modeling.

**Data Loading and Cleaning**

The initial section of the code involves loading a dataset from a CSV file using Pandas. The dataset is then cleaned to handle missing values and duplicated entries.

**Data Transformation**

The code performs various transformations on specific columns, converting strings to numeric data types for features like RAM and Weight. Additionally, it separates information from the 'ScreenResolution' column to extract specific details related to screen resolution.

**Exploratory Data Analysis**

The next section conducts exploratory data analysis using visualizations:

Distribution Plot: Visualizes the distribution of laptop prices.
Bar Plots: Illustrates the distribution of laptop prices across different laptop brands, touchscreen availability, IPS display, CPU brands, OS types, and more.
Scatter Plot: Shows the relationship between laptop screen size (Inches) and prices.
Histogram and Density Plot: Depicts the distribution and density of laptop weights.
Feature Engineering
The code performs feature engineering by categorizing memory types (HDD, SSD) and GPU brands. It also categorizes operating systems into Windows, Mac, and Others/No OS/Linux categories.

**Visualization and Insights**

The code generates visualizations to gain insights into the dataset:

Bar Plots: Illustrates how laptop prices vary based on GPU brands and operating systems.
Histogram and Scatter Plot: Visualizes the distribution and relationship between laptop weights and prices.
Conclusion
The code concludes with visualizations and transformations aimed at understanding the relationships and distributions within the dataset, setting the stage for further analysis or modeling.

**Summary and Results**

Provide a summary of the insights gained from the exploratory analysis, any significant correlations discovered, and how these findings might influence subsequent modeling or predictive analysis.

**Libraries used**

Pandas (import pandas as pd)
Matplotlib (import matplotlib.pyplot as plt)
NumPy (import numpy as np)
Seaborn (import seaborn as sns)
