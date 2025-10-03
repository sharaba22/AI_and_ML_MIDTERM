import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset from CSV
data = pd.read_csv('data.csv')

# Step 2: Calculate Pearson correlation coefficient
correlation = data['x'].corr(data['y'], method='pearson')
print(f"Pearson correlation coefficient: {correlation:.4f}")

# Step 3: Create scatter plot with regression line
plt.figure(figsize=(8, 6))
sns.regplot(x='x', y='y', data=data, ci=None, line_kws={'color': 'red'})
plt.title('Scatter Plot of x vs y with Regression Line')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.tight_layout()
plt.show()
