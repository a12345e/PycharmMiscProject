import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


data = pd.DataFrame({
    'sqft': [750, 900, 1100, 1300, 1500, 1750, 2000, 2200, 2500, 2800],
    'price': [150000, 180000, 210000, 240000, 270000, 300000, 340000, 370000, 410000, 450000]
})

plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # perfect-prediction line
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Predicted vs Actual')
plt.show()
# sns.set_theme(style="whitegrid")  # nice default styling
#
# # Scatter plot with a regression line — one line does what took several in matplotlib
# sns.regplot(data=data, x='sqft', y='price')




plt.show()