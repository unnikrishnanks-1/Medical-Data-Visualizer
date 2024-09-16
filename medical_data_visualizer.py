import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Load data
df = pd.read_csv('medical_examination.csv')

# 2. Add overweight column
df['overweight'] = (df['weight'] / (df['height'] / 100) ** 2).apply(lambda x: 1 if x > 25 else 0)

# 3. Normalize cholesterol and glucose values
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)

# 4. Function to draw the categorical plot
def draw_cat_plot():
    # 5. Melt the dataframe to long format for categorical variables
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # 6. Group by cardio, variable, and value to get counts
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

    # 7. Draw the catplot using seaborn
    fig = sns.catplot(x='variable', y='total', hue='value', col='cardio', kind='bar', data=df_cat).fig

    # 8. Save the plot to a file
    fig.savefig('catplot.png')
    return fig

# 10. Function to draw the heatmap
def draw_heat_map():
    # 11. Clean the data
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) &
                 (df['height'] >= df['height'].quantile(0.025)) &
                 (df['height'] <= df['height'].quantile(0.975)) &
                 (df['weight'] >= df['weight'].quantile(0.025)) &
                 (df['weight'] <= df['weight'].quantile(0.975))]

    # 12. Calculate the correlation matrix
    corr = df_heat.corr()

    # 13. Generate a mask for the upper triangle of the heatmap
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14. Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 12))

    # 15. Draw the heatmap with consistent float formatting
    sns.heatmap(corr, annot=True, fmt=".1f", mask=mask, square=True, linewidths=0.5, center=0, vmin=-0.1, vmax=0.25, cmap='coolwarm')

    # 16. Save the heatmap to a file
    fig.savefig('heatmap.png')
    return fig
