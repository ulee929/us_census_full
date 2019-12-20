import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.style.use('seaborn-darkgrid')
params = {'figure.figsize': (15, 5),
          'legend.fontsize': 'x-large',
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large'}
pylab.rcParams.update(params)


class Visualizations():
    '''Matplotlib and Seaborn visualizations for EDA'''

    def __init__(self, data, columns, target_column="income"):
        '''
        Parameters
        ----------
        data : pandas.core.frame.DataFrame
            dataframe containing data for EDA
        columns : list
            list of column names (must exist in dataframe)
        target_column : str
            target column used for some visualizations
        '''

        self.data = data
        self.columns = columns
        self.target_column = target_column

    def histogram(self):
        '''Plots histograms for each feature'''

        # Set up the figure and the desired size
        fig = plt.figure(figsize=(14, 20))
        # Plot a histogram for each column
        for idx, column in enumerate(self.columns, 1):
            # Set up the subplots for individual histograms
            ax = fig.add_subplot(int(len(self.columns) / 2),
                                 int(len(self.columns) / 2),
                                 idx)
            # Draw the histogram
            ax.hist(self.data[column], bins=20)
            # Set the titles
            ax.set_title(f'Distribution of {column}', size=15)
        # Adjust the size so plots don't run into each other
        fig.subplots_adjust(top=0.93, wspace=0.6)
        # Clean look
        plt.tight_layout()

    def heatmap(self):
        '''Plots Seaborn's correlation heatmap'''
        # Compute the correlation matrix
        corr = self.data.corr()
        # Generate a mask for the upper triangle
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        # Set up the figure and the subplot
        f, ax = plt.subplots(figsize=(12, 12))
        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        # Draw the hearmap
        sns.heatmap(corr, cmap=cmap, vmax=.3, center=0,
                    square=True, linewidths=.5, mask=mask,
                    cbar_kws={"shrink": 0.5, "use_gridspec": True},
                    annot=True, fmt='.2f', annot_kws={"size": 9}, robust=True)
        # Add column names at proper rotation
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=45)
        # Add title
        plt.title(f'Correlation Heatmap of Features and Income')
        # Adjust the size
        plt.subplots_adjust(left=0.12, bottom=0.21, right=0.86, top=0.88,
                            wspace=0.20, hspace=0.20)

    def boxplots(self):
        '''Plots Matplotlib's boxplots'''

        # Set up the figure
        fig = plt.figure(figsize=(14, 20))
        # Plot a box plot for each feature
        for idx, column in enumerate(self.columns, 1):
            # Set up individual subplots
            plt.subplot(int(len(self.columns) / 2),
                        int(len(self.columns) / 2),
                        idx)
            # Plot the boxplot on those subplots
            plt.boxplot(self.data[column])
            # Add a title
            plt.title(f'Boxplot for: {column}')
        # Adjust the plots location
        fig.subplots_adjust(top=0.93, wspace=0.6)
        # Clean look
        plt.tight_layout()

    def categorical_bars(self):
        '''Plots Bar Graphs for Categorical Variables'''

        # Set up the figure
        fig = plt.figure(figsize=(24, 24))
        # Plot a box plot for each feature
        for idx, column in enumerate(self.columns, 1):
            # Set up individual subplots
            plt.subplot(int(len(self.columns) / 2),
                        int(len(self.columns) / 2),
                        idx)
            # Plot the boxplot on those subplots
            self.data[column].value_counts()[:20].plot(kind='barh')
            # Add a title
            plt.title(f'{column}')
        # Adjust the plots location
        fig.subplots_adjust(top=0.93, wspace=0.6)
        # Clean look
        plt.tight_layout()
        plt.show()
