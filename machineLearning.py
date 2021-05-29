import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import time
import colors

# prep
from sklearn.model_selection import train_test_split

# models
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans

# validation libraries
from sklearn import metrics

# colors
from matplotlib.colors import ListedColormap, colorConverter, LinearSegmentedColormap

# Read csv file
full_df = pd.read_csv('./files/globalterrorism.csv')

rel_df = full_df[['gname', 'region', 'region_txt', 'country', 'country_txt', 'suicide', 'success', 'iyear', 'imonth',
                  'iday', 'nkill', 'nkillus', 'nkillter', 'nwound', 'property', 'specificity', 'attacktype1', 'attacktype1_txt', 'targtype1', 'targtype1_txt']].dropna()

clean_df = rel_df.loc[(rel_df != 'Unknown').all(1)]

# include only the rows with groupnames that are involved in more than 1000 (top 10) attacks
group_names = clean_df['gname'].value_counts()[0:10].index.tolist()

smaller_df = clean_df.loc[clean_df['gname'].isin(group_names)]


def gaussianNB(target, features):
    """ supervised """

    df = smaller_df

    X = df[features].to_numpy()
    y = df[target].to_numpy()

    # Split test and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)

    disp = metrics.plot_confusion_matrix(gnb, X_test, y_test)
    disp.figure_.suptitle(
        "Confusion Matrix predicting success based on #kills and #wounded")
    #x_vals = range(0, len(target_names))
    #plt.xticks(x_vals, target_names, rotation=25, fontsize=5)
    #plt.yticks(x_vals, target_names, rotation=75, fontsize=5)

    score = "{:.2f}".format(gnb.score(X_test, y_test))

    plot_name = saveImage()

    return plot_name, score


def kMeans(target, features):
    """ unsupervised, clustering classification"""

    df = smaller_df

    X = df[features].to_numpy()
    y = df[target].to_numpy()

    kmeans = KMeans(n_clusters=3)  # 3 - clusters
    grouping = kmeans.fit(X)

    # The cluster centers are stored in thecluster_centers_ attribute, and we plot them as triangles
    discrete_scatter(X[:, 0], X[:, 1], grouping.labels_, markers='o')
    discrete_scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[
                     :, 1], markers='^', markeredgewidth=2)  # range(len(kmeans.cluster_centers_)),

    plt.title(
        'Kmeans clustering with 3 clusters based on #killed, \n #killedterrorists, the day and the month.')

    silhouette_avg = metrics.silhouette_score(X, grouping.labels_)

    plot_name = saveImage()

    return plot_name, silhouette_avg


def regression(target, features):

    df = smaller_df

    X = df[features].to_numpy()
    y = df[target].to_numpy()

    # Split test and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    plt.scatter(X_test, y_test,  color=colors.ORANGE)
    plt.plot(X_test, y_pred, color=colors.YELLOW, linewidth=2)
    plt.xlabel(target)
    plt.ylabel(features[0])

    plt.title('A regression model for #wounded and #killed')

    plot_name = saveImage()

    return plot_name


def saveImage():
    new_plot_name = "plot" + str(time.time()) + ".png"

    for filename in os.listdir('static/'):
        if filename.startswith('plot'):  # not to remove other images
            os.remove('static/' + filename)

    plt.tight_layout()
    plt.savefig('static/' + new_plot_name, dpi=140)
    plt.close()

    return new_plot_name


def discrete_scatter(x1, x2, y=None, markers=None, s=10, ax=None,
                     labels=None, padding=.2, alpha=1, c=None, markeredgewidth=None):
    """Adaption of matplotlib.pyplot.scatter to plot classes or clusters.

    Parameters
    ----------

    x1 : nd-array
        input data, first axis

    x2 : nd-array
        input data, second axis

    y : nd-array
        input data, discrete labels

    cmap : colormap
        Colormap to use.

    markers : list of string
        List of markers to use, or None (which defaults to 'o').

    s : int or float
        Size of the marker

    padding : float
        Fraction of the dataset range to use for padding the axes.

    alpha : float
        Alpha value for all points.
    """
    if ax is None:
        ax = plt.gca()

    if y is None:
        y = np.zeros(len(x1))

    unique_y = np.unique(y)

    if markers is None:
        markers = ['o', '^', 'v', 'D', 's', '*',
                   'p', 'h', 'H', '8', '<', '>'] * 10

    if len(markers) == 1:
        markers = markers * len(unique_y)

    if labels is None:
        labels = unique_y

    # lines in the matplotlib sense, not actual lines
    lines = []

    current_cycler = mpl.rcParams['axes.prop_cycle']

    for i, (yy, cycle) in enumerate(zip(unique_y, current_cycler())):
        mask = y == yy
        # if c is none, use color cycle
        if c is None:
            color = cycle['color']
        elif len(c) > 1:
            color = c[i]
        else:
            color = c
        # use light edge for dark markers
        if np.mean(colorConverter.to_rgb(color)) < .4:
            markeredgecolor = "black"
        else:
            markeredgecolor = "black"

        lines.append(ax.plot(x1[mask], x2[mask], markers[i], markersize=s,
                             label=labels[i], alpha=alpha, c=color,
                             markeredgewidth=markeredgewidth,
                             markeredgecolor=markeredgecolor)[0])

    if padding != 0:
        pad1 = x1.std() * padding
        pad2 = x2.std() * padding
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_xlim(
            min(x1.min() - pad1, xlim[0]), max(x1.max() + pad1, xlim[1]))
        ax.set_ylim(
            min(x2.min() - pad2, ylim[0]), max(x2.max() + pad2, ylim[1]))
    return lines
