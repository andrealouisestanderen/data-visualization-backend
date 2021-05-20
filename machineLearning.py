import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import colors

# prep
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, MaxAbsScaler, QuantileTransformer, OneHotEncoder

# models
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, LinearRegression, Ridge, RidgeCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# validation libraries
from IPython.display import display
from sklearn import metrics
from matplotlib.colors import ListedColormap

# Read csv file
df = pd.read_csv('./files/globalterrorism.csv')

clean_df = df[['region', 'region_txt', 'country', 'country_txt', 'suicide', 'success', 'iyear', 'imonth', 'iday',
               'nkill', 'nkillus', 'nkillter', 'specificity', 'attacktype1', 'targtype1']].dropna()

# could have weapon_type, gname, target_type, suicide, success also as targets

# Transform pandas dataset to dataset for scikit learn
X = clean_df[['attacktype1', 'targtype1']].to_numpy()
y = clean_df['country'].to_numpy()

# Split test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#ohe = OneHotEncoder(sparse=False)
#terrorism_train_transformed = ohe.fit_transform(target)


def gaussianNB():
    """ supervised """
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)

    plt.figure()
    plt.xlim(0, 7)
    plt.ylim(0, 15)
    plot_ellipse(plt.gca(), gnb.theta_[0], np.identity(
        2)*gnb.sigma_[0], color=colors.GREY)
    plot_ellipse(plt.gca(), gnb.theta_[1], np.identity(
        2)*gnb.sigma_[1], color=colors.GREEN)

    plot_name = saveImage()

    return plot_name


def kNeighbours():
    """ unsupervised, clustering classification"""

    # Create color maps
    cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
    cmap_bold = ['darkorange', 'c', 'darkblue']

    h = .02

    # build the model
    clf = KNeighborsClassifier(n_neighbors=10)
    clf.fit(X_train, y_train)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    sns.scatterplot(x=X[:, 0], y=X[:, 1],
                    palette=cmap_bold, alpha=1.0, edgecolor="black")
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    plot_name = saveImage()

    return plot_name


def regression():
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    plt.scatter(X_test, y_test,  color=colors.ORANGE)
    plt.plot(X_test, y_pred, color=colors.YELLOW, linewidth=2)

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


def plot_ellipse(ax, mu, sigma, color="k", label=None):
    """
    Based on
    http://stackoverflow.com/questions/17952171/not-sure-how-to-fit-data-with-a-gaussian-python.
    """
    from matplotlib.patches import Ellipse
    # Compute eigenvalues and associated eigenvectors
    vals, vecs = np.linalg.eigh(sigma)

    # Compute "tilt" of ellipse using first eigenvector
    x, y = vecs[:, 0]
    theta = np.degrees(np.arctan2(y, x))

    # Eigenvalues give length of ellipse along each eigenvector
    w, h = 2 * np.sqrt(vals)

    ax.tick_params(axis='both', which='major', labelsize=20)
    ellipse = Ellipse(mu, w, h, theta, color=color, label=label)  # color="k")
    ellipse.set_clip_box(ax.bbox)
    ellipse.set_alpha(0.9)
    ax.add_artist(ellipse)
    return ellipse
