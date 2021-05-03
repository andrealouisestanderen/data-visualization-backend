import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import colors
import time


def getColumns(file):
    df = pd.read_csv('./files/' + file)
    columns = df.columns.tolist()
    logic_columns = []
    for column in columns:
        if ' ' not in column:
            logic_columns.append(column)

    return logic_columns


def barChart(file, column, sort_by_count, bins):
    df = pd.read_csv('./files/' + file)
    sort = sort_by_count == "1"
    df_parsed = df[[column]]
    values = df_parsed[column].value_counts(sort=sort).index.tolist()
    counts = df_parsed[column].value_counts(sort=sort).tolist()
    if bins == 'head':
        values = df_parsed[column].value_counts(sort=sort).index.tolist()[:10]
        counts = df_parsed[column].value_counts(sort=sort).tolist()[:10]

    # Make plot
    if sort:
        plt.bar(range(len(values)), list(
            map(float, counts)), color=colors.GREEN)
        plt.xticks(range(len(values)), values, rotation='vertical')

    else:
        plt.bar(values, counts, color=colors.GREEN)
        plt.xticks(rotation='vertical')

    plt.xlabel(column)
    plt.ylabel('Amount')

    # save img
    plot_name = saveImage()

    return plot_name


def lineChart(file, year, group):
    # in case of globalterrorism.csv, recommended groups are:
    # attacktype1_txt, targtype1_txt, region_txt, success, suicide, weaptype1_txt
    count = 0
    style = 0
    styles = ['solid', 'dashed', 'dotted', 'dashdot']
    clrs = colors.COLORS

    df = pd.read_csv('./files/' + file)
    df_parsed = df[[year, group]]
    group_list = df_parsed[group].unique()

    for member in group_list:
        df_group = df_parsed.loc[df_parsed[group] == member]

        values = df_group[year].value_counts().sort_index(
            ascending=True).index.tolist()
        counts = df_group[year].value_counts(
        ).sort_index(ascending=True).tolist()
        plt.plot(values, counts, color=clrs[count],
                 label=member, linestyle=styles[style])

        if count < len(clrs) - 1:
            count += 1
        else:
            style += 1
            count = 0
        if style == len(styles)-1:
            style = 0

    plt.legend()
    plt.xlabel(year)
    plt.ylabel(group)

    # save img
    plot_name = saveImage()

    return plot_name


def scatterPlot(file, column1, column2):
    df = pd.read_csv('./files/' + file)
    x = df[[column1]]
    y = df[[column2]]
    plt.scatter(x, y, color=colors.GREEN)

    plt.xlabel(column1)
    plt.ylabel(column2)

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
