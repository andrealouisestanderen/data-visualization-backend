import analysis
import machineLearning
from flask import Flask, session, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
import json
import matplotlib
matplotlib.use('Agg')

UPLOAD_FOLDER = './files'
ALLOWED_EXTENSIONS = set(['csv'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# make this like the image html in cloud
@app.route("/", methods=['GET', 'POST'])
def uploadFile():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            columns = analysis.getColumns(filename)
            return render_template('analysis.html', columns=columns, filename=filename)

    return render_template('upload.html')


@app.route("/bar-chart")
def barChart():

    file = request.args.get('file')
    column1 = request.args.get('column1')
    sort = request.args.get('sort')
    bars = request.args.get('bars')

    plot_name = analysis.barChart(file, column1, sort, bars)

    return render_template('result.html', plot=plot_name)


@app.route("/line-chart")
def lineChart():

    file = request.args.get('file')
    time = request.args.get('column1')
    col2 = request.args.get('column2')
    col3 = request.args.get('column3')
    col4 = request.args.get('column4')
    bins = request.args.get('bins')

    plot_name = analysis.lineChart(file, time, col2, col3, col4, bins)

    return render_template('result.html', plot=plot_name)


@app.route("/scatter-plot")
def scatterPlot():

    file = request.args.get('file')
    column1 = request.args.get('column1')
    column2 = request.args.get('column2')
    bins = request.args.get('bins')

    plot_name = analysis.scatterPlot(file, column1, column2, bins)

    return render_template('result.html', plot=plot_name)


@app.route("/histogram-plot")
def histogramPlot():

    file = request.args.get('file')
    column1 = request.args.get('column1')
    column2 = request.args.get('column2')
    stat = request.args.get('stat')
    bins = request.args.get('bins')

    plot_name = analysis.histogramPlot(file, column1, column2, stat, bins)

    return render_template('result.html', plot=plot_name)


@app.route("/box-plot")
def boxPlot():

    file = request.args.get('file')
    column1 = request.args.get('column1')
    column2 = request.args.get('column2')
    hue = request.args.get('column3')
    bins = request.args.get('bins')

    plot_name = analysis.boxPlot(file, column1, column2, hue, bins)

    return render_template('result.html', plot=plot_name)


@app.route("/map-plot")
def mapPlot():

    file = request.args.get('file')
    lonlat = request.args.get('lonlat')
    plot_col = request.args.get('column2')
    countries = request.args.get('column3')

    plot_name = analysis.mapPlot(file, lonlat, countries, plot_col)

    return render_template('result.html', plot=plot_name)


@app.route("/gaussian-nb")
def gaussianNB():

    plot_name = machineLearning.gaussianNB()

    return render_template('result.html', plot=plot_name)


@app.route("/knn-clf")
def kNeighbours():

    plot_name = machineLearning.kNeighbours()

    return render_template('result.html', plot=plot_name)


@app.route("/regression")
def regression():

    plot_name = machineLearning.regression()

    return render_template('result.html', plot=plot_name)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
