import analysis
from flask import Flask, session, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
import json
import matplotlib
matplotlib.use('Agg')

print("BAR CHART: http://127.0.0.1:5000/bar-chart?file=globalterrorism.csv&column=iyear&sort=0")
print("LINE CHART: http://127.0.0.1:5000/line-chart?file=globalterrorism.csv&year=iyear&group=region_txt")

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
            return render_template('home.html', columns=columns, filename=filename)

    return render_template('upload.html')


@ app.route("/bar-chart")
def barChart():

    file = request.args.get('file')
    column1 = request.args.get('column1')
    sort = request.args.get('sort')
    bars = request.args.get('bars')

    plot_name = analysis.barChart(file, column1, sort, bars)

    return render_template('index.html', plot=plot_name)


@ app.route("/line-chart")
def lineChart():

    file = request.args.get('file')
    time = request.args.get('column1')
    col2 = request.args.get('column2')
    col3 = request.args.get('column3')
    col4 = request.args.get('column4')
    bins = request.args.get('bins')

    plot_name = analysis.lineChart(file, time, col2, col3, col4, bins)

    return render_template('index.html', plot=plot_name)


@ app.route("/scatter-plot")
def scatterPlot():

    file = request.args.get('file')
    column1 = request.args.get('column1')
    column2 = request.args.get('column2')
    bins = request.args.get('bins')

    plot_name = analysis.scatterPlot(file, column1, column2, bins)

    return render_template('index.html', plot=plot_name)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
