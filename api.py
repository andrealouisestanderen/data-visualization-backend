from flask import Flask, request, render_template
import os
import matplotlib
matplotlib.use('Agg')
import analysis

print("BAR CHART: http://127.0.0.1:5000/bar-chart?file=globalterrorism.csv&column=iyear&sort=0")
print("LINE CHART: http://127.0.0.1:5000/line-chart?file=globalterrorism.csv&year=iyear&group=region_txt")

app = Flask(__name__)

@app.route("/")
def hello():
    return render_template('index_test.html')

@app.route("/bar-chart")
def barChart():

    file = request.args.get('file')
    column = request.args.get('column')
    sort = request.args.get('sort')
    
    plot_name = analysis.barChart(file, column, sort)

    return render_template('index.html', plot=plot_name)


@app.route("/line-chart")
def lineChart():

    file = request.args.get('file')
    year = request.args.get('year')
    group = request.args.get('group')
    
    plot_name = analysis.lineChart(file, year, group)

    return render_template('index.html', plot=plot_name)