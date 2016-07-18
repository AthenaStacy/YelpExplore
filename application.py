from flask import Flask,render_template,request,redirect
app = Flask(__name__)

@app.route('/hello_page')
def hello_world():
    # this is a comment, just like in Python
    # note that the function name and the route argument
    # do not need to be the same.
    return 'Hello world!'

@app.route('/figure1')
def uploaded_file():
    filename = 'n_vs_date.png'
    return render_template('graph1.html', filename=filename)

@app.route('/figure2')
def uploaded_file2():
    filename = 'barchart.png'
    return render_template('graph2.html', filename=filename)

    
if __name__ == '__main__':
	app.run(port=33507)
#       app.run()
