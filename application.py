from flask import Flask
app_lulu = Flask(__name__)

@app.route('/hello_page')
def hello_world():
    # this is a comment, just like in Python
    # note that the function name and the route argument
    # do not need to be the same.
    return 'Hello world!'
    
if __name__ == '__main__':
#	app.run(port=33507)
       app.run()
