import os
from flask import Flask, render_template, request
from flask_wtf.csrf import CSRFProtect

app = Flask(__name__, static_url_path='/static')
csrf = CSRFProtect(app) 
csrf.init_app(app)
app.secret_key = os.environ['WTF_CSRF_SECRET_KEY']
app.config['SESSION_TYPE'] = 'filesystem'

@app.route('/', methods=['GET'])
def main():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    if 'input_text' not in request.form:
        return '', 400
    input_text = request.form['input_text']
    if len(input_text > 100):
        return '', 414
    import ipdb; ipdb.sset_trace()

if __name__ == '__main__':
    app.run(port=8080, debug=True)
