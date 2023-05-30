import sys
import os
import librosa
from flask import Flask, render_template, request, send_file
from flask_wtf.csrf import CSRFProtect
from utils import load_model
import soundfile as sf
sys.path.append('.')
import predict

app = Flask(__name__, static_url_path='/static')
csrf = CSRFProtect(app) 
csrf.init_app(app)
app.secret_key = os.environ['WTF_CSRF_SECRET_KEY']
app.config['SESSION_TYPE'] = 'filesystem'

sp_encoder = load_model('peewee-tts-models', 'speaker_embedding_model.pth')
tt2_model = load_model('peewee-tts-models', 'tacotron_2_model.pth')

sp_embeds = predict.get_speaker_embedding(sp_encoder, [
    'app/static/pee_wee_ba_25.wav',
    'app/static/pee_wee_ba_51.wav',
    'app/static/pee_wee_ba_62.wav'
])

@app.route('/', methods=['GET'])
def main():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    if 'input_text' not in request.form:
        return '', 400
    input_text = request.form['input_text']
    if len(input_text) > 100:
        return '', 414
    prediction = predict.predict(input_text, sp_embeds, tt2_model)
    sf.write('app/temp/query.wav', prediction, 22050)
    return send_file('temp/query.wav', mimetype='audio/wav')

if __name__ == '__main__':
    app.run(port=8080, debug=True)
