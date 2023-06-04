# Pee Wee Text-To-Speech

### Description

Pee Wee TTS is a text-to-speech model designed to mimic the speech of the character Pee Wee Herman, using a general-purpose voice cloning algorithm. The approach can be found in more detail here: [Transfer Learning from Speaker Verification](https://arxiv.org/pdf/1806.04558.pdf). The architecture contains a synthesizer, which is responsible for converting the text to speech, with a speaker embedding inserted at each time step in the attention layer. This speaker embedding encodes the information and inflections pertaining to each individual speaker, and models the difference between them. With this model, it is possible to genenerate speech that sounds like a particular character, using a small sample of recorded speech (~30s-1:30s) as reference. The speaker does not have to be included in the training set.

### Speaker Embeddings

The speaker embeddings are generated from a model that is trained independently, and describes the relationships between different speakers. The approach can be found here: [Generalized End-To-End Loss For Speaker Verification](https://arxiv.org/pdf/1710.10467.pdf). The distance between each utterance and the centroid of the speaker, in embedding space, is minimized. The distance between each utterance and the centroids of other speakers is maximized. Here are the results of the latest training loop for the speaker embedding model:
![encoder_loss](https://github.com/liambarstad/peewee-tts/assets/25653466/18bc754f-bf10-4320-b7bc-443e11d589d2)

It should be noted that new work has improved the modeling of speaker embedding space, and a more power model can be substituted

### Synthesizer

The synthesizer, or the model that correlates the text and speech itself, uses a Tacotron 2 architecture. The paper for Tacotron 2 can be found here: [Natural TTS Synthesis By Conditioning Wavenet On Mel Spectrogram](https://arxiv.org/pdf/1712.05884.pdf). At each time step in the attention layer, the encoded text is concatenated with the speaker embeddings, i.e. the predictions from the trained speaker embedding model.

### Installation

To install this repo locally, first clone the repository and enter the directory in the terminal.
If you have not already, install anaconda and create a new python environment and activate it

    conda create --name peeweetts python=3.8
    conda activate peeweetts

Navigate here to download the dataset: https://www.openslr.org/60/. This is the LibriTTS dataset, which consists of speech snippets of various speakers reading lines from books, organized by speaker and chapter. Unzip the data zip file of choice, and add it to the /data folder in the project directory. You can change the location of the data in the /config section, which contains the list of hyperparameters used for training. In order for the synthesizer to converge, it is also recommended to utilize the following datasets: [VCTK](https://datashare.ed.ac.uk/handle/10283/2950), [VoxCeleb2](https://paperswithcode.com/dataset/voxceleb2), [RyanSpeech](https://arxiv.org/pdf/2106.08468.pdf). However, this repository does not currently support these datasets

This project uses [MLFlow](https://mlflow.org/), which is used for model management, deployment, training scripts, and parameter logging. MLFlow constructs a conda environment to containerize and package models. In order to run locally, you will need to install the dependencies in conda.yaml on your local machine:

    conda env update -n peeweetts --file conda.yaml

MlFlow should now be installed automatically. In the terminal, you can now run the ```mlflow``` command to run training jobs, access the server, and build docker images. Please refer to the MLFlow documentation for more information.
The list of available scripts to run can be found in ```MLProject```. There is a script to train the speaker embedding model:

    mlflow run . -e train_encoder # uses the default configuration file config/speaker_recog_encoder_dev.yml
    mlflow run . -e train_encoder -P config_path=config/speaker_recog_encoder_prod.yaml # prod environment, typically a larger version of the data
    mlflow run . -e train_encoder -P save_model=true # saves the model, which can be accessed in the mlflow server
    
After the encoder model is saved, you will need to register the model in MLFlow, and edit the appropriate value in config/tacotron_2_dev.yml or config/tacotron_2_prod.yaml, so that the TT2 training script can pull the trained model. Then, similarly, you can run:

    mlflow run . -e train_tacotron_2
    mlflow run . -e train_tacotron_2 -P config_path=config/tacotron_2_prod.yml
    mlflow run . -e train_tacotron_2 -P config_path=config/tacotron_2_prod.yaml -P save_model=true
    
In order to compile a Docker image with the trained models, refer to the MLFlow documentation and delete the app/ directory. The current serving architecture that I have set up does not utilize this feature, but it is usually better to deploy in a microservice locally, or in the cloud. MLFlow offers native support for almost all major cloud services.

Here are some of my references:
https://github.com/CorentinJ/Real-Time-Voice-Cloning : Great repo that uses the original Tacotron architecture with the speaker embedding model
https://github.com/NVIDIA/tacotron2 : Vanilla implementation of Tacotron 2 without the insertion of speaker embeddings
