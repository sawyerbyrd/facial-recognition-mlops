To run localy:

1.) run this in one terminal to start the MLFlow server
    pip install -r requirements.txt
    mlflow server --host 0.0.0.0 --port 5000 &

2.) in a seperate terminal, run this to train:
    python3 train.py

To stop local server:

1.) use "fg" in terminal to bring the process to the foreground
        - Note: "&" started the process in the backround

2.) press ctrl + c to end the process