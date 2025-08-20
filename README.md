# postureAI
Posture detection AI (CLI) that reminds you when you are slouching

1) Create virtual env and install:
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

2) Prepare training data:
   python data_collection.py

3) Train classifier with the data:
   python train.py

4) Launch the main Flask Application:
   python app.py

5) Open in browser:
   Open web browser and navigate to http://127.0.0.1:5000

6) Press 'q' in the video window to quit.


