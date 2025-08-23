# postureAI
Posture detection AI (CLI) that reminds you when you are slouching

# Usage steps:

### 1) Create virtual env and install:
   python -m venv venv  
   source venv/bin/activate  
   pip install -r requirements.txt  

### 2) Prepare training data:
   python data_collection.py  
   Press 'g' and hold good posture for around 20-30 seconds, then press 'b' and hold bad posture for around 20-30 seconds. Press q to quit. The data will be saved into a csv file.   

### 2.1) Camera index check:
   If there is a 'Failed to access camera' error, there may be a problem with privacy settings or camera index. Ensure that the environment you are running this script in has access to your local webcam. If the issue still    persists, run 'python camera.py' to check if there is another camera index to use. If so, replace the '0' on line 11 in 'data_collection.py' and line 17 in 'app.py' with the correct index and run the data collection        script again. 

### 3) Train classifier with the data:
   python train.py

### 4) Launch the main Application:
   python app.py  
   the Posture detection algorithm should be running!

### 5) Press 'q' in the video window to quit.



