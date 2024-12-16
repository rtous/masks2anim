# masks2anim

## Setup

Repo:
```
git clone https://github.com/rtous/masks2anim.git
cd masks2anim
```

Virtual env:
```
python3.11 -m venv myvenv
source myvenv/bin/activate
```

Dependencies:
```
pip install matplotlib==3.9.2
pip install opencv-python==3.4.17.61
pip install shapely==2.0.2
pip install numpy==1.26.0
pip install imutils==0.5.4
pip install dlib==19.24.6
```

Download into models folder the model file of the dlib's HOG face detector (shape_predictor_68_face_landmarks.dat):
```
mkdir models

wget https://github.com/italojs/facial-landmarks-recognition/raw/refs/heads/master/shape_predictor_68_face_landmarks.dat -P models
```

## Test

python test.py