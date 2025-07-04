import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load trained model
model = load_model('my_emotion_model.h5')

# Emotion labels used in training
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load image (change name if using a different file)
img = cv2.imread('test_image2.jpeg')

# Convert to grayscale (model was trained on grayscale images)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Use OpenCV's built-in face detector
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Detect faces in the image
faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

for (x, y, w, h) in faces:
    roi_gray = gray[y:y+h, x:x+w]
    roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

    if np.sum([roi_gray]) != 0:
        roi = roi_gray.astype('float') / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # Make prediction
        prediction = model.predict(roi)[0]
        label = emotion_labels[np.argmax(prediction)]

        # Draw results on the image
        cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)
    else:
        cv2.putText(img, "No Face Found", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

# Show the image
cv2.imshow("Emotion Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
