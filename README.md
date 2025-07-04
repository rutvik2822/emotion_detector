
# Emotion Detection using Deep Learning 🎭

This project uses Convolutional Neural Networks (CNNs) to detect emotions from human facial expressions using the FER2013 dataset.

## 📁 Project Structure
- `2_train_model.py`: Trains the CNN model using preprocessed FER2013 data.
- `3_test_on_image.py`: Predicts emotions from static image files.
- `4_test_on_webcam.py`: Detects emotions in real-time using your webcam.
- `my_emotion_model.h5`: Trained model file (if not too large to upload).
- `requirements.txt`: Python packages used in this project.
- `.gitignore`: Excludes unnecessary folders like `venv/`.

## 💡 Emotions Recognized
- 😄 Happy
- 😢 Sad
- 😠 Angry
- 😱 Fear
- 😐 Neutral
- 😲 Surprise
- 🤢 Disgust

## ⚙️ Requirements

Install required dependencies:
```bash
pip install -r requirements.txt
```

## 🧪 Run Emotion Detection

**To train your own model:**
```bash
python 2_train_model.py
```

**To test with image:**
```bash
python 3_test_on_image.py
```

**To test with webcam:**
```bash
python 4_test_on_webcam.py
```

## 📌 Notes
- FER2013 dataset is required in a folder structure with `train/` and `test/` subfolders.
- Model performance is for educational purposes and may vary.

## 🙌 Project By
**Rutvik Devdare**  
B.Tech IT Student, D.Y. Patil University  
GitHub: [rutvik2822](https://github.com/rutvik2822)
