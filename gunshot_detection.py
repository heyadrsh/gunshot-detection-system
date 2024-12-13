import os
import numpy as np
import librosa
import sounddevice as sd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import threading
import queue
import time

class GunshotDetector:
    def __init__(self):
        self.sample_rate = 44100
        self.duration = 2  # seconds
        self.model = self._build_model()
        self.classes = ['AK-12', 'AK-47', 'IMI Desert Eagle', 'M16', 'M249', 'M4', 'MG-42', 'MP5', 'Zastava M92']
        self.audio_queue = queue.Queue()
        
    def _build_model(self):
        model = models.Sequential([
            layers.Input(shape=(128, 87, 1)),  # Mel spectrogram shape
            layers.Conv2D(32, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(len(self.classes), activation='softmax')
        ])
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        return model

    def extract_features(self, audio_path=None, audio_data=None):
        if audio_path:
            audio, _ = librosa.load(audio_path, sr=self.sample_rate, duration=self.duration)
        else:
            audio = audio_data
            
        if len(audio) < self.sample_rate * self.duration:
            audio = np.pad(audio, (0, self.sample_rate * self.duration - len(audio)))
            
        mel_spec = librosa.feature.melspectrogram(y=audio, 
                                                sr=self.sample_rate,
                                                n_mels=128,
                                                fmax=8000)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db.reshape(128, 87, 1)  # Reshape for CNN input

    def prepare_dataset(self):
        X = []
        y = []
        
        for idx, class_name in enumerate(tqdm(self.classes, desc="Loading dataset")):
            class_path = os.path.join('dataset', class_name)
            for audio_file in os.listdir(class_path):
                if audio_file.endswith('.wav'):  # Adjust based on your audio format
                    file_path = os.path.join(class_path, audio_file)
                    features = self.extract_features(audio_path=file_path)
                    X.append(features)
                    y.append(idx)
        
        X = np.array(X)
        y = np.array(y)
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train(self):
        print("Preparing dataset...")
        X_train, X_test, y_train, y_test = self.prepare_dataset()
        
        print("Training model...")
        self.model.fit(X_train, y_train,
                      epochs=20,
                      batch_size=32,
                      validation_data=(X_test, y_test))
        
        self.model.save('gunshot_model.h5')
        print("Model saved as 'gunshot_model.h5'")

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Status: {status}")
        self.audio_queue.put(indata[:, 0])

    def process_audio_stream(self):
        while True:
            audio_chunk = self.audio_queue.get()
            features = self.extract_features(audio_data=audio_chunk)
            prediction = self.model.predict(np.expand_dims(features, axis=0), verbose=0)
            confidence = np.max(prediction)
            if confidence > 0.8:  # Confidence threshold
                weapon_type = self.classes[np.argmax(prediction)]
                print(f"Detected: {weapon_type} (Confidence: {confidence:.2f})")

    def start_detection(self):
        try:
            print("Loading model...")
            self.model = models.load_model('gunshot_model.h5')
        except:
            print("No trained model found. Please train the model first.")
            return

        print("Starting real-time detection... Press Ctrl+C to stop.")
        processing_thread = threading.Thread(target=self.process_audio_stream)
        processing_thread.daemon = True
        processing_thread.start()

        with sd.InputStream(channels=1,
                          samplerate=self.sample_rate,
                          blocksize=int(self.sample_rate * self.duration),
                          callback=self.audio_callback):
            while True:
                time.sleep(0.1)

if __name__ == "__main__":
    detector = GunshotDetector()
    
    while True:
        choice = input("Enter 1 to train model, 2 to start detection, or q to quit: ")
        if choice == '1':
            detector.train()
        elif choice == '2':
            detector.start_detection()
        elif choice.lower() == 'q':
            break 