🐶 Dog vs 🐱 Cat Classifier — CNN Binary Classification


Built from scratch. Trained on pixels. Deployed with love.


What is this?

A deep Convolutional Neural Network that looks at an image and tells you — is that a dog or a cat? Simple question. Not-so-simple under the hood.

This was my first CNN project — built entirely from scratch using Keras, trained on thousands of images, and wrapped into a clean Streamlit web app where you just drag, drop, and get an answer in seconds.

No pretrained weights. No transfer learning shortcuts. Just raw architecture, backpropagation, and patience.


Demo

Upload an ImageGet the PredictionDrag & drop any dog or cat imageModel returns class + confidence %


Cat uploaded → CAT — 99.96% confidence ✅

Dog uploaded → DOG — model knows.




Model Architecture

Three convolutional blocks stacked progressively — each one learning to see more complex features than the last.

Input Image (256 × 256 × 3)
        │
┌───────▼────────┐
│  Conv2D (32)   │  ← edges, colors, basic textures
│  BatchNorm     │
│  ReLU          │
│  MaxPooling    │
└───────┬────────┘
        │
┌───────▼────────┐
│  Conv2D (64)   │  ← fur patterns, eye shapes, snout outlines
│  BatchNorm     │
│  ReLU          │
│  MaxPooling    │
└───────┬────────┘
        │
┌───────▼────────┐
│  Conv2D (128)  │  ← complex structures — ears, whiskers, paws
│  ReLU          │
│  MaxPooling    │
└───────┬────────┘
        │
   Flatten (131,072)
        │
  Dense(128) + Dropout(0.2)
  Dense(64)  + Dropout(0.1)
  Dense(1, sigmoid)          ← binary output: 0 = Cat, 1 = Dog

Total Parameters: 16,879,297 (~64.39 MB)

Trainable Parameters: 16,879,105


Training Details

ParameterValueInput Shape256 × 256 × 3OptimizerAdamLoss FunctionBinary CrossentropyEpochs50 (Early Stopping triggered at Epoch 9)Batch Size32Best Epoch4

Training Log (Key Epochs)

EpochTrain AccuracyVal Accuracy162.66%60.02%372.47%71.90%579.47%74.72%783.76%74.98%9 (stopped)87.25%74.18%


Early stopping restored weights from Epoch 4 — the point of peak generalization before overfitting started kicking in.



Accuracy Curve

The training accuracy climbed steadily while validation accuracy plateaued around 74–75% — a classic sign of a model that's learning real features but starting to memorize beyond that. Early stopping did its job.


Tech Stack

ToolPurposePythonCore languageTensorFlow / KerasModel building & trainingNumPyArray operationsMatplotlibAccuracy/loss visualizationStreamlitWeb app deploymentGoogle ColabTraining environment


Project Structure

dog-vs-cat-classifier/
│
├── dog_vs_cat_classifier.ipynb   # Full training notebook (Colab)
├── app.py                         # Streamlit web app
├── model/
│   └── dog_cat_model.h5           # Saved trained model
├── sample_images/
│   ├── cat.jpg
│   └── dog.jpg
└── README.md


How to Run the App

1. Clone the repo

bashgit clone https://github.com/RathodKrrish/dog-vs-cat-classifier.git
cd dog-vs-cat-classifier

2. Install dependencies

bashpip install tensorflow streamlit numpy pillow matplotlib

3. Launch the Streamlit app

bashstreamlit run app.py

4. Open in browser

Go to http://localhost:8501 — upload any image of a dog or cat and hit Predict.


How the Streamlit App Works


Upload a JPG / JPEG / PNG image (max 200MB)
App resizes it to 256 × 256 and normalizes pixel values
Model runs a forward pass and outputs a sigmoid probability
If probability > 0.5 → 🐶 DOG
If probability ≤ 0.5 → 🐱 CAT
Confidence score is displayed as a percentage



Results


Best Validation Accuracy: ~75%
Cat prediction on test image: 99.96% confidence
Model handles clear, centered images of dogs and cats well



What I Learned

This project wasn't just about accuracy numbers. It was about understanding why things work the way they do.


How Conv2D + MaxPooling progressively compresses spatial information while extracting features
Why BatchNormalization stabilizes training and speeds convergence
How Dropout acts as a regularizer — randomly killing neurons so the model doesn't lean on any single path
Why early stopping is smarter than just running all 50 epochs and hoping for the best
What the train vs. val accuracy gap actually means in practice



What's Next


 Try Transfer Learning with MobileNetV2 or EfficientNetB0 for better accuracy
 Add Grad-CAM visualization — show where in the image the model is looking
 Deploy on Hugging Face Spaces for public access
 Experiment with data augmentation to reduce overfitting



Author
Krish Rathod
B.Sc. AI & Machine Learning | Aspiring Data Scientist


"First CNN. Won't be the last."
