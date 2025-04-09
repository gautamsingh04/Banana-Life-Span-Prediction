# 🍌 Banana Life Span Prediction

A deep learning project to classify the ripeness of bananas and predict their remaining eatable days using image data and the MobileNetV2 model.

## 📁 Dataset

The dataset is included in this repository as a ZIP file: [`BANANA.zip`](./BANANA.zip).  
Unzip it and ensure the folder structure resembles:

```
BANANA/
├── unripe/
├── ripe/
└── overripe/
```

Each folder contains images representing a different ripeness stage.

## 🚀 Features

- 🧠 **Model**: MobileNetV2 for efficient and accurate banana ripeness classification.
- 🔁 **Data Augmentation**: Rotation, zoom, and flips improve generalization.
- 📈 **Training Visualization**: Training accuracy and loss are plotted using Matplotlib.
- 🗓️ **Custom Prediction Mapping**: Converts ripeness scores into estimated eatable days.
- 🧃 **Real-world Applicability**: Model is saved for use in future applications or deployments.

## 🛠️ Technologies Used

- Python
- TensorFlow & Keras
- Matplotlib
- NumPy

## 📦 Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/gautamsingh04/Banana-Life-Span-Prediction.git
   cd Banana-Life-Span-Prediction
   ```

2. **Extract Dataset**:
   Unzip `BANANA.zip` into a `BANANA/` folder in the root directory.

3. **Install Dependencies**:
   ```bash
   pip install tensorflow keras matplotlib numpy
   ```

4. **Run the Model**:
   ```bash
   python banana.py
   ```

## 📊 Model Performance

The model was trained on categorized banana images and fine-tuned for accuracy. Evaluation metrics and training curves are plotted to assess model performance.

## 🔮 Future Scope

- Add more classes or ripeness nuances.
- Build a web or mobile app for users to upload banana images and receive predictions.
- Integrate with smart fridges or IoT for automatic fruit monitoring.

## 🤝 Contributing

Feel free to fork the repo and submit pull requests for enhancements, bug fixes, or new features!

## 📄 License

MIT License
