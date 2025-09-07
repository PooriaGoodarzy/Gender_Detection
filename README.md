# Gender Detection 👤⚧️

This project is a deep learning model for **predicting the gender from a face image**.  
The model is trained on the **[UTKFace dataset](https://susanqq.github.io/UTKFace/)** and can make predictions from a test image.

---

## 📂 Project Structure

```
gender_detection/
│── train_model.py # training code (optional if you want to retrain)
│── Live test_model.py # predict gender from your camera
│── requirements.txt # dependencies
│── gender_weights.h5 # pre-trained model weights
│── README.md
```

---

## Installation
```bash
git clone https://github.com/your-username/gender_detection.git
cd gender_detection
pip install -r requirements.txt
```

---

##Usage
- **Train (optional):**`python train_model.py` → only if you want to retrain the model
- **Test:** `python Live test_model.py` → uses the pre-trained weights `gender_weights.h5`
and run the code.

---

## Dataset
Download the **UTKFace dataset** (for training) and place it in:
```
gender_detection/UTKFace_dataset/
```

---

## Notes
- Adjust file/folder paths in the code if you rename anything.
- Training may take time depending on your hardware (GPU recommended).
- The training model has low accuracy. If you pay attention, it is better to increase the epochs in the train_model.py file and run the code.