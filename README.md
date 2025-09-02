# Skin_cancer_detection_using_cnn

This project contains:
- A Colab notebook (`skin_cancer_cnn.ipynb`) to train a CNN on the HAM10000 skin cancer dataset.
- A Streamlit app (`app.py`) to perform predictions on uploaded skin lesion images using the trained model.

## How to use

1. **Train the model:**
   - Run `skin_cancer_cnn.ipynb` in Colab.
   - At the end, save the trained model:
     ```python
     model.save("model.keras")
     ```
   - Download `model.keras` to your local machine.

2. **Run the Streamlit app:**
   - Place `model.keras` in the same directory as `app.py`.
   - Install dependencies:
     ```
     pip install -r requirements.txt
     ```
   - Start app:
     ```
     streamlit run app.py
     ```

## Requirements

See `requirements.txt`
