# RNN & LSTM Text Classification Task

This project demonstrates a text classification pipeline using Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) models. The pipeline processes text data, trains a classification model, and provides an API for predictions.

## Features

- Preprocessing of text data using tokenization, lemmatization, and stopword removal.
- Text classification using an LSTM-based neural network.
- API for real-time predictions using FastAPI.
- Model and tokenizer persistence for reuse.

## Project Structure

```
RNN & LSTM Task/
│
├── api.py                # FastAPI server for predictions
├── classifier.py         # Text preprocessing and classification pipeline
├── Notebook/
│   ├── notebook.ipynb    # Jupyter notebook for data exploration and model training
│   └── Data.csv          # Dataset used for training and testing
├── Objects/              # Directory for storing model and tokenizer objects
│   ├── lstm_classification_model.keras
│   ├── tokenizer.pkl
│   └── label_encoder.pkl
├── static/               # Static files for the API (e.g., HTML templates)
│   └── index.html
└── readme.md             # Project documentation
```

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Virtual environment (optional but recommended)

### Installation

1. Clone the repository:


2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download NLTK datasets:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

### Running the API

1. Start the FastAPI server:
   ```bash
   python api.py
   ```

2. Open your browser and navigate to:
   ```
   http://127.0.0.1:8000
   ```

3. You can use the web interface, or Use the `/predict` endpoint to classify text by sending a POST request with the following JSON payload:
   ```json
   {
       "text": "Your input text here"
   }
   ```
## Usage

### Training the Model

The `notebook.ipynb` file contains the steps for data exploration, preprocessing, and training the LSTM model. Open the notebook in Jupyter and follow the cells to train the model.

### Making Predictions

The `api.py` file provides an API endpoint for predictions. Use tools like Postman or cURL to send requests, or access the provided HTML interface.

### Example Request

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d '{"text": "Sample text for classification"}'
```

### Example Response

```json
{
    "prediction": "entertainment"
}
```

## Dependencies

- Python 3.8+
- TensorFlow
- FastAPI
- NLTK
- NumPy
- Pandas
- Uvicorn

## Acknowledgments

- **Dataset**: The dataset used for training is included in `Notebook/Data.csv`.
- **Libraries**: TensorFlow, NLTK, and FastAPI were used to build this project.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
