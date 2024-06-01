

# Fake News Prediction Model

This repository contains a machine learning model for predicting fake news. The model is trained on the dataset provided by the Kaggle Fake News competition.

## Overview

The goal of this project is to build a classifier that can accurately distinguish between real and fake news articles. The dataset used for training and testing the model is from the Kaggle Fake News competition.

## Dataset

The dataset consists of news articles labeled as fake or real. It includes the following columns:
- `id`: Unique identifier for each news article.
- `title`: The title of the news article.
- `author`: The author of the news article.
- `text`: The main text content of the news article.
- `label`: The label for the news article (1 indicates fake news, 0 indicates real news).

The dataset can be downloaded from [Kaggle Fake News Dataset](https://www.kaggle.com/c/fake-news/data?select=train.csv).

## Model Performance

The model achieves the following performance:
- **Training Accuracy**: 98.65%
- **Testing Accuracy**: 97.9%

## Installation

To run the code in this repository, you need to have Python installed along with the required libraries. You can install the required libraries using the following command:

```sh
pip install -r requirements.txt
```

### Requirements

- pandas
- numpy
- scikit-learn
- nltk

## Usage

1. **Download the dataset**: Download the dataset from the Kaggle link provided above and place it in the project directory.

2. **Preprocess the data**: Run the preprocessing script to clean and prepare the data for training.

3. **Train the model**: Train the model using the training dataset.

4. **Evaluate the model**: Evaluate the model using the testing dataset to check its performance.

### Preprocessing

The `preprocessing.py` script performs the following tasks:
- Remove non-alphabetic characters.
- Convert text to lowercase.
- Remove stopwords.
- Apply stemming to the words.

### Training

The `train.py` script trains the model using the preprocessed training dataset. The model uses logistic regression as the classifier.

### Evaluation

The `evaluate.py` script evaluates the trained model using the testing dataset and outputs the accuracy and other performance metrics.

## Code Structure

```sh
.
├── data
│   └── train.csv
├── preprocessing.py
├── train.py
├── evaluate.py
├── requirements.txt
└── README.md
```

## Example

Here is a basic example of how to use the code in this repository:

1. **Preprocess the data:**

```python
# preprocessing.py
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

nltk.download('stopwords')

port_stem = PorterStemmer()

def stemming(content):
    content = re.sub('[^a-zA-Z]', ' ', content)
    content = content.lower()
    content = content.split()
    content = [port_stem.stem(word) for word in content if not word in stopwords.words('english')]
    content = ' '.join(content)
    return content

df = pd.read_csv('data/train.csv')
df['text'] = df['text'].apply(stemming)
df.to_csv('data/train_preprocessed.csv', index=False)
```

2. **Train the model:**

```python
# train.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/train_preprocessed.csv')
X = df['text']
y = df['label']

tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X = tfidf_vectorizer.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

print(f"Training Accuracy: {model.score(X_train, y_train) * 100:.2f}%")
print(f"Validation Accuracy: {model.score(X_val, y_val) * 100:.2f}%")
```

3. **Evaluate the model:**

```python
# evaluate.py
import pandas as pd
from sklearn.metrics import accuracy_score

# Load the preprocessed test data
test_df = pd.read_csv('data/test_preprocessed.csv')
X_test = test_df['text']
y_test = test_df['label']

# Transform the test data using the fitted TF-IDF vectorizer
X_test = tfidf_vectorizer.transform(X_test)

# Predict and evaluate
y_pred = model.predict(X_test)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Kaggle Fake News Dataset](https://www.kaggle.com/c/fake-news/data?select=train.csv)
- [NLTK Library](https://www.nltk.org/)
- [Scikit-learn Library](https://scikit-learn.org/)

---
```


