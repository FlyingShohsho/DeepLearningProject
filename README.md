# Predicting the Olympic Medal Table - ECE046211 Final Project - Spring 2024

This project aims to simplify and enhance the accuracy of predicting the number of Olympic medals per country. We utilize a transformer model for feature extraction and build our predictions based on various features derived from historical data. The project is implemented in Python using the libraries PyTorch, NumPy, Pandas and SciKit-Learn.
<p align="middle">
  <img src="./assets/Paris2024.png" height="200">
</p>

## Table of Contents
- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Data](#data)
- [Model](#model)
- [Results](#results)
- [Usage](#usage)
- [License](#license)
- [Authors](#authors)

## Project Overview

The task of predicting the Olympic medal table is complex due to the numerous factors involved. In this project, we leverage a transformer model to extract meaningful features and predict the number of medals that each country will win. The project is part of the course ECE046211 - Spring 2024, and focuses on making the prediction process more efficient and accurate.

## Project Structure
/notebooks

    DeepLearningProject_Shoham_Liad_Final.ipynb # Main notebook containing code and analysis

/data # Folder where data is stored
/models # Folder where trained models are saved
/scripts # Python scripts for data processing, model training, etc.
README.md # Project documentation
requirements.txt # Python dependencies


## Installation

To set up this project locally, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/olympic-medal-prediction.git
    cd olympic-medal-prediction
    ```

2. **Create a virtual environment** (optional but recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Download the data** (if applicable):
    - Instructions for downloading and placing the data into the `/data` directory.

### Prerequisites
|Library         | Version |
|--------------------|----|
|`Python`| `3.6.13 (Anaconda3)`|
|`matplotlib`| `3.7.4`|
|`numpy`| `1.26.4`|
|`optuna`| `3.6.1`|
|`scikit_learn`| `1.3.2`|
|`torch`| `2.3.1+cu121`|
|`transformers`| `4.42.4`|
|`torchvision`| `0.11.3`|
|`tqdm`| `4.64.1`|
|`Wikipedia-API`| `0.6.0`|
## Data

The data used in this project includes historical Olympic records, country demographics, economic indicators, and other relevant features. It is assumed that the user will download and preprocess the data following the instructions provided in the notebook.

### Data Preprocessing
- Data cleaning, normalization, and feature extraction steps are detailed in the notebook.
- Ensure that the data is correctly formatted and placed in the `/data` directory before running any scripts or notebooks.

## Model

We employed a transformer model to extract features from the input data. The model was trained using PyTorch, and the following key components were used:

- **Model Architecture**: Transformer-based architecture for feature extraction.
- **Training**: The model was trained on the preprocessed dataset, with key hyperparameters like batch size, learning rate, and the number of epochs.
- **Evaluation**: The model was evaluated using accuracy, mean squared error, and confusion matrices.

### Key Hyperparameters
- Batch size: [batch size]
- Learning rate: [learning rate]
- Number of epochs: [number of epochs]

## Results

We predicted 5 different metrics: Gold medals, Silver Medals, Bronze Medals, Predicted Total(a prediction based on previous total tallies), and Total Predicted(the sum of the separate colour predictions). Naturally, the total predicted and the predicted total were very similar, but predicting the total straightforwardly did slightly better than predicting medals separately and adding them up. This might be due to the fact that predicting a combined number gives more margin of error and doesn't account for the internal composition.

We used two approaches for training and testing our model: the first was to predict for all the countries which participated in Paris and the second was to predict only for countries which had won more than eight medals across the past 6 Olympics(since Sydney 2000), in order to avoid countries(E.G. Soviet and Eugoslavic countries) that have been separated/formed along the years, and also that our generated features are from current up-to-date data which might be unrelated to results from 50+ years ago.

Surprisingly(but also not, since it provides much more data to learn from and more countries to predict), the first approach did far better, as we can see:

|Name of the Model|Accuracy|
|:---:|:---:|
|Logistic Regression|85.357825%|
|MultinomialNB|85.367968%|
|Decision Tree|84.095457%|
|Random Forest|88.779773%|
|Gradient Boosting|88.514205%|
|Neural Network|87.904691%|
|NN+BERT(full, total predicted)|91.66%|
|NN+BERT(full, predicted total)|92.156%|
|NN+BERT(recency filter)|77.142%|

Where the previous model results are taken from [Olympics-Medal-Prediction](https://github.com/hrugved06/Olympics-Medal-Prediction)
The results indicate that the transformer-based approach is effective in predicting the number of medals for each country.
In addition to the improved accuracy, the biggest advantage of our approach is the simplicity of data handling compared to previous work. All we needed to do was use BERT to automatically generate hundreds of features(768, to be exact), whereas previous involved heavy data cleaning and handling and merging multiple huge datasets, which requires great amounts of RAM, in order to get a fraction of our features. When we tried running code from [Olympics 2024 predictions](https://www.kaggle.com/code/asfefdgrg/olympics-2024-predictions#2.-Identifying-and-Gathering-Data-Sources) for comparison, it reached the Colab limit 12GB RAM very quickly while merging dataframes, which is not viable.

Of course, the disadvantage of our approach is that the features BERT generates are a latent space, and so our work can't be used easily for data analysis tasks because we don't have access to the actual meaning of each feature.
## Usage

To use the model for predictions:

1. **Ensure that the data is in place and preprocessed.**

2. **Run the notebook**: Open the `DeepLearningProject_Shoham_Liad_Final.ipynb` and follow the instructions to generate predictions.

## Future Work

While working on this project, we had a few ideas for future work:
1. Since the Olympics happen continuousely every 4 years, we can treat the results as sequential data and utilize an RNN to make predictions.
2. We had BERT fetch features from Wikipedia API. As LLM's grow and develop, it might be beneficial to use one instead. We can also build a custom transformer model to get better results.
3. Dimensionality reduction can be used(e.g. PCA) to reduce the number of features while emphasizing the dominant ones
4. We only predicted for Summer Olympics. There's a Winter Olympics coming up in 2026 :)
5. It can be interesting to build a model for specific athletes, rather than countries. This will be more complex and less suitable for our approach since there is naturally less data available about individual athletes than countries, both for feature extraction and for deep learning.

## Sources & References
### Sources
* The datasets were taken from Kaggle: [Olympic Historical Dataset From Olympedia.org](https://www.kaggle.com/datasets/josephcheng123456/olympic-historical-dataset-from-olympediaorg/data), [Paris 2024 Olympic Summer Games](https://www.kaggle.com/datasets/piterfm/paris-2024-olympic-summer-games).
* The Neural Network and Optuna implementation is taken from [ECE 046211 Tutorials](https://github.com/taldatech/ee046211-deep-learning)
* The project aims to build on a previous method review with a new approach: [Olympics-Medal-Prediction](https://github.com/hrugved06/Olympics-Medal-Prediction)
* We also found a project which was done at the same time as ours, and aimed to beat it with our approach: [Olympics 2024 predictions](https://www.kaggle.com/code/asfefdgrg/olympics-2024-predictions)
* With our deepest apologies to all Elementary school teachers out there... [Wikipedia](https://www.wikipedia.org/)
### Refrences
* [Attention is All You Need](https://arxiv.org/abs/1706.03762)
* [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
* [google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased?text=The+goal+of+life+is+%5BMASK%5D)

## Authors

- **Shoham Grunblat** - [GitHub](https://github.com/FlyingShosho) | [LinkedIn](https://www.linkedin.com/in/shoham-grunblat/)
- **Liad Mordechai** - [GitHub](https://github.com/liadMor123) | [LinkedIn](https://www.linkedin.com/in/liad-mordechai/)
- ECE046211 - Spring 2024


## Acknowledgments

This project is a part of the ECE 046211 Deep Learning course at the Technion. We would like to express our gratitude to Lior Friedman and Prof. Yossi Keshet for their guidance and support throughout this project and the course.

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
