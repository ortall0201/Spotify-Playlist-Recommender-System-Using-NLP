# 🎵 Spotify Playlist Recommender System Using NLP

**Overview**
This project leverages Natural Language Processing (NLP) techniques to develop a personalized playlist recommender system. By analyzing playlist data from Spotify, the system identifies contextual relationships between songs and generates tailored recommendations based on user preferences.

  ![Spotify](Presentation/spotify.png)
_________________________________________

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model](#model)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

____________________________________________________________________________
## Features

- Artist-Based Recommendations: Focuses on artist similarity using fuzzy matching techniques to handle inconsistencies in naming conventions.
- Word2Vec Embeddings: Generates song embeddings to represent relationships within playlists.
- Hard Negative Sampling: Enhances model training by focusing on challenging negative examples.
- Evaluation Metrics: Includes Precision@K, Recall@K, and Mean Average Precision (MAP) to assess performance.
- Cosine Similarity: Used to compute the similarity between song embeddings for accurate recommendations.
- Cleaning and preprocessing of Spotify playlist data to handle missing values, inconsistencies, and special characters.
- Visualization of key insights, such as top artists, track distributions, and playlist diversity.

_____________________________________________________________________________________________________________________

## Installation
**Note**: The commands below should be run in the terminal (Command Prompt, Bash, or Shell), not inside Jupyter Notebook.

1. Clone this repository:
```bash
git clone https://github.com/ortall0201/Spotify-Playlist-Recommender-System-Using-NLP.git
cd Spotify-Playlist-Recommender-System-Using-NLP

3. Install dependencies:
```bash
pip install -r requirements.txt

4. (Optional) Install Jupyter Notebook to explore the project interactively:
```bash
pip install notebook

_____________________________________________________________________________________________________________________
## Technologies Used

Python: Data analysis and machine learning.
pandas, NumPy: Data manipulation and preprocessing.
Matplotlib, Seaborn: Data visualization.
Gensim, SpaCy, or Hugging Face: NLP modeling.
scikit-learn: Recommender system development.
____________________________________________________________________________________________________________________
## Dataset

Name of the dataset: Spotify Playlists-
Type of the dataset: 1.2GB of tabular data for music recommendation

The project uses the Spotify Playlists Dataset, which includes detailed information on user-generated playlists, artists, tracks, and more. This dataset is processed and analyzed in compliance with its terms of use.

You have 2 options: 
1. Download the dataset from Kaggle: https://www.kaggle.com/datasets/andrewmvd/spotify-playlists/data?select=spotify_dataset.csv
2. Use direct public links to the dataset in the project, without downloading.

_________________________________________________________________________________________________________________
## Goals
Extract meaningful insights from Spotify playlist data.
Build a scalable and effective music recommender system using NLP.
Explore the potential of embedding techniques for playlist and track analysis.
_____________________________________________________________________________________________________________________
## Future Work
Incorporate advanced embeddings such as BERT-based models for contextual track recommendations.
Add user-specific personalization for playlist generation.
Extend the analysis to include audio features like tempo, key, and energy.

