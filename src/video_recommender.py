import json
import sys
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import contractions
from src.exception import CustomException

stop_words = nltk.corpus.stopwords.words('english')


class VideoRecommender:
    def __init__(self, json_file):
        # Load and parse data
        self.users, self.videos = self.load_data(json_file)
        # Preprocess video titles
        self.videos['processed_title'] = self.videos['title'].apply(self.preprocess_text)
        # Calculate TF-IDF vectors
        self.tfidf_matrix = self.calculate_tfidf(self.videos['processed_title'])
        # Calculate cosine similarity matrix
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)


    def load_data(self, json_file):
        try:
            # Load JSON data
            with open(json_file, 'r') as f:
                data = json.load(f)
            # Convert to DataFrame
            users = pd.DataFrame(data['users'])
            videos = pd.DataFrame(data['videos'])
            return users, videos
        except Exception as e:
            raise CustomException(e, sys)


    def preprocess_text(self, text):
        try:
            # Remove non-alphanumeric characters, strip whitespace, and convert to lowercase
            text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
            text = text.lower()
            text = text.strip()
            text = contractions.fix(text)
            # tokenize document
            tokens = nltk.word_tokenize(text)
            # filter stopwords out of document
            filtered_tokens = [token for token in tokens if token not in stop_words]
            # re-create document from filtered tokens
            text = ' '.join(filtered_tokens)
            return text
        except Exception as e:
            raise CustomException(e, sys)


    def calculate_tfidf(self, documents):
        try:
            # Calculate TF-IDF vectors for the documents
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(documents)
            return tfidf_matrix
        except Exception as e:
            raise CustomException(e, sys)


    def get_user_watch_history(self, user_id):
        try:
            # Retrieve the watch history for a given user
            user_history = self.users[self.users['user_id'] == user_id]['watch_history'].values
            if user_history:
                return user_history[0]
            return []
        except Exception as e:
            raise CustomException(e, sys)


    def recommend_videos(self, user_id, top_n):
        try:
            # Generate video recommendations for a user
            user_name = self.users[self.users['user_id'] == user_id]['name'].values[0]
            watch_history = self.get_user_watch_history(user_id)
            if not watch_history:
                return []

            watched_indices = self.videos[self.videos['video_id'].isin(watch_history)].index.tolist()
            if not watched_indices:
                return []

            similarity_scores = self.similarity_matrix[watched_indices].mean(axis=0)
            self.videos['similarity_score'] = similarity_scores

            recommended_videos = self.videos[~self.videos['video_id'].isin(watch_history)].sort_values(by='similarity_score', ascending=False)
            recommend_video_indices = recommended_videos['video_id'].head(top_n).tolist()
            recommend_video_titles = recommended_videos['title'].head(top_n).tolist()

            return recommend_video_indices, recommend_video_titles, user_name
        except Exception as e:
            raise CustomException(e, sys)
