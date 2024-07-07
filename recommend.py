import sys
from src.video_recommender import VideoRecommender
from src.exception import CustomException


def get_video_recommendations(user_id, n):
    try:
        recommender = VideoRecommender('./artifacts/data.json')
        recommendations = recommender.recommend_videos(user_id, n)
        return recommendations
    except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    while True:
        try:
            user_id = int(input("Enter user ID between 1 and 20: "))
            if 1 <= user_id <= 20:
                print("\nValid user ID")
                n = int(input("Enter number of recommendations: "))
                recommendations = get_video_recommendations(user_id, n)

                print(f'\n\nTop {n} recommended videos for {recommendations[2]} with ID {user_id}:\
                    \nVideo Titles: {recommendations[1]}\nVideo IDs: {recommendations[0]}')
                break
            else:
                print("\nInvalid user ID. Please enter a value between 1 and 20") 
        except ValueError:
            print("\nInvalid input. Please enter an integer value")
