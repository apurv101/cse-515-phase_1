import os
import torch
import numpy as np
from task_1 import extract_features  # Importing the extract_features function from task_1.py
from numpy import dot
from numpy.linalg import norm
import pickle


# Function to recursively extract and store feature vectors for all videos in the directory and subdirectories
def extract_and_store_features(directory, output_file="video_features.pkl"):
    """
    Extract features for all videos in the given directory (recursively) and store them in a pickle file.

    Args:
        directory (str): The directory containing the video files.
        output_file (str): The file to store the extracted feature vectors (default: video_features.pkl).
    """
    video_features = {}

    # Walk through the directory and subdirectories to find video files
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".avi") or file.endswith(".mp4"):  # Adjust file extensions as necessary
                video_path = os.path.join(root, file)
                relative_path = os.path.relpath(video_path, directory)  # Get the relative path
                print(f"Extracting features for {relative_path}...")

                # Extract feature vectors using the function from task_1.py
                fv_layer3, fv_layer4, fv_avgpool = extract_features(video_path)

                # Store the features in the dictionary using the relative path as the key
                video_features[relative_path] = {
                    'layer3': fv_layer3,
                    'layer4': fv_layer4,
                    'avgpool': fv_avgpool
                }

    # Save the features to a file
    with open(output_file, 'wb') as f:
        pickle.dump(video_features, f)

    print(f"Feature vectors saved to {output_file}")



# Function to load feature vectors from a pickle file
def load_features(feature_file="video_features.pkl"):
    """
    Load stored feature vectors from a pickle file.

    Args:
        feature_file (str): The path to the pickle file containing feature vectors (default: video_features.pkl).
    
    Returns:
        dict: A dictionary of video feature vectors.
    """
    with open(feature_file, 'rb') as f:
        return pickle.load(f)


# Cosine similarity function
def cosine_similarity(v1, v2):
    """
    Compute the cosine similarity between two vectors.
    """
    return dot(v1, v2) / (norm(v1) * norm(v2))


# Function to find the most similar videos to a target video
def find_most_similar(video_features, key_name, m=5):
    """
    Find the 'm' most similar videos to a given target video based on their feature vectors.

    Args:
        video_features (dict): A dictionary containing feature vectors for all videos.
        target_video (str): The filename of the target video.
        m (int): The number of most similar videos to return (default: 5).
    
    Returns:
        list: A list of tuples containing the most similar videos and their similarity scores.
    """
   # Check if the key_name exists in the video_features
    if key_name not in video_features:
        raise KeyError(f"Target video '{key_name}' not found in the stored video features.")
    
    target_features = video_features[key_name]
    similarities = {}

    # Compute similarity between the target video and all other videos
    for video, features in video_features.items():
        if video == key_name:
            continue  # Skip the target video itself

        # Calculate cosine similarity for each feature vector
        similarity_layer3 = cosine_similarity(target_features['layer3'], features['layer3'])
        similarity_layer4 = cosine_similarity(target_features['layer4'], features['layer4'])
        similarity_avgpool = cosine_similarity(target_features['avgpool'], features['avgpool'])

        # Aggregate similarity scores (you can change weighting if necessary)
        total_similarity = (similarity_layer3 + similarity_layer4 + similarity_avgpool) / 3
        similarities[video] = total_similarity

    # Sort by similarity (higher is better for cosine similarity)
    sorted_videos = sorted(similarities.items(), key=lambda item: item[1], reverse=True)

    # Return the top 'm' most similar videos
    return sorted_videos[:m]


if __name__ == "__main__":
    import argparse

    # Argument parser to handle input arguments
    parser = argparse.ArgumentParser(description="Task 5: Extract features for all videos and find most similar ones.")
    parser.add_argument('--directory', type=str, required=True, help='Path to the directory containing the video files.')
    parser.add_argument('--target_video', type=str, help='Filename of the target video.')  # Optional for similarity
    parser.add_argument('--key_name', type=str, help='Key to use for looking up the target video in the feature file.')  # New argument for specifying the key
    parser.add_argument('--m', type=int, default=5, help='Number of most similar videos to return (default: 5).')
    parser.add_argument('--output_file', type=str, default='video_features.pkl', help='Path to save or load the video features (default: video_features.pkl).')
    parser.add_argument('--extract', action='store_true', help='Flag to extract and store features from the video directory.')

    args = parser.parse_args()

    # Step 1: Extract and store features if the --extract flag is provided
    if args.extract:
        extract_and_store_features(args.directory, output_file=args.output_file)

    # Step 2: If --target_video is provided, find most similar videos
    if args.key_name:
        # Load the features from the file
        video_features = load_features(feature_file=args.output_file)

        # Find the most similar videos using the specified key_name
        most_similar_videos = find_most_similar(video_features, args.key_name, m=args.m)

        # Print the results
        print(f"\nTop {args.m} most similar videos to '{args.key_name}':")
        for video, similarity in most_similar_videos:
            print(f"{video}: similarity score = {similarity:.4f}")