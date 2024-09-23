from task_1 import extract_features
from task_5 import extract_and_store_features
import pickle

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Task 4: Extract features and stores features.")
    parser.add_argument('--directory', type=str, required=True, help='Path to the directory containing the video files.')
    parser.add_argument('--output_file', type=str, default='video_features.pkl', help='Path to save or load the video features (default: video_features.pkl).')

    args = parser.parse_args()
    extract_and_store_features(args.directory, output_file=args.output_file)