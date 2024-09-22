
# Task 5: Extracting and Finding the Most Similar Videos Using Feature Vectors

This project implements **Task 5**, which involves:
1. Extracting feature vectors for all videos in a specified directory (including subdirectories) using a pre-trained **R3D-18** model.
2. Finding the most similar videos to a target video based on the cosine similarity of the extracted feature vectors.

The feature extraction uses functions from **task_1.py**, so it’s necessary to have that file available.

## 1. Overview

### Step 1: Extracting and Storing Features
- The script will recursively extract **feature vectors** for all videos in a given directory (and subdirectories) using the **R3D-18** model.
- It extracts feature vectors from three layers:
  - **Layer 3**: Reduced to a 512-dimensional vector.
  - **Layer 4**: Reduced to a 512-dimensional vector.
  - **AvgPool**: A 512-dimensional vector.

The feature vectors are stored in a pickle file (`video_features.pkl` by default) for later use.

### Step 2: Finding the Most Similar Videos
- The script calculates the **cosine similarity** between the target video and all other videos based on their feature vectors.
- You can specify the key used to look up the target video using the `--key_name` argument. This key will be used to match the relative path or filename stored in the `video_features.pkl` file.
- The top `m` most similar videos are returned based on the similarity scores.

## 2. Requirements

Ensure you have the following dependencies installed:
- **Python 3.7+**
- **PyTorch** (for model loading and tensor operations)
- **Torchvision** (for pre-trained video models)
- **OpenCV** (for video visualization)
- **NumPy** (for numerical operations)
- **pickle** (standard Python library for saving and loading objects)

You can install the required packages using the following command:
```bash
pip install torch torchvision opencv-python numpy
```

## 3. How to Run the Script

### 3.1. Command Line Arguments

The script accepts the following arguments:

- `--directory`: The path to the directory containing the video files (and subdirectories).
- `--target_video`: The full path to the target video for feature extraction (optional for similarity).
- `--key_name`: The key to use when looking up the target video in the `video_features.pkl` file (use the relative path or a key format consistent with the extraction process).
- `--m`: The number of most similar videos to return (default: 5).
- `--output_file`: The path to the file where the extracted feature vectors will be saved or loaded from (default: `video_features.pkl`).
- `--extract`: If this flag is provided, the script will extract features for all videos in the specified directory and store them.

### 3.2. Running the Script

#### Step 1: Extract Feature Vectors for All Videos

To extract feature vectors from all videos in a specified directory and store them in a pickle file, run the following command:

```bash
python task_5.py --directory ./target_videos --extract
```

This will extract feature vectors from the videos in the `target_videos` directory and its subdirectories and save them in `video_features.pkl`.

#### Step 2: Find the Most Similar Videos Using `--key_name`

Once the features are extracted and stored, you can run the following command to find the most similar videos by specifying the key used in `video_features.pkl` (e.g., the relative path of the video):

```bash
python task_5.py --directory ./target_videos --key_name cartwheel/Acrobacias_de_un_fenomeno_cartwheel_f_cm_np1_fr_bad_3.avi --m 5
```

This will return the top 5 most similar videos to `cartwheel/Acrobacias_de_un_fenomeno_cartwheel_f_cm_np1_fr_bad_3.avi` and print them along with their similarity scores.

### Example Workflow

1. **Extracting Features**:
   ```bash
   python task_5.py --directory ./target_videos --extract
   ```

2. **Finding the Most Similar Videos**:
   ```bash
   python task_5.py --directory ./target_videos --key_name cartwheel/Acrobacias_de_un_fenomeno_cartwheel_f_cm_np1_fr_bad_3.avi --m 5
   ```

## 4. File Structure

Your project directory should be structured as follows:

```
.
├── task_1.py          # Script containing functions to extract features
├── task_5.py          # This script for extracting features and finding similar videos
└── target_videos/     # Directory containing the video files and subdirectories
```

### task_1.py

The **task_1.py** script is required because this project relies on its `extract_features` function to extract feature vectors from videos using the R3D-18 model.

## 5. How It Works

### 5.1. Feature Extraction (`extract_and_store_features`)

The `extract_and_store_features` function uses the `extract_features` function from **task_1.py** to extract features from three layers of the pre-trained R3D-18 model:
- **Layer 3**: The output tensor is reduced to a 512-dimensional vector.
- **Layer 4**: The output tensor is reduced to a 512-dimensional vector.
- **AvgPool**: Directly provides a 512-dimensional vector.

These vectors are stored in a dictionary and then saved to a pickle file for later use. The key used for each video is its **relative path** from the `target_videos` directory.

### 5.2. Cosine Similarity

Once feature vectors are extracted, the `find_most_similar` function calculates the **cosine similarity** between the target video and all other videos. Cosine similarity is used because it measures the similarity of two vectors regardless of their magnitude.

The similarity score is computed for the feature vectors from all three layers (`layer3`, `layer4`, `avgpool`) and then averaged. The top `m` most similar videos are returned based on the overall similarity score.

### 5.3. Output

The output is a list of the most similar videos to the target video, along with their similarity scores. The higher the score, the more similar the video is to the target video.

## 6. Example Output

Here is an example of the output when you find the top 5 most similar videos to a target video:

```
Top 5 most similar videos to cartwheel/Acrobacias_de_un_fenomeno_cartwheel_f_cm_np1_fr_bad_3.avi:
cartwheel/another_cartwheel_video.avi: similarity score = 0.9753
sword/sword_fight_1.avi: similarity score = 0.9658
ride_bike/bike_trick_2.avi: similarity score = 0.9584
wave/wave_greeting_3.avi: similarity score = 0.9456
drink/drinking_water_1.avi: similarity score = 0.9321
```

## 7. Troubleshooting

- **Feature Extraction Takes Too Long**: If you have many videos, extracting features for all of them can take some time. You may want to split the workload or reduce the number of frames sampled from each video.
- **Video Not Found**: Ensure that the `--key_name` you provide matches the key stored in the `video_features.pkl` file (e.g., use the relative path to the video file).
- **Pickle File Not Found**: If the `video_features.pkl` file is missing, make sure you have run the feature extraction step using the `--extract` flag before attempting to find similar videos.

## 8. Conclusion

This script automates the process of extracting feature vectors from videos and finding the most similar videos based on those features. It uses pre-trained deep learning models to extract features and computes cosine similarity to identify similar videos.

