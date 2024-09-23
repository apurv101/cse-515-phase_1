import cv2
import numpy as np
from sklearn.cluster import KMeans
import math

def colorHistogram(video, r, size, outputFileName):
    videoFrames = cv2.VideoCapture(video)
    numFrames = int(videoFrames.get(cv2.CAP_PROP_FRAME_COUNT))
    framesList = []

    #Get the first middle and last frame of the video
    videoFrames.set(cv2.CAP_PROP_POS_FRAMES, 0)
    check,saveFrame = videoFrames.read()
    if check:
        framesList.append(saveFrame)

    videoFrames.set(cv2.CAP_PROP_POS_FRAMES, numFrames // 2)
    check,saveFrame = videoFrames.read()
    if check:
        framesList.append(saveFrame)

    videoFrames.set(cv2.CAP_PROP_POS_FRAMES, numFrames - 2)
    check,saveFrame = videoFrames.read()
    if check:
        framesList.append(saveFrame)
    videoFrames.release()

    print (f"<Frame # (1-First, 2-Middle, 3-Last); Cell #; [Histogram]")

    #Divide the frames into r number of cells
    for data in range(len(framesList)):
        frame = framesList[data]
        colorFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, gar = colorFrame.shape
        cellHeight = int(height / r)
        cellWidth = int(width / r)
        rgbList = []
        histogramList = []
        for i in range(r):
            for k in range(r):
                curr = colorFrame[i * cellHeight:(i + 1) * cellHeight, k * cellWidth:(k + 1) * cellWidth]
                pixel = curr.reshape(-1, 3)
                km = KMeans(n_clusters=size)
                km.fit(pixel)

                rgb = km.cluster_centers_.astype('int')

                histogram, gar = np.histogram(km.labels_, bins=np.arange(0, size + 1), range=(0,255))

                histogram = histogram.astype('float32')
                histogram /= histogram.sum()
                histogramList.append(histogram)
                rgbList.append(rgb)

    for i in range(len(framesList)):
        for j in range(len(rgbList)):
            list1 = rgbList[j]
            list2 = histogramList[j]
            print (f"<{i + 1}; {j + 1};", end=' ') #[{rgbList[j]}, {histogramList[j]}]>")
            for k in range(size):
                print(f"{list1[k]}, {list2[k]}", end=' ')
            print(">\n")
        cv2.imshow("Frame", framesList[i])
        cv2.waitKey(0)
        
    # Write to output file
    with open(outputFileName, "w") as myfile:
        for i in range(len(framesList)):
            for j in range(len(rgbList)):
                list1 = rgbList[j]
                list2 = histogramList[j]
                myfile.write(f"<{i + 1}; {j + 1}; ")
                for k in range(size):
                    myfile.write(f"{list1[k]}, {list2[k]} ")
                myfile.write(">\n")
    

if __name__ == '__main__':
    import argparse

    # Argument parser to handle input arguments
    parser = argparse.ArgumentParser(description="Task 3: Creates color histograms for a video.")
    parser.add_argument('--video', type=str, help='Path to the target video.') 
    parser.add_argument('--output_file', type=str, default='histogram_features.txt', help='Path to save or load the video features (default: histogram_features.txt).')
    parser.add_argument('--resolution', type=int, default=4, help='Resolution of histogram.')
    parser.add_argument('--size', type=int, default=12, help='Color histogram size.')
    
    args = parser.parse_args()
    colorHistogram(args.video, args.size, args.resolution, args.output_file)
