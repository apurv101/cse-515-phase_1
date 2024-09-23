import cv2
import numpy as np
from sklearn.cluster import KMeans
import math

def colorHistogram(video, r, size):
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

def main():
    video = ['Bodenturnen_2004_cartwheel_f_cm_np1_le_med_0.avi',
             'Blade_Of_Fury_-_Scene_1_sword_exercise_f_cm_np1_ri_med_3.avi',
             'AHF_longsword_against_Rapier_and_Dagger_Fight_sword_f_cm_np2_ri_bad_0.avi',
             'CastAway2_drink_u_cm_np1_le_goo_8.avi']
    colorHistogram(video[1], 4, 12)

if __name__ == '__main__':
    main()
