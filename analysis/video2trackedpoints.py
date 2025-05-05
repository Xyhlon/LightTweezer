#!/bin/python3

import argparse
import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

def display_frame(frame):
    cv2.imshow("Frame", frame)
    cv2.waitKey(1)

def display_hist(frame):
    plt.hist(frame.ravel(),256)
    plt.show()

if __name__ == "__main__":
    max_points = 100
    temporal_depth = 3
    causal_length = 200
    limit_frame = None

    args = argparse.ArgumentParser()
    args.add_argument("video", help="The video file from which the object are extraced")
    args.add_argument("--output", help="numpy npy output filename for the tracked object")
    args.add_argument("--temporal_depth", help="object permenance should be satisfied for at least the temporal_depth on frames, otherwise the object is discarded")
    args.add_argument("--causal_length", help="A hard cut off filter no object can jump within a frame that distance")
    args.add_argument('--visualize', help="Visualize the object tracking", action=argparse.BooleanOptionalAction)
    args.add_argument('--filter-output', help="Removes mostly zero columns of not premanent object", action=argparse.BooleanOptionalAction)
    args.add_argument('--limit-frame', help="limit after which frame the tracking is stopped")
    args = args.parse_args()
    filename = args.video
    output = filename.split(".")[0]+"_tracked_obj.npy"
    if args.output:
        output = args.output
    if args.temporal_depth:
        temporal_depth = int(args.temporal_depth)
    if args.causal_length:
        causal_length = int(args.causal_length)
    if args.limit_frame:
        limit_frame = int(args.limit_frame)
    visualize = args.visualize
    filter_output = args.filter_output

    video = cv2.VideoCapture(filename)

    fps = video.get(cv2.CAP_PROP_FPS)
    fcount = video.get(cv2.CAP_PROP_FRAME_COUNT)
    delta = 1/fps
    cols = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    rows = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"{cols}x{rows}")

    lim = fcount - 1
    if limit_frame:
        lim = limit_frame 

    tracked_circles = np.zeros((max_points,2,int(lim+1)), dtype=np.float32)

    frame_count = 0
    with tqdm(total=lim) as pbar:
        while True:
            ret, frame = video.read()
            if not ret or (limit_frame and frame_count>=limit_frame):
                break
            alpha = 1.3
            beta = 0
            higher_contrast = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
            new_image =  np.clip(higher_contrast, 0, 255)
            ret,thresh1 = cv2.threshold(new_image,245,255,cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
            th_opened = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel,iterations=1)

            gray = th_opened[:, :, 1]
            gray = cv2.GaussianBlur(gray, (25, 25), 0)
            # display_frame(gray)
            # edges = cv2.Canny(gray,90,10)
            # display_frame(edges)
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 35, param1=90, param2=10, minRadius=0, maxRadius=12)

            # select the x,y coordinates of the cirlces
            circles = circles[0, :]
            circles = circles[:,:2]

            # temporal coherence matching
            x = circles.shape[0]
            if frame_count>temporal_depth:
                for i,col in enumerate(circles):
                    for past in range(1,temporal_depth+1):
                        prev = tracked_circles[:x, :, frame_count-past]
                        prev_ = prev-col
                        # prev_[prev_<0] = -prev_[prev_<0]
                        prev_ = prev_**2
                        dist = np.sum(prev_,axis=1)
                        nearest_loc = np.argmin(dist)
                        if dist[nearest_loc] < causal_length:
                            tracked_circles[nearest_loc, :, frame_count] = col
                            circles[i] = col
                            break
                        else:
                            circles[i] = 0
            else:
                tracked_circles[:x, :, frame_count] = circles[circles[:, 1].argsort()]

            if visualize:
                # visual detected circles 
                if circles is not None:
                    circlesi = np.around(circles).astype(np.uint16)
                    for i in circlesi:
                        center = (i[0], i[1])
                        # circle center
                        cv2.circle(frame, center, 1, (0, 100, 100), 3)
                        # circle outline
                        radius = 10 
                        cv2.circle(frame, center, radius, (255, 0, 255), 3)

                display_frame(frame)


            frame_count+=1


            # allow breaking with q
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            pbar.update(1)


    video.release()
    if filter_output:
        frames = tracked_circles.shape[2]
        # mask = tracked_circles[:,:,frames//2]==0
        # # mask = tracked_circles[:,:,-1]==0
        # mask = np.all(mask, axis=1)
        # tracked_circles = tracked_circles[~mask]
        p_non_zero = np.count_nonzero(tracked_circles, axis=2)/frames
        mask = np.all(p_non_zero > 0.5, axis=1)
        tracked_circles = tracked_circles[mask]
        tracked_circles = tracked_circles[:,:,temporal_depth:]




    with open(output, 'wb') as f:
        np.save(f, tracked_circles)
