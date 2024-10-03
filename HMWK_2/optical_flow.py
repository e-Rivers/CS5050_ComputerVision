import cv2 
import numpy as np

def debug(thing, title):
    print('----------------------------------------------------')
    print(title)
    print(thing )
    print('----------------------------------------------------')

def lucas_kanade_method(video_path):
    # Read the video 
    cap = cv2.VideoCapture(video_path)
 
    # Parameters for ShiTomasi corner detection
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
 
    # Parameters for Lucas Kanade optical flow
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )
 
    # Create random colors
    color = np.random.randint(0, 255, (100, 3))
 
    # Reached first frame.
    reached_ff = False
    while not reached_ff:
        # Take first frame and find corners in it
        ret, old_frame = cap.read()
        if np.sum(old_frame) != 0:
            reached_ff = True    


    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    #debug(old_gray_int, 'actual first frame')

    #old_gray = old_gray_int / 255
    #old_gray = old_gray.astype('float32')
    #debug(old_gray.dtype, 'old gray')

    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    #debug(p0, 'p0')

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    #mask = mask.astype('float32')
    debug(mask.dtype, 'mask')

    while True:
        # Read new frame
        ret, frame = cap.read()
        #frame = frame_int / 255
        #frame = frame.astype('float32')

        if not ret:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #debug(frame_gray.dtype, 'frame gray')


        # Calculate Optical Flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            old_gray, frame_gray, p0, None, **lk_params
        )
        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        #debug(p1, 'p1')
    
        # Draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            debug((int(a), int(b)), 'a b')
            debug((int(c), int(d)), 'c d')
            # debug(color[i].tolist(), 'color')
            cv2.imshow('before line', mask)
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            debug(mask.shape, 'mask shape')
            debug(mask[int(a), int(b)], 'line in mask')
            debug(mask[int(c), int(d)], 'line in mask')
            cv2.imshow('after line', mask)
            #debug(mask, 'mask with line I guess')
            frame = cv2.circle(frame, (int(a), int(b)), 5, tuple(color[i].tolist()), 2)
    
        # Display the demo
        img = cv2.add(frame, mask)
        cv2.imshow("frame", img)
        k = cv2.waitKey(25) & 0xFF
        if k == 27:
            break
    
        # Update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

lucas_kanade_method('testOpticalFlow.mp4')
