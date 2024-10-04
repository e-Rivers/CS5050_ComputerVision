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
    debug(p0, 'p0')

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
        debug(p1, 'p1')
    
        # Draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            debug(new, 'new shape')
            debug(old, 'old shape')
            a, b = new.ravel()
            c, d = old.ravel()
            debug((int(a), int(b)), 'a b')
            debug((int(c), int(d)), 'c d')
            # debug(color[i].tolist(), 'color')
            cv2.imshow('before line', mask)
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 10)
            debug(mask.shape, 'mask shape')
            cv2.imshow('after line', mask)
            #debug(mask, 'mask with line I guess')
            frame = cv2.circle(frame, (int(a), int(b)), 5, tuple(color[i].tolist()), 10)
            frame = cv2.circle(frame, (int(c), int(d)), 5, tuple(color[i].tolist()), 10)
    
        # Display the demo
        img = cv2.add(frame, mask)
        cv2.imshow("frame", img)
        k = cv2.waitKey(25) & 0xFF
        if k == 27:
            break
    
        # Update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)


def dense_optical_flow(method, video_path, params=[], to_gray=False):
    # Read the video and first frame
    cap = cv2.VideoCapture(video_path)
    
    # Reached first frame.
    reached_ff = False
    while not reached_ff:
        # Take first frame and find corners in it
        ret, old_frame = cap.read()
        if np.sum(old_frame) != 0:
            reached_ff = True
 
    # crate HSV & make Value a constant
    hsv = np.zeros_like(old_frame)
    hsv[..., 1] = 255
 
    
    # Preprocessing for exact method
    if to_gray:
        old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    while True:
        # Read the next frame
        ret, new_frame = cap.read()
        frame_copy = new_frame
        if not ret:
            break
    
        # Preprocessing for exact method
        if to_gray:
            new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
    
        # Calculate Optical Flow
        flow = method(old_frame, new_frame, None, *params)
    
        # Encoding: convert the algorithm's output into Polar coordinates
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # Use Hue and Value to encode the Optical Flow
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    
        # Convert HSV image into BGR for demo
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imshow("frame", frame_copy)
        cv2.imshow("optical flow", bgr)
        k = cv2.waitKey(25) & 0xFF
        if k == 27:
            break
    
        # Update the previous frame
        old_frame = new_frame


xd = 'lucaskanade_dense'
video_path = 'testOpticalFlow6.mp4'
# if xd == 'lucaskanade_dense':
#     method = cv2.optflow.calcOpticalFlowSparseToDense
#     frames = dense_optical_flow(method, video_path, to_gray=True)
# # if to_gray:
# #     new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
# elif xd == 'farneback':
#     method = cv2.calcOpticalFlowFarneback
#     params = [0.5, 3, 15, 3, 5, 1.2, 0]  # default Farneback's algorithm parameters
#     frames = dense_optical_flow(method, video_path, params, to_gray=True)
# elif xd == "rlof":
#     method = cv2.optflow.calcOpticalFlowDenseRLOF
#     frames = dense_optical_flow(method, video_path)

lucas_kanade_method(video_path)