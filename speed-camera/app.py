import cv2
import numpy as np

MAX_NUM_CORNERS = 100
PX_PER_CM = 370
FPS = 30
DISTANCE_THRESH = 10
REFRESH_RATE = 20

def d2(p, q):
	p = np.array(p)
	q = np.array(q)
	return np.linalg.norm(p-q)

colors = np.random.randint(0, 255, (MAX_NUM_CORNERS, 3))
frame_counter = 0

video = cv2.VideoCapture('test.mov')

_, old_frame = video.read()
frame_counter = frame_counter + 1
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
prev_pts = cv2.goodFeaturesToTrack(image=old_gray, maxCorners=MAX_NUM_CORNERS,
	qualityLevel=0.3, minDistance=7, blockSize=7)

"""
for pt in prev_pts:
	x, y = pt.ravel()
	cv2.circle(old_frame, (x,y), 5, (0,255,0), -1)
	
cv2.imshow('features', old_frame)
cv2.waitKey(0)
"""

mask = np.zeros_like(old_frame)
mask_text = np.zeros_like(old_frame)
while True:
	if frame_counter % REFRESH_RATE == 0:
		mask_text.fill(0)
	
	_, frame = video.read()
	frame_counter = frame_counter + 1
	frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	next_pts, statuses, _ = cv2.calcOpticalFlowPyrLK(prevImg=old_gray, nextImg=frame_gray, prevPts=prev_pts,
		winSize=(15,15),
		maxLevel=2,
		criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 10, 0.03))
	
	good_next_pts = next_pts[statuses==1]
	good_old_pts = prev_pts[statuses==1]
	
	old_gray = frame_gray.copy()
	prev_pts = good_next_pts.reshape(-1, 1, 2)
	
	for i, (next, old) in enumerate(zip(good_next_pts, good_old_pts)):
		x, y = next.ravel()
		r, s = old.ravel()
		
		# Drawing
		cv2.line(mask, (x,y), (r,s), list(colors[i]), 2)
		cv2.circle(frame, (x,y), 5, list(colors[i]), -1)
		
		# Speed
		distance = d2((x,y), (r,s))
		if distance > DISTANCE_THRESH:
			speed_str = str(distance / PX_PER_CM * FPS) + ' cm/s'
			print speed_str
			cv2.putText(mask_text, speed_str, (x,y), cv2.FONT_HERSHEY_TRIPLEX, 0.5, list(colors[i]))
			
		
	frame_final = cv2.add(frame, mask)
	frame_final = cv2.add(frame_final, mask_text)
	cv2.imshow('frame', frame_final)
	if cv2.waitKey(1) & 0xFF == ord('q'): break

cv2.destroyAllWindows()
video.release()

