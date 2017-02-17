# import cv2
# vidcap = cv2.VideoCapture('Resources/v1.mp4')
# for timeframe in range(40000,50000,500):
# 	vidcap.set(cv2.cv.CV_CAP_PROP_FRAME_COUNT,7000)
# 	timeframe = 4000
# 	success,image = vidcap.read()
# 	if success:
# 		cv2.imwrite("Resources/v1/frame%dsec.jpg" %(int(timeframe/1000)) , image)
import random
import numpy as np
from sklearn.cross_validation import KFold
def arrayFlattener(someHighDimArray):
	flattend_array = [] 
	for width in range(np.shape(someHighDimArray)[0]):
		flattend_array.append([x for list1 in someHighDimArray[1] for x in list1])
	return flattend_array

womenNude = np.load('resized nude women.npy')
menNude = np.load('resized nude men.npy')
menDressed = np.load('resized dressed men.npy')
womenDressed = np.load('resized dressed women.npy')
set_Nude = np.array(list(womenNude)+list(menNude))
set_Dressed = np.array(list(womenDressed)+list(menDressed))
flattened_array_Nude = arrayFlattener(set_Nude)
flattened_array_Dressed = arrayFlattener(set_Dressed)
predictor_nude = [1 for x in range(len(flattened_array_Nude))]
predictor_dressed = [0 for x in range(len(flattened_array_Dressed))]
totalarray = zip(flattened_array_Nude,predictor_nude)+zip(flattened_array_Dressed,predictor_dressed)
random.shuffle(totalarray)
totalXs = flattened_array_Nude+flattened_array_Dressed
totalYs = predictor_nude+predictor_dressed
kf = KFold(len(totalYs), n_folds=4, shuffle = True)
for train_index, test_index in kf:
	trainsetX = [totalXs[index1] for index1 in train_index]
	trainsetY = [totalYs[index1] for index1 in train_index]
	testsetX = [totalXs[index2] for index2 in test_index]
	testsetY = [totalYs[index2] for index2 in test_index]
	