from scipy import misc, ndimage
from sklearn.cross_validation import KFold
import numpy as np
import os,re, math
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import urllib, cStringIO
from PIL import Image
import cv2, os, cPickle, random

scaler = StandardScaler()
basepath = '/home/kiritee/Downloads/Innovaccer Projects/Hackathon/Train files/'
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
recognizer = cv2.createLBPHFaceRecognizer()

def faceDetection(imagePath,cascPath,imageArray = []):
	if imagePath != '':
		cascPath = "haarcascade_frontalface_default.xml"
		faceCascade = cv2.CascadeClassifier(cascPath)
		image = cv2.imread(imagePath)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	else:
		image = imageArray
		cascPath = "haarcascade_frontalface_default.xml"
		faceCascade = cv2.CascadeClassifier(cascPath)
		if len(np.shape(imageArray))<3:
			gray = imageArray
		else:
			gray = cv2.cvtColor(imageArray, cv2.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(30, 30)
	)
	# print("Found {0} faces!".format(len(faces)))
	allCordinates = []
	for (x, y, w, h) in faces:
		allCordinates.append([x, y, w, h])
		cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
	# cv2.imshow("Faces found", image)
	# cv2.waitKey(0)
	if len(faces) > 0:
		return allCordinates
	return []

def blurringFaces(imagePath,imageArray = []):
	#(x, y, w, h)
	if imagePath != '': 
		coordinates = faceDetection(imagePath,cascadePath,[])
		image = ndimage.imread(imagePath, mode="RGB")
	else:
		coordinates = faceDetection('',cascadePath,imageArray)
		image = imageArray
	image.flags.writeable = True
	if coordinates != []:
		for k in range(len(coordinates)):	
			coord = coordinates[k]
			for i in range(coordinates[k][0],coordinates[k][0]+coordinates[k][3]):
				for j in range(coordinates[k][1],coordinates[k][1]+coordinates[k][2]):
					image[j,i] = 0
	return np.array(image)

def resizingImages(basepath,resizeShape):
	for foldername in ['nude men','nude women','dressed men','dressed women']:
		for root, dirnames, filenames in os.walk(basepath+foldername):
			for filename in filenames:
				if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
					filepath = os.path.join(root, filename)
					print filepath
					image = ndimage.imread(filepath, mode="RGB")
					image = blurringFaces(filepath,[])
					image_resized = misc.imresize(image, [resizeShape[0], resizeShape[1]])
					misc.imsave(basepath + 'resized '+ foldername + "/" + filename, image_resized)

def normalizeImages(inputImage):
	pixelValues = misc.imread(inputImage)
	normalizedList = []
	for row in pixelValues:
		normalizedList.append([int(math.ceil(np.mean(eachpixel)))/float(255) for eachpixel in row])
	normalizedList = np.array(normalizedList)
	return normalizedList

def allImagesHyperArrayConversion(folderNameList):
	training_set_input = []
	training_set_output = []
	test_set_input = []
	test_set_output = []
	for i,foldername in enumerate(folderNameList):
		allImagesArray = []
		for root, dirnames, filenames in os.walk(basepath+foldername):
			for filename in filenames:
				filepath = os.path.join(root, filename)
				allImagesArray.append(normalizeImages(filepath))
		npyFilename = foldername+".npy"
		np.save(npyFilename, allImagesArray)

def arrayFlattener(someHighDimArray):
	flattend_array = [] 
	for width in range(np.shape(someHighDimArray)[0]):
		flattend_array.append([x for list1 in someHighDimArray[1] for x in list1])
	return flattend_array

def singleMultiDimArrayFlattener(someHighDimArray):
	return np.array([x for list1 in someHighDimArray for x in list1])

def imageModificationAndResult(URL,modelPickleFile,localFile = ''):
	if URL != '':
		image = testAnImageOnline(URL)
	else:
		image = misc.imread(localFile)
	image = blurringFaces('',image)
	resized = misc.imresize(image, [64,64])
	normalizeAndFlattendImage = imageNormalizerAndScaler(resized)
	return resultFromStoredModel('dumps/model/NeuralNet_classifier.pkl',normalizeAndFlattendImage)

def testAnImageOnline(URL):
	file = cStringIO.StringIO(urllib.urlopen(URL).read())
	img = Image.open(file)
	return np.asarray(img)

def imageNormalizerAndScaler(imageArray):
	normalizedList = []
	for row in imageArray:
		normalizedList.append([int(math.ceil(np.mean(eachpixel)))/float(255) for eachpixel in row])
	normalizedList = np.array(normalizedList)
	flattenedArray = singleMultiDimArrayFlattener(normalizedList)
	return flattenedArray

def resultFromStoredModel(pickleFileOfTrainedModel,flattenedArray):
	with open(pickleFileOfTrainedModel, 'rb') as fid:
		clf_loaded = cPickle.load(fid)
	flattenedArray = flattenedArray.reshape(1,-1)
	p = clf_loaded.predict(flattenedArray)
	proba =  "%.3f" %(max(clf_loaded.predict_proba(flattenedArray)[0])*100)
	if p == 1:
		p = "Nude"
	else:
		p = "Not Nude"
	return p,proba

def trainingNeuralNetwork(tupleOfNeurons):
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
	Xtrain = np.array(list(flattened_array_Dressed[0:700])+list(flattened_array_Nude[0:1200]))
	Ytrain = np.array(list(predictor_dressed[0:700])+list(predictor_nude[0:1200]))
	# Xtrain = np.array(list(flattened_array_Dressed[:])+list(flattened_array_Nude[:]))
	# Ytrain = np.array(list(predictor_dressed[:])+list(predictor_nude[:]))
	# zipped = zip(list(Xtrain),list(Ytrain))
	# random.shuffle(zipped)
	# newXtrain = np.array([value[0] for value in zipped])
	# newYtrain = np.array([value[1] for value in zipped])
	# XValidate = np.array(list(flattened_array_Dressed[700:917])+list(flattened_array_Nude[1200:1570]))
	# YValidate = np.array(list(predictor_dressed[700:917])+list(predictor_nude[1200:1570]))
	# Xtest = np.array(list(flattened_array_Dressed[917:])+list(flattened_array_Nude[1570:]))
	# Ytest= np.array(list(predictor_dressed[917:])+list(predictor_nude[1570:]))
	totalXs = flattened_array_Nude+flattened_array_Dressed
	totalYs = predictor_nude+predictor_dressed
	kf = KFold(len(totalYs), n_folds=4, shuffle = True)
	for train_index, test_index in kf:
		trainsetX = [totalXs[index1] for index1 in train_index]
		trainsetY = [totalYs[index1] for index1 in train_index]
		testsetX = [totalXs[index2] for index2 in test_index]
		testsetY = [totalYs[index2] for index2 in test_index]
		# Neural Network
		print "Training started..."
		clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=tupleOfNeurons ,random_state=1)
		clf.fit(trainsetX, trainsetY)
		with open('NeuralNet_classifier.pkl', 'wb') as fid:
			cPickle.dump(clf, fid)    
		with open('NeuralNet_classifier.pkl', 'rb') as fid:
			clf_loaded = cPickle.load(fid)
		predicted = clf_loaded.predict(testsetX)
		accuracy_nude = 0
		accuracy_dressed = 0
		wrongOnes = 0
		for x in range(len(predicted)):
			if predicted[x] == testsetY[x]:
				if predicted[x] == 1:
					accuracy_nude += 1
				else:
					accuracy_dressed += 1
			else:
				wrongOnes += 1
		print "Accuracy: ",(accuracy_nude+accuracy_dressed)*100/float(accuracy_nude+accuracy_dressed+wrongOnes)

def videocaptureAndOutputImages(filepath):
	vidcap = cv2.VideoCapture(filepath)
	(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
	if int(major_ver)  < 3 :
		fps = vidcap.get(cv2.cv.CV_CAP_PROP_FPS)
		print "Frames per second: {0}".format(fps)
	else :
		fps = vidcap.get(cv2.CAP_PROP_FPS)
		print "Frames per second : {0}".format(fps)
	success,image = vidcap.read()
	count = 0
	success = True
	listOfFrames = []
	while success:
		success,image = vidcap.read()
		count += 1
		if count%25 == 0.0:
			listOfFrames.append(image)
			# cv2.imwrite("Resources/v1/frame%d.jpg" % (count//25), image)
	return listOfFrames

def maskingBlackRegions(imageArray):
	mask = imageArray > 0
	coords = np.argwhere(mask)
	x0, y0, z0 = coords.min(axis=0)
	x1, y1, z1 = coords.max(axis=0) + 1
	cropped = imageArray[x0:x1, y0:y1]
	return cropped


def clusteringAndRemovalOfFrames(filepath):
	arrayOfFrames = videocaptureAndOutputImages(filepath)
	resizedArrayOfFrames = [misc.imresize(image, [4,4]) for image in arrayOfFrames]
	reArr = resizedArrayOfFrames
	[cv2.imwrite("Resources/v1/frame%d.jpg" %(i+1),maskingBlackRegions(arrayOfFrames[i+1])) for i in range(len(reArr)-1) if abs(np.sum(reArr[i] - reArr[i+1])) > 256]
	return [imageModificationAndResult("",'NeuralNet_classifier.pkl','Resources/v1/frame%d.jpg' %(i+1)) for i in range(len(reArr)-1) if abs(np.sum(reArr[i]-reArr[i+1])) > 256]


if __name__ == '__main__':
	# resizingImages(basepath,[64,64])
	# allImagesHyperArrayConversion(['resized nude men','resized nude women','resized dressed men','resized dressed women'])
	# trainingNeuralNetwork((20))
	out =  imageModificationAndResult("",'dumps/model/NeuralNet_classifier.pkl','Resources/20.jpg')
	print "Sureness of prediction: ", out[1],"\nResult: ",out[0]
	# print clusteringAndRemovalOfFrames("Resources/v1.mp4")