# Core libraries
import os
import glob
import shutil
import io
import sys
import torch
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image
import time
# np.set_printoptions(threshold=np.inf, linewidth=np.inf) #inf = infinity 
from sklearn import metrics

# PyTorch
import torch
from torch.utils import data
from torch.autograd import Variable
import torchvision
from torchvision import transforms

# Local libraries
from bum.MetricLearningIdentification.utilities.utils import Utilities
from bum.MetricLearningIdentification.models.embeddings import resnet50

# Import our dataset class
from bum.MetricLearningIdentification.datasets_copy.OpenSetCows2020.OpenSetCows2020_only_test import OpenSetCows2020

import cv2

"""
File for inferring the embeddings of the test portion of a selected database and
evaluating its classification performance using KNN
"""

# python only_test.py --model_path=output/fold_0/best_model_state.pkl --folds_file=datasets_copy/OpenSetCows2020/splits/10-90-custom.json --save_path=output/fold_0/

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "2,3"  # Set the GPU 2 to use

# torch.manual_seed(40)
# torch.cuda.manual_seed(40)
# torch.cuda.manual_seed_all(40)

# For a trained model, let's evaluate it
def evaluateModel(args, recog, regist, flag):
	start_time = time.time() #러닝타임 측정
	recog_cow = recog
	regist_cow = regist
	# recog_cow = False
	# regist_cow = True
	if recog_cow == True and flag == 0:
		src_path = '/workspace/testFastAPI/test1'
	elif recog_cow == True and flag == 1:
		src_path = '/workspace/testFastAPI/test2'
	elif regist_cow == True:
		# print('====================================11111111111111')
		src_path = "/workspace/testFastAPI/test2"
	new_path = "/workspace/bum/MetricLearningIdentification/datasets_copy/OpenSetCows2020/images/test/"
	file_count = len(os.listdir(src_path))
    # print(len(file_count))
	file_list = read_all_file(src_path)
	copy_all_file(file_list, new_path)
 
	
 	# Load the relevant datasets
	# train_dataset = Utilities.selectDataset(args, True)
	test_dataset = Utilities.selectDataset(args, False)
	print(len(test_dataset))

	# embedding npz file load
	dt = np.load('/workspace/bum/MetricLearningIdentification/output/fold_0/train_embeddings.npz')
	train_embeddings = dt['embeddings']
	train_labels = dt['labels']

	# Get the embeddings and labels of the training set and testing set
	# train_embeddings, train_labels= inferEmbeddings(args, train_dataset, "train")
	test_embeddings, test_labels = inferEmbeddings(args, test_dataset, "test")
	# print(f'=========================test_embeddings shape{test_embeddings.shape}')
	# print(f'=========================test_labels shape{test_labels.shape}')
 
	# tt = np.load('/workspace/bum/MetricLearningIdentification/output/fold_0/test_embeddings.npz')
	# tt1 = tt['embeddings']
	# tt2 = tt['labels']
 
	# print(train_embeddings.shape)
	# print(train_labels.shape)
	# print(tt1.shape)
	# print(tt2.shape)
 
 
	if regist_cow == True:
		accuracy, pred, result_label= KNNAccuracy(train_embeddings, train_labels, test_embeddings, test_labels)
		register_cow(train_embeddings, train_labels, test_embeddings, file_count)

	# Classify them
	# accuracy = KNNAccuracy(train_embeddings, train_labels, test_embeddings, test_labels)
	if recog_cow == True:
		accuracy, pred, result_label= KNNAccuracy(train_embeddings, train_labels, test_embeddings, test_labels)
	# print(pred)
	# print(result_label)
	# print(test_labels)

	#remove image file after result
	[os.remove(f) for f in glob.glob("/workspace/bum/MetricLearningIdentification/datasets_copy/OpenSetCows2020/images/test/*/*")]
 
	# Write it out to the console so that subprocess can pick them up and close
	# sys.stdout.write(f"Accuracy={str(accuracy)}")
	dt.close()
	print("러닝 타임 : {}".format(time.time() - start_time))
	print('Done')
	return result_label
	# sys.stdout.write("Done.")
	# sys.stdout.flush()
	# sys.exit(0)

def read_all_file(path):
    output = os.listdir(path)
    file_list = []

    for i in output:
        if os.path.isdir(path+"/"+i):
            file_list.extend(read_all_file(path+"/"+i))
        elif os.path.isfile(path+"/"+i):
            file_list.append(path+"/"+i)

    return file_list

def copy_all_file(file_list, new_path):
    dir_list = os.listdir('/workspace/bum/MetricLearningIdentification/datasets_copy/OpenSetCows2020/images/test')
    # print(dir_list)
    for src_path in file_list:
        file = src_path.split("/")[-1]
        # print(src_path)
        # print(new_path)
        # print(file_list)
        # print(file)
        for i in range(len(dir_list)):
        	shutil.copyfile(src_path, new_path+dir_list[i]+"/"+file)
        	# print(dir_list[i])
        	# print(new_path+dir_list[0]+file)
    
    # for i in range(len(dir_list)):
    #     shutil.copyfile(img, first_dst+dir_list[i]+'/test.jpg')

def register_cow(train_embeddings, train_labels, test_embeddings, file_count):
    # print(test_embeddings[1] == test_embeddings[2]) #False 테스트 1번이미지, 2번이미지 추출값
    # print(test_embeddings[1] == test_embeddings[3]) #False 테스트 1번이미지, 3번이미지 추출값
    # print(test_embeddings[1] == test_embeddings[4]) #True 테스트 1번이미지, 1번이미지 추출값
    
    new_label = train_labels[-1]+1
    for i in range(1, file_count+1):
        re_test_embedding = test_embeddings[i].reshape((1, -1))
        # print(re_test_embedding.shape)
        # add_embeddings = np.append(train_embeddings, re_test_embedding, axis=0)
        # add_labels = np.append(train_labels, new_label)
        train_embeddings = np.append(train_embeddings, re_test_embedding, axis=0)
        train_labels = np.append(train_labels, new_label)
    
    # make new label folder
    os.makedirs(f'/workspace/bum/MetricLearningIdentification/datasets_copy/OpenSetCows2020/images/test/{int(new_label)}', exist_ok=True)
    os.makedirs(f'/workspace/testFastAPI/images/{int(new_label)}', exist_ok=True)
    
    # profile photo copy
    src_path = '/workspace/testFastAPI/test2/0.jpg'
    new_path = '/workspace/testFastAPI/images/'
    shutil.copyfile(src_path, new_path+str(int(new_label))+"/"+"0.jpg")
    # print(f"{new_path}{int(new_label)}/0.jpg")
    
    
    # save new data
    save_path = os.path.join('/workspace/bum/MetricLearningIdentification/output/fold_0', 'train_embeddings.npz')
    np.savez(save_path, embeddings=train_embeddings, labels=train_labels)
    
    # print(train_labels)
    # print(f'=========================add_embeddings shape{train_embeddings.shape}')
    # print(f'=========================add_labels shape{train_labels.shape}')
    # print(f'=========================train_embeddings shape{train_embeddings.shape}')
    # print(f'=========================train_labels shape{train_labels.shape}')

# Use KNN to classify the embedding space
def KNNAccuracy(train_embeddings, train_labels, test_embeddings, test_labels ,n_neighbors=7): #n_neighbors default = 5
    # Define the KNN classifier
    neigh = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-4)
    # Give it the embeddings and labels of the training set
    neigh.fit(train_embeddings, train_labels)

    # Total number of testing instances
    total = len(test_labels-1)

    # Get the predictions from KNN
    predictions = neigh.predict(test_embeddings)
    # print(test_embeddings)
    p1 = predictions.astype(int)
    # print(predictions)
    # print(f'test: {predictions[0]}')
    # print(train_labels)
    # print(test_labels)
    
    conf_list = []
    fail_list = []
    vic_list =[]
    c_count = 0
    threshold = 0.75
    
    y_predict = neigh.predict_proba(test_embeddings)
    # print(type(p1))
    # print(type(y_predict))
    
    # p1 = np.delete(p1, 0)
    # y_predict = np.delete(y_predict, 0)
    
    # print(p1)
    print(y_predict)
    
    for i in range(len(p1)):
        # print(y_predict)
        confidence = y_predict[i][y_predict[i].argmax()]
        # print(y_predict[i])
        # print(len(y_predict[i]))
        # print(confidence)
        if p1[i] == test_labels[i] and confidence >= threshold:
            vic = 'success'
            vic_list.append(p1[i])
            conf_list.append([p1[i], round(confidence, 3)])
            c_count += 1
        else:
            vic = 'failed'
            fail_list.append([f'{p1[i]} : {round(confidence, 3)} : {vic}'])
        # conf_list.append(f'{p1[i]} : {round(confidence, 3)} : {vic}')
    
    print(vic_list)
    print(conf_list)
    # print(fail_list)
    # print(len(fail_list))
    # print(min(vic_list))
    if len(vic_list) > 0:
        vic_label = vic_list[0]
    else:
        vic_label = None
    
    # How many were correct?
    correct = (predictions == test_labels).sum()

    # Compute accuracy
    # accuracy = (float(correct) / total) * 100
    accuracy = (float(c_count) / total) * 100
    
    # print(f'======================        {accuracy}        ======================')
    # print(f'======================        {accuracy2}        ======================')
    
    return accuracy, predictions, vic_label

# Infer the embeddings for a given dataset
def inferEmbeddings(args, dataset, split):
	# Wrap up the dataset in a PyTorch dataset loader
	data_loader = data.DataLoader(dataset, batch_size=args.batch_size, num_workers=6, shuffle=False)
	
	# Define our embeddings model
	model = resnet50(pretrained=True, num_classes=dataset.getNumClasses(), ckpt_path=args.model_path, embedding_size=args.embedding_size)
	# print(dataset.getNumClasses())
	# Put the model on the GPU and in evaluation mode
	model.cuda()
	model.eval()
	
	# timage = np.array(cv2.imread('/workspace/bum/MetricLearningIdentification/datasets_copy/OpenSetCows2020/images/test/002/2.jpg'))
	# timage = cv2.resize(timage, dsize=(214, 214),interpolation=cv2.INTER_LINEAR)
	# timage_swap = np.swapaxes(timage, 0,2)
	# timage_swap = np.expand_dims(timage_swap, axis=0)
 
	# test_tensor = torch.from_numpy(timage_swap).type(torch.cuda.FloatTensor)
	# test_result = model(test_tensor)
	# print(model(test_result))
	# print(model(test_result).shape)
	
 
	# Embeddings/labels to be stored on the testing set
	outputs_embedding = np.zeros((1,args.embedding_size))
	labels_embedding = np.zeros((1))
	total = 0
	correct = 0

 
	# Iterate through the testing portion of the dataset and get
	for images, _, _, labels, _ in tqdm(data_loader, desc=f"Inferring {split} embeddings"):
		# Put the images on the GPU and express them as PyTorch variables
		# print(type(images))
		# print(images.shape)
  
		images = Variable(images.cuda())
		
  		# Get the embeddings of this batch of images
		outputs = model(images)

		# print(images.shape)
		# print(outputs)
		# if co == 0:
		# 	break
		
		# Express embeddings in numpy form
		embeddings = outputs.data
		embeddings = embeddings.cpu().numpy()

		# Convert labels to readable numpy form
		labels = labels.view(len(labels))
		# print(labels)
		labels = labels.cpu().numpy()
		# print(labels)

		# Store testing data on this batch ready to be evaluated
		outputs_embedding = np.concatenate((outputs_embedding,embeddings), axis=0)
		labels_embedding = np.concatenate((labels_embedding,labels), axis=0)
		# print(outputs_embedding)
		# print(labels_embedding)
	
	# If we're supposed to be saving the embeddings and labels to file
	if args.save_embeddings:
		# Construct the save path
		save_path = os.path.join(args.save_path, f"{split}_embeddings.npz")
		
		# Save the embeddings to a numpy array
		np.savez(save_path,  embeddings=outputs_embedding, labels=labels_embedding)

	return outputs_embedding, labels_embedding


# Main/entry method
if __name__ == '__main__':
	# Collate command line arguments
	parser = argparse.ArgumentParser(description='Params')

	# Required arguments
	parser.add_argument('--model_path', nargs='?', type=str, default="/workspace/bum/MetricLearningIdentification/output/fold_0/best_model_state.pkl", 
						help='Path to the saved model to load weights from')
	parser.add_argument('--folds_file', type=str, default="/workspace/bum/MetricLearningIdentification/datasets_copy/OpenSetCows2020/splits/10-90-custom.json",
						help="The file containing known/unknown splits")
	parser.add_argument('--save_path', type=str, default='/workspace/bum/MetricLearningIdentification/output/fold_0',
						help="Where to store the embeddings")

	parser.add_argument('--dataset', nargs='?', type=str, default='only_test_OpenSetCows2020', 
						help='Which dataset to use')
	parser.add_argument('--batch_size', nargs='?', type=int, default=16,
						help='Batch Size')
	parser.add_argument('--embedding_size', nargs='?', type=int, default=128, 
						help='Size of the dense layer for inference')
	parser.add_argument('--current_fold', type=int, default=0,
						help="The current fold we'd like to test on")
	parser.add_argument('--save_embeddings', type=bool, default=True,
						help="Should we save the embeddings to file")
	args = parser.parse_args()


	# print(args.model_path)
	# print(args.folds_file)
	# print(args.save_path)
	# Let's infer some embeddings
	# evaluateModel(args)