import shutil
import os


img = r'/workspace/bum/30.jpg'
first_dst = r'/workspace/bum/MetricLearningIdentification/datasets_copy/OpenSetCows2020/images/test/'

# print(len(os.listdir('/workspace/bum/copy_test')))

dir_list = os.listdir('/workspace/bum/MetricLearningIdentification/datasets_copy/OpenSetCows2020/images/test')
# print(dir_list)
# print(first_dst+dir_list[0]+'/test.jpg')
for i in range(len(dir_list)):
    shutil.copyfile(img, first_dst+dir_list[i]+'/test.jpg')
