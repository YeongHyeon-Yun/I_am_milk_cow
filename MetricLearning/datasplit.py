import splitfolders 

#### input dataset that want to split
input_folder = '/workspace/HToutput'  

output_folder= '/workspace/bum/MLI_HT/datasets_copy/OpenSetCows2020/images'

splitfolders.ratio(input_folder, output= output_folder, seed=1337, ratio = (0.9, 0.1))