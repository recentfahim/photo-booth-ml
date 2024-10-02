import os


base_dataset_dir_a = os.path.join(os.getcwd(), 'prep', 'A')
base_dataset_dir_b = os.path.join(os.getcwd(), 'prep', 'B')

trainA = os.listdir(os.path.join(base_dataset_dir_a, 'train'))
testA = os.listdir(os.path.join(base_dataset_dir_a, 'test'))
valA = os.listdir(os.path.join(base_dataset_dir_a, 'val'))

trainB = os.listdir(os.path.join(base_dataset_dir_b, 'train'))
testB = os.listdir(os.path.join(base_dataset_dir_b, 'test'))
valB = os.listdir(os.path.join(base_dataset_dir_b, 'val'))

trainA_without_ext = [i.split('.')[0] for i in trainA]
testA_without_ext = [i.split('.')[0] for i in testA]
valA_without_ext = [i.split('.')[0] for i in valA]

trainB_without_ext = [i.split('.')[0] for i in trainB]
testB_without_ext = [i.split('.')[0] for i in testB]
valB_without_ext = [i.split('.')[0] for i in valB]

a_ext = '.jpg'
b_ext = '.png'

count = 1
for file in trainA_without_ext:
    if file in trainB_without_ext:
        img_name = 'img_' + str(count)
        new_img_a = os.path.join(base_dataset_dir_a, 'train', img_name + a_ext)
        old_img_a = os.path.join(base_dataset_dir_a, 'train', file + a_ext)
        new_img_b = os.path.join(base_dataset_dir_b, 'train', img_name + b_ext)
        old_img_b = os.path.join(base_dataset_dir_b, 'train', file + b_ext)
        os.rename(old_img_a, new_img_a)
        os.rename(old_img_b, new_img_b)
        count = count + 1


count = 1
for file in testA_without_ext:
    if file in testB_without_ext:
        img_name = 'img_' + str(count)
        new_img_a = os.path.join(base_dataset_dir_a, 'test', img_name + a_ext)
        old_img_a = os.path.join(base_dataset_dir_a, 'test', file + a_ext)
        new_img_b = os.path.join(base_dataset_dir_b, 'test', img_name + b_ext)
        old_img_b = os.path.join(base_dataset_dir_b, 'test', file + b_ext)
        os.rename(old_img_a, new_img_a)
        os.rename(old_img_b, new_img_b)
        count = count + 1



count = 1
for file in valA_without_ext:
    if file in valB_without_ext:
        img_name = 'img_' + str(count)
        new_img_a = os.path.join(base_dataset_dir_a, 'val', img_name + a_ext)
        old_img_a = os.path.join(base_dataset_dir_a, 'val', file + a_ext)
        new_img_b = os.path.join(base_dataset_dir_b, 'val', img_name + b_ext)
        old_img_b = os.path.join(base_dataset_dir_b, 'val', file + b_ext)
        os.rename(old_img_a, new_img_a)
        os.rename(old_img_b, new_img_b)
        count = count + 1
