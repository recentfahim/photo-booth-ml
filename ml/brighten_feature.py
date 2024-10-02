import sys
import subprocess
import shutil
import os
import cv2
import numpy as np

python_executable = sys.executable


def apply_brighten_1(img_folder, img_name):
    name = img_name.split('.')[0]
    source = os.path.join(os.getcwd(), img_folder, img_name)
    image = cv2.imread(source, 1)  
    added_image = np.concatenate([image, image], 1)
    
    temp = create_dir(name)
    destination = os.path.join(temp, name, 'test', img_name)
    cv2.imwrite(destination, added_image)
    
    test_dir = os.path.join(temp, name)
        
    command = [
        python_executable,
        "filter/brighten/test.py",
        "--dataroot",
        test_dir,
        "--model",
        "pix2pix",
        "--direction",
        "AtoB",
        "--name",
        "brighten_pix2pix",
        "--checkpoint",
        "filter/brighten/pretrained_model/Brighten/First/",
        "--results_dir",
        "filter/brighten/temp_results/",
        "--aspect_ratio",
        "1.0",
        "--load_size",
        "1024",
        "--crop_size",
        "1024"
    ]

    subprocess.run(command)

    result_dir = os.path.join(os.getcwd(), 'filter', 'brighten', 'temp_results', 'brighten_pix2pix','test_latest', 'images')

    if os.path.exists(os.path.join(result_dir, name + '_fake_B.png')):
        os.rename(os.path.join(result_dir, name + '_fake_B.png'), os.path.join(result_dir, name + '.png'))
    
    img = cv2.imread(os.path.join(result_dir, name + '.png'))
    h, w, _ = image.shape
    resized = cv2.resize(img, (w, h), interpolation= cv2.INTER_LINEAR)

    output_destination = os.path.join(os.getcwd(), 'static', 'filter_image', name + '.png')

    cv2.imwrite(output_destination, resized)
    



def apply_brighten_2(img_folder, img_name):
    name = img_name.split('.')[0]
    source = os.path.join(os.getcwd(), img_folder, img_name)
    image = cv2.imread(source, 1)  
    added_image = np.concatenate([image, image], 1)
    
    temp = create_dir(name)
    destination = os.path.join(temp, name, 'test', img_name)
    cv2.imwrite(destination, added_image)
    
    test_dir = os.path.join(temp, name)
        
    command = [
        python_executable,
        "filter/brighten/test.py",
        "--dataroot",
        test_dir,
        "--model",
        "pix2pix",
        "--direction",
        "AtoB",
        "--name",
        "brighten_pix2pix",
        "--checkpoint",
        "filter/brighten/pretrained_model/Brighten/Second/",
        "--results_dir",
        "filter/brighten/temp_results/",
        "--aspect_ratio",
        "1.0",
        "--load_size",
        "1024",
        "--crop_size",
        "1024"
    ]

    subprocess.run(command)

    result_dir = os.path.join(os.getcwd(), 'filter', 'brighten', 'temp_results', 'brighten_pix2pix','test_latest', 'images')

    if os.path.exists(os.path.join(result_dir, name + '_fake_B.png')):
        os.rename(os.path.join(result_dir, name + '_fake_B.png'), os.path.join(result_dir, name + '.png'))
    
    img = cv2.imread(os.path.join(result_dir, name + '.png'))
    h, w, _ = image.shape
    resized = cv2.resize(img, (w, h), interpolation= cv2.INTER_LINEAR)

    output_destination = os.path.join(os.getcwd(), 'static', 'filter_image', name + '.png')

    cv2.imwrite(output_destination, resized)




def apply_brighten_3(img_folder, img_name):
    name = img_name.split('.')[0]
    source = os.path.join(os.getcwd(), img_folder, img_name)
    image = cv2.imread(source, 1)  
    added_image = np.concatenate([image, image], 1)
    
    temp = create_dir(name)
    destination = os.path.join(temp, name, 'test', img_name)
    cv2.imwrite(destination, added_image)
    
    test_dir = os.path.join(temp, name)
        
    command = [
        python_executable,
        "filter/brighten/test.py",
        "--dataroot",
        test_dir,
        "--model",
        "pix2pix",
        "--direction",
        "AtoB",
        "--name",
        "brighten_pix2pix",
        "--checkpoint",
        "filter/brighten/pretrained_model/Brighten/Third/",
        "--results_dir",
        "filter/brighten/temp_results/",
        "--aspect_ratio",
        "1.0",
        "--load_size",
        "1024",
        "--crop_size",
        "1024"
    ]

    subprocess.run(command)

    result_dir = os.path.join(os.getcwd(), 'filter', 'brighten', 'temp_results', 'brighten_pix2pix','test_latest', 'images')

    if os.path.exists(os.path.join(result_dir, name + '_fake_B.png')):
        os.rename(os.path.join(result_dir, name + '_fake_B.png'), os.path.join(result_dir, name + '.png'))
    
    img = cv2.imread(os.path.join(result_dir, name + '.png'))
    h, w, _ = image.shape
    resized = cv2.resize(img, (w, h), interpolation= cv2.INTER_LINEAR)

    output_destination = os.path.join(os.getcwd(), 'static', 'filter_image', name + '.png')

    cv2.imwrite(output_destination, resized)



def apply_brighten_4(img_folder, img_name):
    name = img_name.split('.')[0]
    source = os.path.join(os.getcwd(), img_folder, img_name)
    image = cv2.imread(source, 1)  
    added_image = np.concatenate([image, image], 1)
    
    temp = create_dir(name)
    destination = os.path.join(temp, name, 'test', img_name)
    cv2.imwrite(destination, added_image)
    
    test_dir = os.path.join(temp, name)
        
    command = [
        python_executable,
        "filter/brighten/test.py",
        "--dataroot",
        test_dir,
        "--model",
        "pix2pix",
        "--direction",
        "AtoB",
        "--name",
        "brighten_pix2pix",
        "--checkpoint",
        "filter/brighten/pretrained_model/Brighten/Fourth/",
        "--results_dir",
        "filter/brighten/temp_results/",
        "--aspect_ratio",
        "1.0",
        "--load_size",
        "1024",
        "--crop_size",
        "1024"
    ]

    subprocess.run(command)

    result_dir = os.path.join(os.getcwd(), 'filter', 'brighten', 'temp_results', 'brighten_pix2pix','test_latest', 'images')

    if os.path.exists(os.path.join(result_dir, name + '_fake_B.png')):
        os.rename(os.path.join(result_dir, name + '_fake_B.png'), os.path.join(result_dir, name + '.png'))
    
    img = cv2.imread(os.path.join(result_dir, name + '.png'))
    h, w, _ = image.shape
    resized = cv2.resize(img, (w, h), interpolation= cv2.INTER_LINEAR)

    output_destination = os.path.join(os.getcwd(), 'static', 'filter_image', name + '.png')

    cv2.imwrite(output_destination, resized)




def apply_beauty_1(img_folder, img_name):
    name = img_name.split('.')[0]
    source = os.path.join(os.getcwd(), img_folder, img_name)
    image = cv2.imread(source, 1)  
    added_image = np.concatenate([image, image], 1)
    
    temp = create_dir(name)
    destination = os.path.join(temp, name, 'test', img_name)
    cv2.imwrite(destination, added_image)
    
    test_dir = os.path.join(temp, name)
        
    command = [
        python_executable,
        "filter/brighten/test.py",
        "--dataroot",
        test_dir,
        "--model",
        "pix2pix",
        "--direction",
        "AtoB",
        "--name",
        "beauty_pix2pix",
        "--checkpoint",
        "filter/brighten/pretrained_model/Beauty/1st/",
        "--results_dir",
        "filter/brighten/temp_results/",
        "--aspect_ratio",
        "1.0",
        "--load_size",
        "1024",
        "--crop_size",
        "1024"
    ]

    subprocess.run(command)

    result_dir = os.path.join(os.getcwd(), 'filter', 'brighten', 'temp_results', 'beauty_pix2pix','test_latest', 'images')

    if os.path.exists(os.path.join(result_dir, name + '_fake_B.png')):
        os.rename(os.path.join(result_dir, name + '_fake_B.png'), os.path.join(result_dir, name + '.png'))
    
    img = cv2.imread(os.path.join(result_dir, name + '.png'))
    h, w, _ = image.shape
    resized = cv2.resize(img, (w, h), interpolation= cv2.INTER_LINEAR)

    output_destination = os.path.join(os.getcwd(), 'static', 'filter_image', name + '.png')

    cv2.imwrite(output_destination, resized)



def apply_beauty_2(img_folder, img_name):
    name = img_name.split('.')[0]
    source = os.path.join(os.getcwd(), img_folder, img_name)
    image = cv2.imread(source, 1)  
    added_image = np.concatenate([image, image], 1)
    
    temp = create_dir(name)
    destination = os.path.join(temp, name, 'test', img_name)
    cv2.imwrite(destination, added_image)
    
    test_dir = os.path.join(temp, name)
        
    command = [
        python_executable,
        "filter/brighten/test.py",
        "--dataroot",
        test_dir,
        "--model",
        "pix2pix",
        "--direction",
        "AtoB",
        "--name",
        "beauty_pix2pix",
        "--checkpoint",
        "filter/brighten/pretrained_model/Beauty/2nd/",
        "--results_dir",
        "filter/brighten/temp_results/",
        "--aspect_ratio",
        "1.0",
        "--load_size",
        "1024",
        "--crop_size",
        "1024"
    ]

    subprocess.run(command)

    result_dir = os.path.join(os.getcwd(), 'filter', 'brighten', 'temp_results', 'beauty_pix2pix','test_latest', 'images')

    if os.path.exists(os.path.join(result_dir, name + '_fake_B.png')):
        os.rename(os.path.join(result_dir, name + '_fake_B.png'), os.path.join(result_dir, name + '.png'))
    
    img = cv2.imread(os.path.join(result_dir, name + '.png'))
    h, w, _ = image.shape
    resized = cv2.resize(img, (w, h), interpolation= cv2.INTER_LINEAR)

    output_destination = os.path.join(os.getcwd(), 'static', 'filter_image', name + '.png')

    cv2.imwrite(output_destination, resized)



def apply_beauty_3(img_folder, img_name):
    name = img_name.split('.')[0]
    source = os.path.join(os.getcwd(), img_folder, img_name)
    image = cv2.imread(source, 1)  
    added_image = np.concatenate([image, image], 1)
    
    temp = create_dir(name)
    destination = os.path.join(temp, name, 'test', img_name)
    cv2.imwrite(destination, added_image)
    
    test_dir = os.path.join(temp, name)
        
    command = [
        python_executable,
        "filter/brighten/test.py",
        "--dataroot",
        test_dir,
        "--model",
        "pix2pix",
        "--direction",
        "AtoB",
        "--name",
        "beauty_pix2pix",
        "--checkpoint",
        "filter/brighten/pretrained_model/Beauty/3rd/",
        "--results_dir",
        "filter/brighten/temp_results/",
        "--aspect_ratio",
        "1.0",
        "--load_size",
        "1024",
        "--crop_size",
        "1024"
    ]

    subprocess.run(command)

    result_dir = os.path.join(os.getcwd(), 'filter', 'brighten', 'temp_results', 'beauty_pix2pix','test_latest', 'images')

    if os.path.exists(os.path.join(result_dir, name + '_fake_B.png')):
        os.rename(os.path.join(result_dir, name + '_fake_B.png'), os.path.join(result_dir, name + '.png'))
    
    img = cv2.imread(os.path.join(result_dir, name + '.png'))
    h, w, _ = image.shape
    resized = cv2.resize(img, (w, h), interpolation= cv2.INTER_LINEAR)

    output_destination = os.path.join(os.getcwd(), 'static', 'filter_image', name + '.png')

    cv2.imwrite(output_destination, resized)



def apply_watermak_1(img_folder, img_name):
    name = img_name.split('.')[0]
    source = os.path.join(os.getcwd(), img_folder, img_name)
    image = cv2.imread(source, 1)  
    added_image = np.concatenate([image, image], 1)
    
    temp = create_dir(name)
    destination = os.path.join(temp, name, 'test', img_name)
    cv2.imwrite(destination, added_image)
    
    test_dir = os.path.join(temp, name)
        
    command = [
        python_executable,
        "filter/brighten/test.py",
        "--dataroot",
        test_dir,
        "--model",
        "pix2pix",
        "--direction",
        "AtoB",
        "--name",
        "watermark_pix2pix",
        "--checkpoint",
        "filter/brighten/pretrained_model/Watermark/1st/",
        "--results_dir",
        "filter/brighten/temp_results/",
        "--aspect_ratio",
        "1.0",
        "--load_size",
        "1024",
        "--crop_size",
        "1024"
    ]

    subprocess.run(command)

    result_dir = os.path.join(os.getcwd(), 'filter', 'brighten', 'temp_results', 'watermark_pix2pix','test_latest', 'images')

    if os.path.exists(os.path.join(result_dir, name + '_fake_B.png')):
        os.rename(os.path.join(result_dir, name + '_fake_B.png'), os.path.join(result_dir, name + '.png'))
    
    img = cv2.imread(os.path.join(result_dir, name + '.png'))
    h, w, _ = image.shape
    resized = cv2.resize(img, (w, h), interpolation= cv2.INTER_LINEAR)

    output_destination = os.path.join(os.getcwd(), 'static', 'filter_image', name + '.png')

    cv2.imwrite(output_destination, resized)


def create_dir(name):
    temp = os.path.join(os.getcwd(), 'filter', 'brighten', 'temp_testset')
    if not os.path.exists(os.path.join(temp, name)):
        os.mkdir(os.path.join(temp, name))
        os.mkdir(os.path.join(temp, name, 'test'))
    
    return temp

def copy_input_image():
    pass


def rename_predicted_name():
    pass


def copy_output_image():
    pass