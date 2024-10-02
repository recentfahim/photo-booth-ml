import os
import uuid


image_path = 'static'
filter_image_path = os.path.join(image_path, 'filter_image')
original_image_path = os.path.join(image_path, 'original_image')


def filter_image_output(folder_name):
    folder_name = folder_name.split('/')[:-1]
    folder_name = '/'.join(folder_name)
    output_folder_name = folder_name + '/filter_image/'
    create_directory(output_folder_name)

    if (os.name == 'nt'):
        output_folder_name = 'D:\\Photo Booth\\photo-booth\\static\\filter_image\\'
    
    return output_folder_name


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def filter_request(filter_name, image):
    from filter.pencil_sketch import pencil_sketch
    from filter.cantoon_filter import cantoon_filter
    from filter.remove_bg import remove_background
    from filter.brighten.brighten_feature import apply_brighten_1, apply_brighten_2, apply_brighten_3, apply_brighten_4, apply_beauty_1, apply_beauty_2, apply_beauty_3, apply_watermak_1
    from filter.whitening import whitening_method1, whitening_method2, whitening_method3, whitening_method4, \
        whitening_method5, whitening_method6, whitening_method7, whitening_method8

    if not filter_name and not image:
        if not image:
            return {
                'message': 'Image is not found',
                'success': False
            }
        if not filter_name:
            return {
                'message': 'You did not enter any filter',
                'success': False
            }
    else:
        image_name = str(uuid.uuid4()) + '.' + image.filename.split('.')[-1]

        create_directory(original_image_path)

        image.save(os.path.join(original_image_path, image_name))

        if filter_name == 'cartoon':
            cantoon_filter(original_image_path, image_name)
        elif filter_name == 'pencil_sketch':
            pencil_sketch(original_image_path, image_name)
        elif filter_name == 'background_remove':
            remove_background(original_image_path, image_name)
        elif filter_name == 'brighten_1':
            apply_brighten_1(original_image_path, image_name)
        elif filter_name == 'brighten_2':
            apply_brighten_2(original_image_path, image_name)
        elif filter_name == 'brighten_3':
            apply_brighten_3(original_image_path, image_name)
        elif filter_name == 'brighten_4':
            apply_brighten_4(original_image_path, image_name)
        elif filter_name == 'beauty_1':
            apply_beauty_1(original_image_path, image_name)
        elif filter_name == 'beauty_2':
            apply_beauty_2(original_image_path, image_name)
        elif filter_name == 'beauty_3':
            apply_beauty_3(original_image_path, image_name)
        else:
            return {
                'message': 'You entered wrong filters. Please visit /available-filter/ to see the correct name',
                'success': False
            }

        return {
            'message': 'Filter applied and saved image in the server',
            'success': True,
            'image_name': image_name
        }


def get_image_response_path(image_name) -> dict:
    original_path = os.path.join(original_image_path, image_name)
    filter_path = os.path.join(filter_image_path, image_name)
    if not os.path.exists(filter_path):
        filter_path = filter_path.replace('.jpg', '.png')
    context = {
        'original_image': original_path,
        'filter_image': filter_path
    }

    return context
