import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_sceneflow_img(filepath):
    classes = [d for d in os.listdir(filepath) if os.path.isdir(os.path.join(filepath, d))]
    image = [img for img in classes if img.find('frames_cleanpass') > -1]
    disp = [dsp for dsp in classes if dsp.find('disparity') > -1]

    all_left_img = []
    all_right_img = []
    all_left_disp = []
    test_left_img = []
    test_right_img = []
    test_left_disp = []

    # monkaa_path = filepath + [x for x in image if 'monkaa' in x][0]
    # monkaa_disp = filepath + [x for x in disp if 'monkaa' in x][0]
    #
    #
    # monkaa_dir  = os.listdir(monkaa_path)
    #
    # for dd in monkaa_dir:
    #   for im in os.listdir(monkaa_path+'/'+dd+'/left/'):
    #    if is_image_file(monkaa_path+'/'+dd+'/left/'+im):
    #     all_left_img.append(monkaa_path+'/'+dd+'/left/'+im)
    #     all_left_disp.append(monkaa_disp+'/'+dd+'/left/'+im.split(".")[0]+'.pfm')
    #
    #   for im in os.listdir(monkaa_path+'/'+dd+'/right/'):
    #    if is_image_file(monkaa_path+'/'+dd+'/right/'+im):
    #     all_right_img.append(monkaa_path+'/'+dd+'/right/'+im)
    #
    # flying_path = filepath + [x for x in image if x == 'frames_cleanpass'][0]
    # flying_disp = filepath + [x for x in disp if x == 'frames_disparity'][0]
    # flying_dir = flying_path+'/TRAIN/'
    # subdir = ['A','B','C']
    #
    # for ss in subdir:
    #    flying = os.listdir(flying_dir+ss)
    #
    #    for ff in flying:
    #      imm_l = os.listdir(flying_dir+ss+'/'+ff+'/left/')
    #      for im in imm_l:
    #       if is_image_file(flying_dir+ss+'/'+ff+'/left/'+im):
    #         all_left_img.append(flying_dir+ss+'/'+ff+'/left/'+im)
    #
    #       all_left_disp.append(flying_disp+'/TRAIN/'+ss+'/'+ff+'/left/'+im.split(".")[0]+'.pfm')
    #
    #       if is_image_file(flying_dir+ss+'/'+ff+'/right/'+im):
    #         all_right_img.append(flying_dir+ss+'/'+ff+'/right/'+im)
    #
    # flying_dir = flying_path+'/TEST/'
    #
    # subdir = ['A','B','C']
    #
    # for ss in subdir:
    #    flying = os.listdir(flying_dir+ss)
    #
    #    for ff in flying:
    #      imm_l = os.listdir(flying_dir+ss+'/'+ff+'/left/')
    #      for im in imm_l:
    #       if is_image_file(flying_dir+ss+'/'+ff+'/left/'+im):
    #         test_left_img.append(flying_dir+ss+'/'+ff+'/left/'+im)
    #
    #       test_left_disp.append(flying_disp+'/TEST/'+ss+'/'+ff+'/left/'+im.split(".")[0]+'.pfm')
    #
    #       if is_image_file(flying_dir+ss+'/'+ff+'/right/'+im):
    #         test_right_img.append(flying_dir+ss+'/'+ff+'/right/'+im)

    driving_dir = filepath + [x for x in image if 'driving' in x][0] + '/'
    driving_disp = filepath + [x for x in disp if 'driving' in x][0]

    subdir1 = ['35mm_focallength', '15mm_focallength']
    subdir2 = ['scene_backwards', 'scene_forwards']
    subdir3 = ['fast', 'slow']

    for i in subdir1:
        for j in subdir2:
            for k in subdir3:
                imm_l = os.listdir(driving_dir + i + '/' + j + '/' + k + '/left/')
                for im in imm_l:
                    if is_image_file(driving_dir + i + '/' + j + '/' + k + '/left/' + im):
                        all_left_img.append(driving_dir + i + '/' + j + '/' + k + '/left/' + im)
                    all_left_disp.append(
                        driving_disp + '/' + i + '/' + j + '/' + k + '/left/' + im.split(".")[0] + '.pfm')

                    if is_image_file(driving_dir + i + '/' + j + '/' + k + '/right/' + im):
                        all_right_img.append(driving_dir + i + '/' + j + '/' + k + '/right/' + im)

    return all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp


def get_kitti_2015_img(filepath):
    left_fold = 'image_2/'
    right_fold = 'image_3/'
    disp_L = 'disp_occ_0/'
    disp_R = 'disp_occ_1/'

    image = [img for img in os.listdir(filepath + left_fold) if img.find('_10') > -1]

    left_train = [filepath + left_fold + img for img in image]
    right_train = [filepath + right_fold + img for img in image]
    disp_train_L = [filepath + disp_L + img for img in image]

    return left_train, right_train, disp_train_L


def get_kitti_2012_img(filepath):
    left_fold = 'colored_0/'
    right_fold = 'colored_1/'
    disp_noc = 'disp_occ/'

    image = [img for img in os.listdir(filepath + left_fold) if img.find('_10') > -1]

    train = image[:]
    val = image[160:]

    left_train = [filepath + left_fold + img for img in train]
    right_train = [filepath + right_fold + img for img in train]
    disp_train = [filepath + disp_noc + img for img in train]

    left_val = [filepath + left_fold + img for img in val]
    right_val = [filepath + right_fold + img for img in val]
    disp_val = [filepath + disp_noc + img for img in val]

    return left_train, right_train, disp_train, left_val, right_val, disp_val
