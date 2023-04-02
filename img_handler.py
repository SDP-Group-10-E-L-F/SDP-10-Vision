import os
from PIL import Image
from pyheif import read_heif
import subprocess
import pillow_heif
import numpy as np
import random
import time


def convert2jpg(dir_path):
    pillow_heif.register_heif_opener()
    # Loop through each file in the folder
    for file_name in os.listdir(dir_path):
        if file_name.endswith(".HEIC") or file_name.endswith(".heic"):

            # Open the HEIC image
            img = Image.open(os.path.join(dir_path, file_name))

            # # Convert the HEIC image to JPEG
            # jpeg_image = heic_image.convert("RGB")

            # Set the new filename to be the same as the original, but with a ".jpg" extension
            new_filename = os.path.splitext(file_name)[0] + ".jpg"

            # Save the JPEG image with the new filename
            print(f"Converting into JPEG image: {file_name}")
            img.save(os.path.join(dir_path, new_filename))

            # Delete the original HEIC file
            print(f"Removing {file_name}")
            os.remove(os.path.join(dir_path, file_name))


def reflect_all(dir_path):
    # Get a list of all the image filenames in the folder
    files = [f for f in os.listdir(dir_path) if
             os.path.isfile(os.path.join(dir_path, f)) and f.endswith(('.jpg', '.jpeg', '.png'))]

    # Loop through each image file and copy and reflect it
    for filename in files:
        # Open the image file
        img = Image.open(os.path.join(dir_path, filename))

        # Paste the original image onto the left half of the new image
        new_img = img

        # Flip the original image horizontally
        relected_img = new_img.transpose(method=Image.FLIP_LEFT_RIGHT)

        # Save the new image with a "_reflected" suffix
        new_filename = os.path.splitext(filename)[0] + "_reflected" + os.path.splitext(filename)[1]
        relected_img.save(os.path.join(dir_path, new_filename))


def data_augmentation(dir_path):
    # Get a list of all the image filenames in the folder
    files = [f for f in os.listdir(dir_path) if
             os.path.isfile(os.path.join(dir_path, f)) and f.endswith(('.jpg', '.jpeg', '.png'))]

    # Loop through each image file and copy and reflect it
    for filename in files:
        # Open the image file
        img = Image.open(os.path.join(dir_path, filename))

        # Paste the original image onto the left half of the new image
        hori_flipped_img = img
        verti_flipped_img = img
        rotate_90_img = img
        rotate_270_img = img

        # Flip the original image horizontally
        hori_flipped_img = hori_flipped_img.transpose(method=Image.FLIP_LEFT_RIGHT)
        verti_flipped_img = verti_flipped_img.transpose(method=Image.FLIP_TOP_BOTTOM)
        rotate_90_img = rotate_90_img.transpose(method=Image.ROTATE_90)
        rotate_270_img = rotate_270_img.transpose(method=Image.ROTATE_270)

        # hori_flipped_img_filename = os.path.splitext(filename)[0] + "_hori_flipped" + os.path.splitext(filename)[1]
        # hori_flipped_img.save(os.path.join(dir_path, hori_flipped_img_filename))

        verti_flipped_img_filename = os.path.splitext(filename)[0] + "_verti_flipped" + os.path.splitext(filename)[1]
        verti_flipped_img.save(os.path.join(dir_path, verti_flipped_img_filename))

        rotate_90_img_filename = os.path.splitext(filename)[0] + "_rotate_90" + os.path.splitext(filename)[1]
        rotate_90_img.save(os.path.join(dir_path, rotate_90_img_filename))

        rotate_270_img_filename = os.path.splitext(filename)[0] + "_rotate_270" + os.path.splitext(filename)[1]
        rotate_270_img.save(os.path.join(dir_path, rotate_270_img_filename))


def resize_imgs(dir_path):
    width, height = 400, 500
    for f in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, f)) and f.endswith(('.jpg')):
            image = Image.open(os.path.join(dir_path, f))
            print("Resizing")
            resized_img = image.resize((width, height))
            resized_img_name = os.path.splitext(f)[0] + "_resized" + os.path.splitext(f)[1]
            resized_img.save(os.path.join(dir_path, resized_img_name))
            print(f"Removing {f}")
            os.remove(os.path.join(dir_path, f))

    res = [f for f in os.listdir(dir_path) if
           os.path.isfile(os.path.join(dir_path, f)) and f.endswith(('.jpg'))]
    return res

def panel_assemble(right_panel_imgs, left_panel_imgs, middle_panel_imgs, wooden_panel_imgs):
    tmp_folder = os.getcwd() + "/Panel_Data/tmp/"
    result_folder = os.getcwd() + "/Panel_Data/result/"

    # Building first columns
    first_col_filenames = random.choices(left_panel_imgs, k=3)
    first_col_imgs = [Image.open(f) for f in first_col_filenames]
    first_col = Image.fromarray(np.vstack(first_col_imgs))
    first_col_path = tmp_folder + "a_1st_col_tmp.jpg"
    first_col.save(first_col_path)

    # Building second columns
    second_col_filenames = random.choices(middle_panel_imgs, k=2) + [random.choice(wooden_panel_imgs)]
    second_col_imgs = [Image.open(f) for f in second_col_filenames]
    second_col = Image.fromarray(np.vstack(second_col_imgs))
    second_col_path = tmp_folder + "b_2nd_col_tmp.jpg"
    second_col.save(second_col_path)

    # Building third columns
    third_col_filename = random.choices(right_panel_imgs, k=3)
    third_col_imgs = [Image.open(f) for f in third_col_filename]
    third_col = Image.fromarray(np.vstack(third_col_imgs))
    third_col_path = tmp_folder + "c_3rd_col_tmp.jpg"
    third_col.save(third_col_path)

    tmp_imgs = sorted([os.path.join(tmp_folder, f) for f in os.listdir(tmp_folder) if
                       os.path.isfile(os.path.join(tmp_folder, f)) and f.endswith(('.jpg'))])

    col_imgs = [Image.open(i) for i in tmp_imgs]
    imgs_comb = Image.fromarray(np.hstack(col_imgs))
    imgs_comb = imgs_comb.resize((400, 500))
    suffix = time.strftime('%Y%m%d%H%M%S') + str(round(time.time() * 10000000))
    imgs_comb.save(result_folder + f"empty_{suffix}.jpg")

    os.remove(first_col_path)
    os.remove(second_col_path)
    os.remove(third_col_path)

def get_imgs_list(dir_path):
    imgs_list = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if
                       os.path.isfile(os.path.join(dir_path, f)) and f.endswith(('.jpg'))]
    return imgs_list



if __name__ == '__main__':
    # Set the path of the folder containing the images
    # folder_list = ["Leftmove-Panel-resized",
    #                "Rightmove-Panel-resized",
    #                "Upmove-Panel-resized",
    #                "Background-resized"]
    #
    # for folder_name in folder_list:
    #     path = os.getcwd() + "/Panel_Data/" + folder_name
    #     # data_augmentation(path)
    #     resize_imgs(path)
    path = os.getcwd() + "/Panel_Data/"

    # wooden_panel_dir = path + "Wooden-Panel"
    # convert2jpg(wooden_panel_dir)
    # resize_imgs(wooden_panel_dir)
    # reflect_all(wooden_panel_dir)


    # # background_dir = path + "Background-resized"
    # right_panel_dir = path + "Rightmove-Panel-resized"
    # left_panel_dir = path + "Leftmove-Panel-resized"
    # middle_panel_dir = path + "Upmove-Panel-resized"
    # wooden_panel_dir = path + "Wooden-Panel-resized"
    #
    # # background_imgs = get_imgs_list(background_dir)
    # right_panel_imgs = get_imgs_list(right_panel_dir)
    # left_panel_imgs = get_imgs_list(left_panel_dir)
    # middle_panel_imgs = get_imgs_list(middle_panel_dir)
    # wooden_panel_imgs = get_imgs_list(wooden_panel_dir)
    # #
    # for i in range(0, 500):
    #     panel_assemble(right_panel_imgs, left_panel_imgs, middle_panel_imgs, wooden_panel_imgs)

    # data_augmentation(path + 'result')



    # tmp
    # row1_pngs = [f"hand{i}.png" for i in range(1,4)]
    # row1_imgs = [Image.open(f) for f in row1_pngs]
    # row1_imgs = [img.resize((400, 300)) for img in row1_imgs]
    # row1 = Image.fromarray(np.hstack(row1_imgs))
    # row1.save("r1.jpg")
    #
    # row2_pngs = [f"hand{i}.png" for i in range(4, 7)]
    # row2_imgs = [Image.open(f) for f in row2_pngs]
    # row2_imgs = [img.resize((400, 300)) for img in row2_imgs]
    # row2 = Image.fromarray(np.hstack(row2_imgs))
    # row2.save("r2.jpg")
    #
    # row_imgs = [Image.open(i) for i in ["r1.jpg", "r2.jpg"]]
    # imgs_comb = Image.fromarray(np.vstack(row_imgs))
    # imgs_comb.save("result.jpg")

    imgs =[Image.open(f) for f in ["PR.jpeg", "clothes_confusion_matrix.jpeg"]]
    imgs = [img.resize((1500, 1500)) for img in imgs]
    img = Image.fromarray(np.hstack(imgs))
    img.save("Cloth_result.jpeg")




