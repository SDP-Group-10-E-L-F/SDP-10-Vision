from PIL import Image 
import os
import sys


# cwd = os.getcwd()  # Get the current working directory (cwd)
# path = cwd + '/images/'
# files = os.listdir(path)  # Get all the files in that directory
#
# for file in files:
#     if file.endswith('.png'):
#         print(file)
#         img = Image.open(path + file)
#         file_name, file_ext = os.path.splitext(file)
#         rgb_img = img.convert('RGB')
#         rgb_img.save(path + file_name+'.jpg')
#         os.remove(path + file)
#         continue
#     else:
#         continue


cwd = os.getcwd()  # Get the current working directory (cwd)
path = cwd + '/labels/'
files = os.listdir(path)  # Get all the files in that directory

for file in files:
    if '.jpg.txt' in file:
        print(file + " ---> " + file.replace(".jpg.txt", ".txt"))
        old = os.path.join(path, file)
        new = os.path.join(path, file.replace(".jpg.txt", ".txt"))
        os.rename(old, new)
        continue
    else:
        continue

# def img2label_paths(img_paths):
#     # Define label paths as a function of image paths
#     sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
#     return ['txt'.join(x.replace(sa, sb, 1).rsplit(x.split('.')[-1], 1)) for x in img_paths]
#
# if __name__ == '__main__':
#     cwd = os.getcwd()  # Get the current working directory (cwd)
#     path = cwd + '/images/'
#     print(img2label_paths(path))

