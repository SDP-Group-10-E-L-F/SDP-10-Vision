import os
import glob
import shutil

def classes_statistics(dataset):
    cwd = os.getcwd()
    path = cwd + '/' + dataset
    train_labels = path + '/train/labels'
    test_labels = path + '/test/labels'
    valid_labels = path + '/valid/labels'
    all_labels = [train_labels, test_labels, valid_labels]
    stat = {}
    all = []
    for labels in all_labels:
        for label_file in glob.glob(f"{labels}/*.txt"):
            f = open(label_file, "r")
            lines = f.readlines()
            if len(lines) != 0:
                return


def remove_unwant(dataset, unwant_label):
    cwd = os.getcwd()
    path = cwd + '/' + dataset
    train_labels = path + '/train/labels'
    test_labels = path + '/test/labels'
    valid_labels = path + '/valid/labels'
    all_labels = [train_labels, test_labels, valid_labels]
    total_count = 0
    for labels in all_labels:
        for label_file in glob.glob(f"{labels}/*.txt"):
            f = open(label_file, "r")
            line = f.readlines()
            if len(line) != 0:
                line_list = line[0].split(" ")
                if line_list[0] == str(unwant_label):
                    total_count += 1
                    print("Removing unwant data")
                    img_file = os.path.splitext(label_file)[0].replace("labels", "images") + '.jpg'
                    print(img_file)
                    os.remove(img_file)
                    print(label_file)
                    os.remove(label_file)
    print(f"Task Completed: removed {total_count} unwant data ")


def replace_labels(dataset, prevLabels, newLabel):
    cwd = os.getcwd()
    path = cwd + '/' + dataset
    train_labels = path + '/train/labels'
    test_labels = path + '/test/labels'
    valid_labels = path + '/valid/labels'
    all_labels = [train_labels, test_labels, valid_labels]
    for labels in all_labels:
        for file in glob.glob(f"{labels}/*.txt"):
            read = open(file, 'r')
            lines = read.readlines()
            read.close()
            newlines = []
            for line in lines:
                if int(line[0]) != newLabel and int(line[0]) in prevLabels:
                    print(f"Replacing {line[0]} with {newLabel}")
                    newlines.append(f'{newLabel}' + line[1:])
                else:
                    newlines.append(line)
            write = open(file, 'w')
            write.writelines(newlines)
            write.close()


def unify_class(dataset, prevLabels, targetLabel):
    cwd = os.getcwd()
    path = cwd + '/' + dataset
    train_labels = path + '/train/labels'
    test_labels = path + '/test/labels'
    valid_labels = path + '/valid/labels'
    all_labels = [train_labels, test_labels, valid_labels]
    for labels in all_labels:
        for file in glob.glob(f"{labels}/*.txt"):
            f = open(file, "r+")
            line = f.readlines()
            if len(line) != 0:
                line_list = line[0].split(" ")
                currLabel = line_list[0]
                if currLabel != targetLabel and int(currLabel) in prevLabels:
                    line_list[0] = str(targetLabel)
                new_line = ' '.join(line_list)
                f.seek(0)
                f.truncate()
                print(f"Replacing [{line[0]}] with [{new_line}]")
                f.write(new_line)


def copy_batch(sources, destination):
    """
    Args:
        sources: list of source folders
        destination: the folder we want to copy file to
    """
    # Create destination folder
    os.makedirs(destination)
    for source in sources:
        for file in os.listdir(source):
            source_file = f"{source}/{file}"
            if os.path.isfile(source_file):
                shutil.copy(source_file, destination)
                print(f"{source_file} copied")


def merge_dataset(list_dataset, new_name):
    """
    Args:
        list_dataset:
        new_name:
    """
    cwd = os.getcwd()
    train_img_sets = [f"{cwd}/{dataset}/train/images" for dataset in list_dataset]
    test_img_sets = [f"{cwd}/{dataset}/test/images" for dataset in list_dataset]
    valid_img_sets = [f"{cwd}/{dataset}/valid/images" for dataset in list_dataset]

    train_label_sets = [f"{cwd}/{dataset}/train/labels" for dataset in list_dataset]
    test_label_sets = [f"{cwd}/{dataset}/test/labels" for dataset in list_dataset]
    valid_label_sets = [f"{cwd}/{dataset}/valid/labels" for dataset in list_dataset]

    print("Creating train set")
    merged_train_imgset = f"{cwd}/{new_name}/train/images"
    merged_train_labelset = f"{cwd}/{new_name}/train/labels"
    copy_batch(train_img_sets, merged_train_imgset)
    copy_batch(train_label_sets, merged_train_labelset)

    print("Creating test set")
    merged_test_imgset = f"{cwd}/{new_name}/test/images"
    merged_test_labelset = f"{cwd}/{new_name}/test/labels"
    copy_batch(test_img_sets, merged_test_imgset)
    copy_batch(test_label_sets, merged_test_labelset)

    print("Creating valid set")
    merged_valid_imgset = f"{cwd}/{new_name}/valid/images"
    merged_valid_labelset = f"{cwd}/{new_name}/valid/labels"
    copy_batch(valid_img_sets, merged_valid_imgset)
    copy_batch(valid_label_sets, merged_valid_labelset)


if __name__ == '__main__':
    # merge_dataset(['longsleeve','Sweatshirt'], 'longsleeve2')
    # remove_unwant('longsleeve3', 2)
    # remove_unwant('longsleeve3', 3)
    # remove_unwant('longsleeve3', 5)
    # replace_labels('longsleeve3', [0,4], 1)
    # merge_dataset(['longsleeve2', 'longsleeve3'], 'longsleeve4')
    # replace_labels('longsleeve5', [0], 1)
    # merge_dataset(['longsleeve4', 'longsleeve5'], 'longsleeve6')

    # Shorts: label 2
    # for i in [0,1,2,4,5]:
    #     remove_unwant('deepfashion', i)
    # replace_labels('deepfashion', [3], 2)

    #Trousers: label 3
    # for i in [0,1,3,4]:
    #     remove_unwant('pants', i)
    # replace_labels('pants', [2], 3)

    # for i in [1,2]:
    #     remove_unwant('trousers2', i)
    # replace_labels('trousers2', [0], 3)

    #Hands: label 4
    # replace_labels('Hands1', [0,1,2,3,5], 4)


    merge_dataset(['Tshirt1', 'longsleeve6', 'shorts1', 'trousers3', 'Hands1'], 'All_Dataset')
