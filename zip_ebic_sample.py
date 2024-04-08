import argparse
import datetime
import time
import os
import shutil
from typing import List


def getKeySampleFileNamesFromFilesInDir(dirPath):
    all_ele_folders = os.listdir(dirPath)
    output_full_file_name = []
    for ele_folder in all_ele_folders:
        allFiles = os.listdir(os.path.join(dirPath, ele_folder))
        # allFiles.sort(key=lambda x: x.split("___")[1]+"___"+x.split("___")[2])

        allFilesGroupByFileName = {}

        for file in allFiles:
            # file name samples:
            #   0.81___2024_0407_0755_18_228___0755_18_736___E1776157223118966785.jpg
            #   background_0.30___2024_0407_0957_37_471___0957_37_935___E1776157223118966785.jpg
            min_part = file.split("___")[1][:14]
            edge_id_part = file.split("___")[-1]
            groupName = min_part+"___"+edge_id_part
            if groupName not in allFilesGroupByFileName:
                allFilesGroupByFileName[groupName] = []
            allFilesGroupByFileName[groupName].append(
                os.path.join(dirPath, ele_folder, file))

        for group in allFilesGroupByFileName:
            if len(allFilesGroupByFileName[group]) == 1:
                output_full_file_name.append(allFilesGroupByFileName[group][0])
            if len(allFilesGroupByFileName[group]) == 2:
                output_full_file_name.append(allFilesGroupByFileName[group][0])
                output_full_file_name.append(allFilesGroupByFileName[group][1])
            if len(allFilesGroupByFileName[group]) == 3:
                output_full_file_name.append(allFilesGroupByFileName[group][0])
                output_full_file_name.append(
                    allFilesGroupByFileName[group][-1])
            if len(allFilesGroupByFileName[group]) > 3:
                output_full_file_name.append(allFilesGroupByFileName[group][0])
                output_full_file_name.append(allFilesGroupByFileName[group][int(
                    len(allFilesGroupByFileName[group])/2)])
                output_full_file_name.append(
                    allFilesGroupByFileName[group][-1])
    print('total zipped files count: {}'.format(len(output_full_file_name)))
    floor_confid = 0.08
    ceiling_confid = 0.65
    removing_index = []
    for r in output_full_file_name:
        confid = float(r.split('/')[-1].split("___")[0])
        if '0.0599___E1600738913721257985___2023_02_16_14_57_39' in r:
            pass
        if confid < floor_confid or confid > ceiling_confid:
            removing_index.append(r)

    for r in removing_index:
        output_full_file_name.remove(r)
    print('total conid filtered files count: {}'.format(
        len(output_full_file_name)))
    return output_full_file_name


if __name__ == '__main__':
    test_divide = int(3/2)
    test_divide = int(9/2)
    base_dir = '/home/shao/Downloads/image_ground'
    # ebic_image_samples_copied_at_23_04_13_the_first_round_of_moved_dh_to_3060_machine
    output_dir = os.path.join(base_dir, "ebic_image_samples_output")
    file_full_names = getKeySampleFileNamesFromFilesInDir(os.path.join(
        base_dir, "ebic_image_samples_copied_at_23_04_13_the_first_round_of_moved_dh_to_3060_machine"))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for fn in file_full_names:
        shutil.copyfile(fn,
                        os.path.join(output_dir, fn.split('/')[-1]))


def createDirIfNotExist(dirPath):
    if not os.path.exists(dirPath):
        os.mkdir(dirPath)
