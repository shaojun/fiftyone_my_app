import argparse
import datetime
import time
import os
import shutil
from typing import List


def getKeySampleFileNamesFromFilesInDir(dirPath):
    allFiles = os.listdir(dirPath)
    allFiles.sort(key=lambda x: x.split("___")[1]+"___"+x.split("___")[2])
    print('total read files count: {}'.format(len(allFiles)))
    allFilesGroupByFileName = {}
    result = []
    for file in allFiles:
        groupName = file.split("___")[1]+"___"+file.split("___")[2][:14]
        if groupName not in allFilesGroupByFileName:
            allFilesGroupByFileName[groupName] = []
        allFilesGroupByFileName[groupName].append(file)

    
    for group in allFilesGroupByFileName:
        if len(allFilesGroupByFileName[group]) == 1:
            result.append(allFilesGroupByFileName[group][0])
        if len(allFilesGroupByFileName[group]) == 2:
            result.append(allFilesGroupByFileName[group][0])
            result.append(allFilesGroupByFileName[group][1])
        if len(allFilesGroupByFileName[group]) == 3:
            result.append(allFilesGroupByFileName[group][0])
            result.append(allFilesGroupByFileName[group][-1])
        if len(allFilesGroupByFileName[group]) > 3:
            result.append(allFilesGroupByFileName[group][0])
            result.append(allFilesGroupByFileName[group][int(
                len(allFilesGroupByFileName[group])/2)])
            result.append(allFilesGroupByFileName[group][-1])

    floor_confid = 0.12
    ceiling_confid = 0.8
    removing_index = []
    for r in result:
        confid = float(r.split("___")[0])
        if '0.0599___E1600738913721257985___2023_02_16_14_57_39' in r:
            pass
        if confid < floor_confid or confid > ceiling_confid:
            removing_index.append(r)
            
    for r in removing_index:
        result.remove(r)
    print('total filtered files count: {}'.format(len(result)))
    return result


if __name__ == '__main__':
    test_divide = int(3/2)
    test_divide = int(9/2)
    base_dir = '/home/shawn/Downloads/'
    output_dir = os.path.join(base_dir, "ebic_image_samples_output")
    file_names = getKeySampleFileNamesFromFilesInDir(os.path.join(base_dir,"ebic_image_samples"))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    for fn in file_names:
        shutil.copyfile(os.path.join(os.path.join(base_dir,"ebic_image_samples"), fn),
                        os.path.join(output_dir, fn))


def createDirIfNotExist(dirPath):
    if not os.path.exists(dirPath):
        os.mkdir(dirPath)