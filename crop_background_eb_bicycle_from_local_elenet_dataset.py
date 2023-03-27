import fiftyone as fo
import os
import shutil
import random
import cv2
import math
import numpy as np
Mat = np.ndarray[int, np.dtype[np.generic]]


def resize_with_padding(
        image: np.ndarray,
        new_width: int,
        new_height: int,
):
    """
    Reads image data from a file and resizes it to the specified dimensions,
    preserving the aspect ratio and padding on the right and bottom as necessary.

    :param image: numpy array of image (pixel) data
    :param new_width:
    :param new_height:
    :return: resized image data and paddings (width and height)
    """

    # get the dimensions and aspect ratio
    original_height, original_width = image.shape[:2]
    aspect_ratio = original_width / original_height

    # determine the interpolation method we'll use
    if (original_height > new_height) or (original_width > new_width):
        # use a shrinking algorithm for interpolation
        interp = cv2.INTER_AREA
    else:
        # use a stretching algorithm for interpolation
        interp = cv2.INTER_CUBIC

    # if aspect_ratio <= 0.8:
    #     image = cv2.resize(
    #         image, (int(original_width*1.2), int(original_height*1.2)), interpolation=interp)

    # determine the new width and height (may differ from the width and
    # height arguments if using those doesn't preserve the aspect ratio)
    final_width = new_width
    final_height = round(final_width / aspect_ratio)
    if final_height > new_height:
        final_height = new_height
    final_width = round(final_height * aspect_ratio)

    # at this point we may be off by a few pixels, i.e. over
    # the specified new width or height values, so we'll clip
    # in order not to exceed the specified new dimensions
    final_width = min(final_width, new_width)
    final_height = min(final_height, new_height)

    # get the padding necessary to preserve aspect ratio
    pad_bottom = abs(new_height - final_height)
    pad_right = abs(new_width - final_width)

    # scale and pad the image
    scaled_img = cv2.resize(
        image, (final_width, final_height), interpolation=interp)
    # padded_img = cv2.copyMakeBorder(
    #     scaled_img, 0, pad_bottom, 0, pad_right,
    #     borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0], )

    # shao, we choose padding averagely on 4 boarders
    padded_img = cv2.copyMakeBorder(
        scaled_img,
        int(pad_bottom/2), int(pad_bottom / 2),
        int(pad_right/2), int(pad_right/2),
        borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0],
    )

    return padded_img, pad_right, pad_bottom


def _get_resized_image(
        image: np.ndarray,
        new_width: int,
        new_height: int,
):
    """
    Reads image data from a file and resizes it to the specified dimensions,
    preserving the aspect ratio and padding on the right and bottom as necessary.

    :param image: numpy array of image (pixel) data
    :param new_width:
    :param new_height:
    :return: resized image data and scale factors (width and height)
    """

    padded_img, pad_right, pad_bottom = resize_with_padding(
        image, new_width, new_height)

    # get the scaling factors that were used
    original_height, original_width = image.shape[:2]
    scale_x = (new_width - pad_right) / original_width
    scale_y = (new_height - pad_bottom) / original_height

    return padded_img, scale_x, scale_y


def resize_image(
        image_path: str,
        output_image_path: str,
        new_width: int,
        new_height: int,
) -> int:
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    return resize_image(image, output_image_path, new_width, new_height)


def resize_image(
        cv_image: Mat,
        output_image_path: str,
        new_width: int,
        new_height: int,
) -> int:
    """
    Resizes an image.

    :param file_name: file name of the image file
    :param input_images_dir: directory where image file is located
    :param output_images_dir: directory where the resized image file
        should be written
    :param new_width: new width to which the image should be resized
    :param new_height: new height to which the image should be resized
    :return: 0 to indicate successful completion
    """

    image = cv_image
    original_height, original_width = image.shape[:2]

    # resize if necessary
    if (original_width != new_width) or (original_height != new_height):
        image, _, _ = _get_resized_image(image, new_width, new_height)

    # write the scaled/padded image to file in the output directory
    resized_image_path = output_image_path
    cv2.imwrite(resized_image_path, image)

    return 0


def AreTwoRectAreaIntersect(rectA, rectB):
    # rectA middle point
    rectA_width = rectA['x1']-rectA['x0']
    rectA_height = rectA['y1']-rectA['y0']
    rectA_middle_point = (rectA['x0']+rectA_width/2,
                          rectA['y0']+rectA_height/2)

    # rectB middle point
    rectB_width = rectB['x1']-rectB['x0']
    rectB_height = rectB['y1']-rectB['y0']
    rectB_middle_point = (rectB['x0']+rectB_width/2,
                          rectB['y0']+rectB_height/2)

    threashold_distance = math.sqrt(math.pow(
        rectA_width/2 + rectB_width/2, 2) + math.pow(rectA_height/2 + rectB_height/2, 2))

    actual_distance = math.sqrt(math.pow(
        rectA_middle_point[0] - rectB_middle_point[0], 2) + math.pow(rectA_middle_point[1] - rectB_middle_point[1], 2))
    return actual_distance <= threashold_distance


def load_dataset_from_local_path(dataset_local_dir: str, dataset_name: str,
                                 force_reload_dataset_if_exists: bool = False):
    if dataset_name in fo.list_datasets() and force_reload_dataset_if_exists:
        exists_dataset = fo.load_dataset(dataset_name)
        print("dataset: {} already exists, will delete it...".format(dataset_name))
        exists_dataset.delete()
        print("     deleted with result: {}".format(exists_dataset.deleted))
    elif dataset_name in fo.list_datasets() and not force_reload_dataset_if_exists:
        return fo.load_dataset(dataset_name)
    dataset_type = fo.types.KITTIDetectionDataset
    dataset = fo.Dataset.from_dir(
        dataset_dir=dataset_local_dir,
        dataset_type=dataset_type,
        name=dataset_name,
    )
    dataset.persistent = True
    return dataset


if __name__ == '__main__':
    # the kitti dataset should follow this structure, otherwise the loaded dataset will be empty (and no error were shown) !!!
    # <dataset_dir>/
    # data/
    #     <uuid1>.<ext>
    #     <uuid2>.<ext>
    #     ...
    # labels/
    #     <uuid1>.txt
    #     <uuid2>.txt
    #     ...

    # you may see debug error like rm -rf ~/.fiftyone/
    src_dataset_local_path = "prepare_yolov5_dataset_from_kitti_format"
    # src_dataset_local_path = "test/elenet_1000_ds"

    dataset_name = "first_time1"
    load_dataset = load_dataset_from_local_path(
        src_dataset_local_path, dataset_name, True)
    # print(load_dataset)

    # app_config = fo.AppConfig()
    # app_config.show_attributes = True
    # session = fo.launch_app(load_dataset, config=app_config)

    target_export_dataset_local_path = "cropped_export".format(
        src_dataset_local_path)
    if not os.path.exists(target_export_dataset_local_path):
        os.mkdir(target_export_dataset_local_path)
    else:
        shutil.rmtree(target_export_dataset_local_path)
        os.mkdir(target_export_dataset_local_path)
        os.mkdir(os.path.join(target_export_dataset_local_path, "background"))
        os.mkdir(os.path.join(target_export_dataset_local_path, "electric_bicycle"))
        os.mkdir(os.path.join(target_export_dataset_local_path, "bicycle"))
        os.mkdir(os.path.join(target_export_dataset_local_path, "people"))
    view = load_dataset.view()

    successfully_background_cropped_times = 0
    successfully_people_cropped_times = 0
    successfully_eb_cropped_times = 0
    successfully_bicycle_cropped_times = 0
    skipped_crop_image_file_count = 0

    for sample in view.iter_samples(progress=True):
        img = cv2.imread(sample.filepath)
        img_h, img_w, c = img.shape
        eb_and_bi_object_boxes = []

        # start crop electric_bicycle and bicycle
        for det in sample.ground_truth.detections:
            if det.label == 'electric_bicycle' or det.label == 'bicycle':
                [x, y, w, h] = det.bounding_box
                x = int(x * img_w)
                y = int(y * img_h)
                h = int(img_h * h)
                w = int(img_w * w)

                if det.label == 'electric_bicycle':
                    successfully_eb_cropped_times += 1
                elif det.label == 'bicycle':
                    successfully_bicycle_cropped_times += 1
                cropped_img = img[y:y + h, x:x+w, :]
                output_filepath = os.path.join(
                    target_export_dataset_local_path + "/"+det.label+"/", sample.filename+"_____" + det.id+".png")
                # resize_image(cropped_img, output_filepath, 224, 224)
                cv2.imwrite(output_filepath, cropped_img)

                eb_and_bi_object_boxes.append(
                    {'x0': x, 'y0': y, 'x1': x+w, 'y1': y+h})
        # end crop electric_bicycle and bicycle

        # start crop people
        # many people treat as eb in classification model.
        if len(eb_and_bi_object_boxes) == 0:
            for det in sample.ground_truth.detections:
                if det.label == 'people':
                    [x, y, w, h] = det.bounding_box
                    x = int(x * img_w)
                    y = int(y * img_h)
                    h = int(img_h * h)
                    w = int(img_w * w)

                    cropped_img = img[y:y + h, x:x+w, :]
                    output_filepath = os.path.join(
                        target_export_dataset_local_path + "/people/", sample.filename+"_____" + det.id+".png")
                    # resize_image(cropped_img, output_filepath, 224, 224)
                    cv2.imwrite(output_filepath, cropped_img)
                    eb_and_bi_object_boxes.append(
                        {'x0': x, 'y0': y, 'x1': x+w, 'y1': y+h})
                    successfully_people_cropped_times += 1
        # end crop people

        # start crop bg
        max_trying_crop_times = 200
        useDynamicCropOutputSize = True
        if useDynamicCropOutputSize:
            crop_width = int(img_w/4)
            crop_height = int(img_h/3)
        else:
            crop_width = 224
            crop_height = 224
        random_cropping_times = 0
        while True:
            # the image center has the priority, so try to crop in center area, and the image height is 1280, width is 720.
            random_x = random.randint(0, img_w/2 - crop_width)
            random_y = random.randint(0, img_h - crop_height)
            random_box = {'x0': random_x, 'y0': random_y,
                          'x1': random_x+crop_width, 'y1': random_y+crop_height}
            random_cropping_times += 1
            if random_cropping_times >= max_trying_crop_times:
                skipped_crop_image_file_count += 1
                # print("skip the cropping as max cropping times reached for {}".format(
                #     sample.filename))
                break

            intersected = False
            for obj_box in eb_and_bi_object_boxes:
                if(AreTwoRectAreaIntersect(obj_box, random_box)):
                    # print("{} has a eb_box: (x0:{},y0:{},x1:{},y1:{}) intersect with proposed random_box: (x0:{},y0:{},x1:{},y1:{})".format(
                    #     sample.filename, eb_box['x0'], eb_box['y0'], eb_box['x1'], eb_box['y1'], random_box['x0'], random_box['y0'], random_box['x1'], random_box['y1']))
                    intersected = True
                    break
            if intersected:
                continue

            # random drop some cropping since it's getting too much
            if random_x % 10 <= 4:
                break
            cropped_img = img[random_y:random_y +
                              crop_height, random_x:random_x+crop_width, :]
            output_filepath = os.path.join(
                target_export_dataset_local_path+"/background/", sample.filename+"_____" + det.id+".png")
            retval = cv2.imwrite(output_filepath, cropped_img)
            if not retval:
                print("save cropped image: {} failed from original file: {}".format(
                    output_filepath, sample.filename))
            else:
                successfully_background_cropped_times += 1
            break
        # end crop bg
    print("TOTAL bg cropped times: {}, people cropped times: {}, eb cropped times: {}, bicycle cropped times: {}. Skipped_crop_image_file_count: {}".format(
        successfully_background_cropped_times,
        successfully_people_cropped_times,
        successfully_eb_cropped_times,
        successfully_bicycle_cropped_times,
        skipped_crop_image_file_count))
