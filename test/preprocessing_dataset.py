import fiftyone as fo
import fiftyone.utils.image


def load_dataset_from_local_path(dataset_local_dir: str, dataset_name: str,
                                 force_reload_dataset_if_exists: bool = False):
    if dataset_name in fo.list_datasets() and force_reload_dataset_if_exists:
        exists_dataset = fo.load_dataset(dataset_name)
        print("dataset: {} already exists, will delete it...".format(dataset_name))
        exists_dataset.delete()
        print("     deleted with result: {}".format(exists_dataset.deleted))
    elif dataset_name in fo.list_datasets() and not force_reload_dataset_if_exists:
        return fo.load_dataset(dataset_name)
    dataset = fo.Dataset.from_dir(
        dataset_dir=dataset_local_dir,
        dataset_type=fo.types.KITTIDetectionDataset,
        name=dataset_name,
    )
    dataset.persistent = True
    return dataset


def reencode_to_png_in_place(dataset):
    # fiftyone.utils.image.reencode_images(
    #     dataset, ext='.png', force_reencode=True, delete_originals=True, num_workers=None, skip_failures=False)
    # size (width, height) for each image. One dimension can be -1, in which case the aspect ratio is preserved
    fiftyone.utils.image.transform_images(dataset, size=(960, 1280), min_size=None, max_size=None, ext='.png',
                                          force_reencode=True, delete_originals=True, num_workers=None, skip_failures=False)


if __name__ == '__main__':
    src_dataset_local_path = "test/my_kitti_dataset_dir"
    dataset_name = "first_time1"
    dataset = load_dataset_from_local_path(
        src_dataset_local_path, dataset_name, True)
    # View summary info about the dataset
    print(dataset)

    # Print the first few samples in the dataset
    # print(dataset.head())
    reencode_to_png_in_place(dataset)
