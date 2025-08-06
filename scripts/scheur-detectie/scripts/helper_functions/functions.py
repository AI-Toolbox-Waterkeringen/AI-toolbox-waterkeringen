import shutil
import os
import cv2
from glob import glob
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

# # Helper functions # #


def copy_images(x_kfold, y_kfold, image_folder, mask_folder):
    """[summary]

    Args:
        x_kfold ([type]): [description]
        y_kfold ([type]): [description]
        image_folder ([type]): [description]
        mask_folder ([type]): [description]
    """
    for img, mask in zip(x_kfold, y_kfold):
        target_img = image_folder / Path(img).name
        target_mask = mask_folder / Path(mask).name
        shutil.copy(img, target_img)
        shutil.copy(mask, target_mask)


def copy_file(source_file, source_folder, destination_folder):
    """[summary]

    Args:
        source_file ([type]): [description]
        source_folder ([type]): [description]
        destination_folder ([type]): [description]
    """
    shutil.copy(source_folder / source_file.name, destination_folder / source_file.name)


def folder_buildr(folder, make_dir=False, img_dir="img", masks_dir="masks"):
    """[summary]

    Args:
        folder ([type]): [description]

    Returns:
        [type]: [description]
    """
    image_folder = folder / img_dir
    masks_folder = folder / masks_dir
    if make_dir:
        folder_makedirs(image_folder, masks_folder)
    return image_folder, masks_folder


def folder_makedirs(image_folder, masks_folder):
    """[summary]

    Args:
        image_folder ([type]): [description]
        masks_folder ([type]): [description]
    """
    for folder in [image_folder, masks_folder]:
        os.makedirs(folder, exist_ok=True)


def kfold_folder_buildr(folder, sub_folder, fold, split_set, make_dir=False):
    """[summary]

    Args:
        folder ([type]): [description]
        fold ([type]): [description]
        split_set ([type]): [description]

    Returns:
        [type]: [description]
    """
    image = folder / "fold_{}/{}/{}/img".format(str(fold), sub_folder, split_set)
    masks = folder / "fold_{}/{}/{}/masks".format(str(fold), sub_folder, split_set)
    if make_dir:
        folder_makedirs(image, masks)
    return image, masks


def copy_file(source_file, source_folder, destination_folder):
    """[summary]

    Args:
        source_file ([type]): [description]
        source_folder ([type]): [description]
        destination_folder ([type]): [description]
    """
    shutil.copy(source_folder / source_file.name, destination_folder / source_file.name)


def write_image_to_subimages(
    img, imname, target_folder, target_width, target_height, multiply=1, dpi=None
):
    """
    Functie om van een image meerdere subimages te maken en deze weg te schrijven naar een folder
    Input:
    img: image array (nrow,ncol,:)
    imname: Naam van de image (zonder extentie)
    target_folder: Waar de images geschreven moeten worden
    target_width: Breedte van de subimages
    target_height: Hoogte van de subimages

    Returns:
    None
    """

    ar_width, ar_height = img.shape[1], img.shape[0]
    N_width = round(ar_width / target_width)
    N_height = round(ar_height / target_height)

    if N_width * target_width == img.shape[1]:
        N_width -= 1
    if N_height * target_height == img.shape[0]:
        N_height -= 1

    for w in range(N_width + 1):
        col_start = min(w * target_width, ar_width - target_width)
        col_end = min((w + 1) * target_width, ar_width)
        for h in range(N_height + 1):
            row_start = min(h * target_height, ar_height - target_height)
            row_end = min((h + 1) * target_height, ar_height)
            subset = img[row_start:row_end, col_start:col_end, :] * multiply

            name_out = "{}_rowstart_{}_rowend_{}_colstart_{}_colend_{}.png".format(
                imname, row_start, row_end, col_start, col_end
            )
            cv2.imwrite(str(target_folder / name_out), subset)


def idx_compr(data_list, idx_list):
    """[summary]

    Args:
        data_list ([type]): [description]
        idx_list ([type]): [description]

    Returns:
        [type]: [description]
    """
    return [data_list[i] for i in idx_list]


def load_data_cross(path, k, split=0.2):
    """
        Function to split trainingsdata in k batches

    Args:
        path ([pathlib.PosixPath,str]): path with sub directories img and masks
        k ([int])					  : Number of folds
                split ([int])				  : fraction of data that is used for test/validation data
                                                                                if split = 0.2: Size trainingsdata = 0.6, size val = 0.2, size test = 0.2
    Returns:
         Xta [list]					  : list of k lists with trainings images per fold
                 yta [list]					  : list of k lists with trainings masks per fold
                 Xva [list]					  : list of k lists with validation images per fold
                 yva [list]					  : list of k lists with validation masks per fold
                 Xte [list]					  : list of k lists with test images per fold
                 yte [list]					  : list of k lists with test masks per fold

    """
    datasets = [Xta, yta, Xva, yva, Xte, yte] = [[], [], [], [], [], []]

    images = sorted(glob(os.path.join(path, "img/*.png")))
    masks = sorted(glob(os.path.join(path, "masks/*.png")))

    print("#img = {} , #mask= {}".format(len(images), len(masks)))

    X, y = images, masks
    valid_size = int(split * len(X))
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    for train_idx, test_idx in kf.split(X):
        X_train, y_train = idx_compr(X, train_idx), idx_compr(y, train_idx)
        temp_x, temp_y = idx_compr(X, test_idx), idx_compr(y, test_idx)

        X_test, X_val = train_test_split(temp_x, test_size=0.5, random_state=42)
        y_test, y_val = train_test_split(temp_y, test_size=0.5, random_state=42)

        temp_data = [X_train, y_train, X_val, y_val, X_test, y_test]
        for idx, set in enumerate(datasets):
            set.append(temp_data[idx])

    return (Xta, yta), (Xva, yva), (Xte, yte)
