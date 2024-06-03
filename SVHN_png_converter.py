from glob import glob
from PIL import Image
import scipy.io
import numpy as np

if __name__ == "__main__":
    # mat_files = glob("./Datasets/SVHN/mat/*.mat")
    counter = 0
    # for mat_file in mat_files:
    mat_file = "./Datasets/SVHN/mat/train_32x32.mat"
    mat = scipy.io.loadmat(mat_file)
    images = mat["X"]

    for i in range(images.shape[3]):
        image = images[:, :, :, i]
        img = Image.fromarray(np.uint8(image))
        img.save("./Datasets/SVHN/png_train/image_{}.png".format(counter))
        counter += 1
