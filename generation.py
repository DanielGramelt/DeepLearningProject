import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from skimage.filters import threshold_multiotsu
from PIL import Image
import multiprocessing

def multi_otsu(output_path, img):
    # Setting the font size for all plots.
    matplotlib.rcParams['font.size'] = 9

    # The input image.
    image = np.asarray(img)
    # Applying multi-Otsu threshold for the default value, generating
    # three classes.
    thresholds = threshold_multiotsu(image)

    # Using the threshold values, we generate the three regions.
    regions = np.digitize(image, bins=thresholds)

    plt.imshow(regions, cmap='gray')

    # Remove axis and labels
    plt.axis('off')

    # Save the image
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)


def process_file(filename, dir, output_dir):
    path = os.path.join(dir, filename)
    image = Image.open(path).convert('L')
    output_path = os.path.join(output_dir, filename)
    multi_otsu(output_path, image)

def main():
    dir = '../Dataset/validation_set/scissors'
    output_dir = '../Dataset/validation_otsu/scissors'
    filenames = [filename for filename in os.listdir(dir) if filename.endswith(".png")]

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.starmap(process_file, [(filename, dir, output_dir) for filename in filenames])

if __name__ == '__main__':
    main()
