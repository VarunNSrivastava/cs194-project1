# CS194-26 (CS294-26): Project 1 starter Python code

# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images
import math
from time import time
import numpy as np
import skimage as sk
import skimage.io as skio
from skimage.transform import rescale, rotate
import matplotlib
import os
from random import randint

matplotlib.use('TkAgg')
skio.use_plugin('matplotlib')


def NCC(v1, v2):
    # returns NCC of 1d arrays v1 and v2.
    nv1 = v1 / np.linalg.norm(v1)
    nv2 = v2 / np.linalg.norm(v2)
    return np.dot(nv1, nv2)


def sample(guess, res, size):
    # returns all points in a +/-size window around guess
    # where size is upscaled based on resolution
    return [
        (int(h / res + guess[0]), int(w / res + guess[1]))
        for h in range(-size, size)
        for w in range(-size, size)
    ]


class Image:
    def __init__(self, file):
        time_0 = time()

        self.image = skio.imread(file)
        print(f"imread takes {time() - time_0} seconds")

        self.image_height, self.image_width = self.image.shape

        # Initial guesses for centers of each image
        self.blue_center = int(self.image_height / 6), int(self.image_width / 2)
        self.green_center = int(3 * self.image_height / 6), int(self.image_width / 2)
        self.red_center = int(5 * self.image_height / 6), int(self.image_width / 2)

        # Initial (conservative) estimates for the size of each image
        self.top_height = int(self.blue_center[0] * 0.75)
        self.bottom_height = int((self.image_height - self.red_center[0]) * 0.75)
        self.left_width = int(0.8 * (self.image_width / 2))
        self.right_width = int(0.8 * (self.image_width / 2))


    def get_image(self, center, res=1):

        # print(center[0] - self.top_height)
        # print(center[0] + self.bottom_height)
        # print(center[1] - self.left_width)
        # print(center[1] + self.right_width)

        img = self.image[center[0] - self.top_height: center[0] + self.bottom_height,
              center[1] - self.left_width: center[1] + self.right_width]
        return rescale(img, res, anti_aliasing=False)

    def get_score(self, center1, center2, res=1):
        # print(f"Center 1 is {center1}")
        # print(f"Center 2 is {center2}")

        img1 = self.get_image(center1, res=res)
        img2 = self.get_image(center2, res=res)

        return NCC(img1.flatten(), img2.flatten())

    def calc_center(self, guess, res):
        if res >= 2:
            print(f"Best guess is {guess}.")
            score = self.get_score(self.blue_center, guess, 1)
            print(f"With a score of {score}.")
            return guess
        if res > 1:
            res = 1
        # gets lattice of points near guess
        points = sample(guess, res, 2)
        for point in points:
            print(f"{point[0]} \t {point[1]}")

        # get best point at current resolution
        best_center = max(points, key=lambda c: self.get_score(self.blue_center, c, res))
        print(f"Best center at res {res} is {best_center}")

        # attempt best guess at higher resolution
        return self.calc_center(best_center, res * 3)

    def calc_dimensions(self):
        # cropping images to be the same size
        centers = [self.blue_center, self.green_center, self.red_center]

        self.top_height = self.blue_center[0]
        self.bottom_height = self.image_height - self.red_center[0]

        self.left_width = min([center[1] for center in centers])
        self.right_width = min([self.image_width - center[1] for center in centers])

    def align(self):
        # somewhat arbitrary starting point for coarsest resolution
        res = 1 / (0.014 * self.image_width + 10)

        self.green_center = self.calc_center(self.green_center, res)
        self.red_center = self.calc_center(self.red_center, res)
        self.calc_dimensions()

    def out(self):
        r = sk.util.img_as_ubyte(self.get_image(self.red_center))
        g = sk.util.img_as_ubyte(self.get_image(self.green_center))
        b = sk.util.img_as_ubyte(self.get_image(self.blue_center))
        return np.dstack([r, g, b])


def main():
    for f in os.listdir("lib"):
        time_0 = time()
        img = Image(os.path.join("lib", f))
        img.align()
        align_time = time() - time_0

        img_name = os.path.splitext(f)[0]

        skio.imsave(os.path.join("out", f"{img_name}.jpeg"), img.out())

        with open(os.path.join("out", f"{img_name}.txt"), "w") as o:
            o.write(f"Aligning time was {align_time} seconds.\n")
            o.write(f"Red center is {img.red_center}. \n Green center is {img.green_center}. \n Blue center is {img.blue_center}.\n")
            o.write(f"NCC scores: \n blue and red: {img.get_score(img.blue_center, img.red_center)}\n")
            o.write(f"red and and green: {img.get_score(img.red_center, img.green_center)}\n")
            o.write(f"green and and blue: {img.get_score(img.green_center, img.blue_center)}\n")


if __name__ == '__main__':
    # img = Image("lib/emir.tiff")
    # img.align()
    # skio.imshow(img.out())
    # skio.show()
    main()
