
import matplotlib.pyplot as plt
from mtcnn import MTCNN, mtcnn
import cv2

# load image from file
filename = "D:\\tetse.jpg"
pixels = plt.imread(filename)
print("Shape of image/array:",pixels.shape)
imgplot = plt.imshow(pixels)


def draw_facebox(filename, result_list):
    data = plt.imread(filename)
    plt.imshow(data)
    ax = plt.gca()
    for result in result_list:
        x, y, width, height = result['box']
        rect = plt.Rectangle((x, y), width, height,fill=False, color='orange')
        ax.add_patch(rect)
        for key, value in result['keypoints'].items():
            dot = plt.Circle(value, radius=5, color='red')
            ax.add_patch(dot)

    plt.show()


detector = mtcnn.MTCNN()
faces = detector.detect_faces(pixels)
draw_facebox(filename, faces)
