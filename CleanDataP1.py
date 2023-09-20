"""
Remove small islands from pickle images
"""
import pickle
import cv2
import numpy as np
from PrepareData import DataPair
DEBUG = False
File = "img_dict_0赵孟頫.pickle"
if __name__ == "__main__":
    with open("Data/" + File, "rb") as f:
        img_dict = pickle.load(f)
    with open('characters.txt', 'r') as file:
        contents = file.read()
    for key in img_dict:
        for data_pair in img_dict[key]:
            original = data_pair.output
            data_pair.output = cv2.normalize(original, None, 0, 255, cv2.NORM_MINMAX)
            _, thresh = cv2.threshold(data_pair.output, 120, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            total_area = data_pair.output.shape[0] * data_pair.output.shape[1]
            color_img = cv2.cvtColor(data_pair.output, cv2.COLOR_GRAY2BGR)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 0.0005 * total_area:
                    # this size is empirical
                    cv2.drawContours(color_img, contour, -1, color=(0, 0, 0), thickness=5)
                    cv2.fillPoly(color_img, [contour], (0, 0, 0))
            data_pair.output = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
            # cv2.fillPoly(data_pair.output, [contour], 0)
            if DEBUG:
                result = cv2.hconcat([original, thresh, data_pair.output])
                cv2.imshow('Result', result)
                label = chr(cv2.waitKey(0))
                cv2.destroyAllWindows()
    with open("Data/" + File + ".P1", "wb") as f:
        pickle.dump(img_dict, f)