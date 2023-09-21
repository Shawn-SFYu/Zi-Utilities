"""
Remove island-noise from pickle images, extra Stage 1 in denoising
    This type of noise has been observed to be prominent after initial image denoising
    Extra stages can be added later according to image quality and actual needs 
    Input:
        input_file: pickle file to be denoised
        debug_mode: whether to visualize images for debug
    Output:
        output_file: named as {input_file}.P1
"""
import pickle
import cv2
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Stage 1, island-noise remover")
    parser.add_argument('--input', default="img_dict_0赵孟頫.pickle", type=str, help="input pickle file")
    parser.add_argument('--island_ratio', default=0.0005, type=float, \
                        help="removed if (islands area/image size) < island_ratio")
    parser.add_argument('--ouput_suffix', default="P1", type=str, help="output_name = input.output_suffix")
    parser.add_argument('--debug_mode', default=False, type=bool, help="image visualized in debug mode")
    args = parser.parse_args()

    with open(args.input, "rb") as f:
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
                if area < args.island_ratio * total_area:
                    # this size is empirical
                    cv2.drawContours(color_img, contour, -1, color=(0, 0, 0), thickness=5)
                    cv2.fillPoly(color_img, [contour], (0, 0, 0))

            data_pair.output = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
            # cv2.fillPoly(data_pair.output, [contour], 0)
            if args.debug_mode:
                result = cv2.hconcat([original, thresh, data_pair.output])
                cv2.imshow('Result', result)
                label = chr(cv2.waitKey(0))
                cv2.destroyAllWindows()
                
    with open(args.input + "." + args.ouput_suffix, "wb") as f:
        pickle.dump(img_dict, f)