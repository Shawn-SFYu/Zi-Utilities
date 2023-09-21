"""
Manual image review and labeling to clean dataset 
    Review Options 1: Pass, 2: OCR+, 3: Noise 4: Drop
"""
import pickle
import cv2
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser("image review and labeling to clean dataset")
    parser.add_argument('--input', default="Data/img_dict_0赵孟頫.pickle.P1",type=str, \
                        help="input pickle file, ideally after stage 1 denoise")
    parser.add_argument('--output_suffix', default="label",type=str, \
                        help="output_name = input.output_suffix")
    args = parser.parse_args()
    with open(args.input, "rb") as f:
        img_dict = pickle.load(f)

    for key in img_dict:
        for data_pair in img_dict[key]:
            combined_image = cv2.hconcat([data_pair.output[::2, ::2], data_pair.input])
            title = (data_pair.label + "__Img_Key") if hasattr(data_pair, 'label') else "Img_Key"
            cv2.imshow(title + str(key)
                    + " Options 1: Pass, 2: OCR+, 3: Noise 4: Drop",
                    combined_image)
            label = chr(cv2.waitKey(0))
            while label.isdigit() is False and label != " ":
                label = cv2.waitKey(0)
                print("Labeling error: please input 0-9")
            if label.isdigit():
                data_pair.label = label
            cv2.destroyAllWindows()
    with open(args.input + "." + args.output_suffix, "wb") as f:
        pickle.dump(img_dict, f)
