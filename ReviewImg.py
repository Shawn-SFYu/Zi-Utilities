import pickle
import cv2

from PrepareData import DataPair


ReviewPath = "Data/img_dict_0赵孟頫.pickle.P1"  # .P1-label
with open(ReviewPath, "rb") as f:
    img_dict = pickle.load(f)
# with open('characters.txt', 'r') as file:
#    contents = file.read()

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
with open(ReviewPath + "-label", "wb") as f:
    pickle.dump(img_dict, f)
