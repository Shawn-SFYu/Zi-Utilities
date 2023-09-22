from PIL import Image
import os 
import numpy as np
import argparse

def main(args):
    all_files = [os.path.join(args.image_dir, fname) \
                 for fname in os.listdir(args.image_dir)]
    all_files.sort(key=lambda x: os.path.getmtime(x), reverse=False)
    
    images = []

    for file in all_files:
        img = Image.open(file)
        img_data = np.asarray(img)

        min = np.percentile(img_data, args.p_min)
        max = np.percentile(img_data, args.p_max)

        normalized_data = np.clip( 255.0 * (img_data - min)/(max - min), 0, 255)

        normalized_img = Image.fromarray(np.uint8(normalized_data))

        images.append(normalized_img)
    
    images[0].save(args.output_path, save_all=True, append_images=images[1:], loop=0, duration=300)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate gif from image series")
    parser.add_argument("--image_dir", help="input image directory")
    parser.add_argument("--output_path", type=str, help="output gif directory")
    parser.add_argument("--p_min", default=5, type=int, \
                        help="exclude p_min percent of low intensity")
    parser.add_argument("--p_max", default=95, type=int, \
                        help="exclude p_max percent of high intensity")
    args = parser.parse_args()
    main(args)
