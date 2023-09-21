"""
Generate image datafolder from TTF files. 
Input:
    --ttf_path  path to ttf files
    --output_dir    "output directory
Output:
    folders and images saved at output_dir 
"""

import os
import logging
import glob
import argparse
from PIL import Image, ImageFont
from gen_base_img import char2img

def ttf2datafolder(char_list, data_path, fonts):
    
    for i, character in enumerate(char_list):
        logging.debug(f"Processing {i} {character} \n")
        char_path = os.path.join(data_path, character)
        if not os.path.exists(char_path):
            os.mkdir(char_path)
        else:
            print(f'Directory exists {i} {character}')
        for j, font in enumerate(fonts):
            img = char2img(character, font=ImageFont.truetype(font, size=200))
            name = "Char-" + str(i) + '-Font-' + str(j) + '.png'
            img.save(os.path.join(char_path, name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate image datafolder from TTF files.")
    parser.add_argument('--ttf_path', default="./Fonts", type=str, help="path to ttf files")
    parser.add_argument('--output_dir', default='./DicData', type=str, help="output directory")
    args = parser.parse_args()

    fonts = glob.glob(args.ttf_path+"/*.ttf")
    with open('characters.txt', 'r') as file:
        # Read the contents of the file into a single string
        char_list = file.read()
    logging.basicConfig(filename='ttf2datafolder.log', level=logging.DEBUG)
    ttf2datafolder(char_list, data_path=args.output_dir, fonts=fonts)



