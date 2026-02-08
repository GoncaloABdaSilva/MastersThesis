import os

def convert(input_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith('.JPG'):
            input_file_path = os.path.join(input_folder, filename)
            new_filename = filename[:-4] + ".jpg"
            new_path = os.path.join(input_folder, new_filename)
            os.rename(input_file_path, new_path)   

    print("Finished converting .JPG masks to .jpg")

script_folder = os.path.dirname(os.path.abspath(__file__))
convert(script_folder)
