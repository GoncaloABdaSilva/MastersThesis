from PIL import Image
import os

def convert_bmp_to_png(input_folder, output_folder):
    for filename in os.listdir(input_folder):
        input_file = os.path.join(input_folder, filename)
            
        # Open the BMP image
        with Image.open(input_file) as img:
            # Get the base name without extension and create the output file path
            base_name, ext = os.path.splitext(filename)
            output_file = os.path.join(output_folder, f"{base_name}.png")
            
            # Save as PNG with optimization
            img.save(output_file, 'PNG', optimize=True)

    print("Finished converting .bmp masks to .png")

script_folder = os.path.dirname(os.path.abspath(__file__))
gt_folder = os.path.join(script_folder, "gt")
new_gt_folder = os.path.join(script_folder, "new_gt")

convert_bmp_to_png(gt_folder, new_gt_folder)
