import easyocr
import os

# if GPU is to be used, allow CUDA
use_GPU = False  # True

reader = easyocr.Reader(['en'], gpu=use_GPU)    # this needs to run only once to load the model into memory
# 80+ supported languages german = 'de'

# path to image directory, get full path to all files
imgs_dir = '/home/anna/workspace/DATA_UTIA/FewImages/imgs'
(_, _, filenames) = next(os.walk(imgs_dir))

# set and create output directory if it doesn't exist
output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# loop through all images in image directory
for img_name in filenames:
    # read image (full path to image)
    img_path = os.path.join(imgs_dir, img_name)
    output = reader.readtext(img_path)

    # create file for each image with the results (name format: result_imgname.txt)
    with open('%s/result_%s.txt' % (output_dir, img_name), 'w') as output_file:
        output_file.write(' '.join(str(result) for result in output) + '\n')



