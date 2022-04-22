# Scene text detection
# CRAFT (craft_text_detector)

import os
from os import walk

from craft_text_detector import (
    read_image,
    load_craftnet_model,
    load_refinenet_model,
    get_prediction,
    export_detected_regions,
    export_extra_results,
    empty_cuda_cache
)

# if GPU is to be used, allow CUDA
use_GPU = False  # True

# path to image directory, get full path to all files
imgs_dir = '/home/anna/workspace/DATA_UTIA/FewImages/imgs'
(_, _, filenames) = next(walk(imgs_dir))
# imgs = [os.path.join(dir_path, f) for f in filenames]

# set and create output directory if it doesn't exist
output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# loop through all images in image directory
for img in filenames:
    # create directory for each image
    output_dir_img = os.path.join(output_dir, img)
    if not os.path.exists(output_dir_img):
        os.makedirs(output_dir_img)

    # read image (full path to image)
    image = read_image(os.path.join(imgs_dir, img))

    # load models
    refine_net = load_refinenet_model(cuda=use_GPU)
    craft_net = load_craftnet_model(cuda=use_GPU)

    # perform prediction
    prediction_result = get_prediction(
        image=image,
        craft_net=craft_net,
        refine_net=refine_net,
        text_threshold=0.7,
        link_threshold=0.4,
        low_text=0.4,
        cuda=use_GPU,
        long_size=1280
    )

    # export detected text regions
    exported_file_paths = export_detected_regions(
        image=image,
        regions=prediction_result["boxes"],
        output_dir=output_dir_img,
        rectify=True
    )

    # export heatmap, detection points, box visualization
    export_extra_results(
        image=image,
        regions=prediction_result["boxes"],
        heatmaps=prediction_result["heatmaps"],
        output_dir=output_dir_img
    )

    # unload models from gpu
    if use_GPU:
        empty_cuda_cache()
