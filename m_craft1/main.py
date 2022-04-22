# Scene text detection
# CRAFT (craft_text_detector)
from craft_text_detector import Craft

# set image path and export folder directory
image = '/home/anna/workspace/DATA_UTIA/FewImages/imgs/1001.jpg'
output_dir = 'output/'

# create a craft instance
craft = Craft(output_dir=output_dir, crop_type="poly", cuda=False)

# apply craft text detection and export detected regions to output directory
prediction_result = craft.detect_text(image)

# unload models from ram/gpu
craft.unload_craftnet_model()
craft.unload_refinenet_model()
