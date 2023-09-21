import openslide
from matplotlib import pyplot as plt
path = "/mnt/d/data/shuangran/1/55.ndpi"
slide = openslide.open_slide(path)


ppm_x = 1 / float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
ppm_y = 1 / float(slide.properties[openslide.PROPERTY_NAME_MPP_Y])

slide_width, slide_height = slide.level_dimensions[0]
img_center_x = (slide_width / 2) * 1000 / ppm_x
img_center_y = (slide_height / 2) * 1000 / ppm_y

# Get the properties from the OpenSlide object
properties = slide.properties

# Get the offsets from the image center to the slide center
offset_from_image_center_x = float(properties['hamamatsu.XOffsetFromSlideCentre'])
offset_from_image_center_y = float(properties['hamamatsu.YOffsetFromSlideCentre'])

# Compute the offsets from the top left to the image center
offset_from_top_left_x = img_center_x - offset_from_image_center_x
offset_from_top_left_y = img_center_y - offset_from_image_center_y

# The offsets from the top left to the slide center are the references for the X and Y coordinates
x_reference = offset_from_top_left_x
y_reference = offset_from_top_left_y

x = offset_from_image_center_x / 1000 * ppm_x
y = offset_from_image_center_y / 1000 * ppm_y
png_0 = slide.read_region((int(x), int(y)), 0, (slide_width // 2,slide_height // 2))
plt.imshow(png_0)
plt.show()
print("")
