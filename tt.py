from vision_agent.tools import load_image, owlv2_object_detection, florence2_sam2_instance_segmentation, countgd_object_detection
image = load_image("5423_group-of-people-sitting-in-a-cafe.jpg")
owl_v2_out = owlv2_object_detection("table", image)

# f2s2_out = florence2_sam2_instance_segmentation("person", image)
# strip out the masks from the output becuase they don't provide useful information when printed
# f2s2_out = [{{k: v for k, v in o.items() if k != "mask"}} for o in f2s2_out]

cgd_out = countgd_object_detection("table", image)

final_out = {  # Change this line
    "owlv2_object_detection": owl_v2_out,
    # "florence2_sam2_instance_segmentation": f2s2_out,
    "countgd_object_detection": cgd_out
}
print(final_out)