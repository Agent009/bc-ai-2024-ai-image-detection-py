from lib.image_operation import image_operation


def classify(image=None, checkpoint="openai/clip-vit-large-patch14", task="zero-shot-image-classification"):
    image_operation(op="classify", image=image, checkpoint=checkpoint, task=task)
