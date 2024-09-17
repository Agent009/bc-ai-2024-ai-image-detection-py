from lib.image_operation import image_operation


def detect(image=None, checkpoint="google/owlv2-base-patch16-ensemble", task="zero-shot-object-detection"):
    image_operation(op="detect", image=image, checkpoint=checkpoint, task=task)
