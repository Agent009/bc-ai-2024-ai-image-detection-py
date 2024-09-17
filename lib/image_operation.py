import os
import sys
from transformers import pipeline
from PIL import Image, ImageDraw

from lib.utils import get_data_path

data_path = get_data_path()


def image_operation(image=None, op="classify", checkpoint="google/owlv2-base-patch16-ensemble", task="zero-shot-object-detection"):
    """
    Performs image operations such as classification or object detection using a pre-trained model.

    Parameters:
    image (str, optional): The name of the image file to load. If not provided, the image file name is obtained from the command-line arguments.
    op (str, optional): The operation to perform on the image. It can be either "classify" or "detect". Defaults to "classify".
    checkpoint (str, optional): The name of the pre-trained model checkpoint to use. Defaults to "google/owlv2-base-patch16-ensemble".
    task (str, optional): The task to perform on the image. It can be either "zero-shot-object-detection" or "image-classification". Defaults to "zero-shot-object-detection".

    Returns:
    None

    Raises:
    ValueError: If the provided operation is not one of the allowed values ("classify", "detect").
    """
    if op not in ["classify", "detect"]:
        raise ValueError("Operation should only allow the following values: classify, detect")

    # Import the model and define the pipeline
    detector = pipeline(model=checkpoint, task=task)
    image_data = get_image(image)
    image = image_data[0]

    # Receive the list of objects to detect as user input
    # The list of words must be separated by single spaces
    labels = input("Enter the labels you want to detect: ").split(" ")
    # Run the pipeline and get the results
    predictions = detector(
        image,
        candidate_labels=labels,
    )

    if op == "detect":
        # Draw the bounding boxes on the image
        draw = ImageDraw.Draw(image)

        # Draw each bounding box and label on the image, along with the score
        for prediction in predictions:
            box = prediction["box"]
            label = prediction["label"]
            score = prediction["score"]

            xmin, ymin, xmax, ymax = box.values()
            draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=1)
            draw.text((xmin, ymin), f"{label}: {round(score, 2)}", fill="black")
            print(f"Detected {label} with confidence {round(score, 2)}")

        # Save the image with the bounding boxes
        save_path = os.path.join(data_path, f"{image_data[1].split('.')[0]}_detection.png")
        image.save(save_path)
    else:
        # Print the probabilities for each word
        i = 1

        for prediction in predictions:
            label = prediction["label"]
            score = prediction["score"] * 100
            suffix = ''

            if 11 <= (i % 100) <= 13:
                suffix = 'th'
            else:
                suffix = ['th', 'st', 'nd', 'rd', 'th'][min(i % 10, 4)]
                
            print(f"The word {label} is the {i}{suffix} most related to the image with a confidence of {score:.2f}%")
            i += 1

def get_image(image) -> (Image, str):
    """
    This function loads an image for further processing.

    Parameters:
    image (str, optional): The name of the image file to load. If not provided, the image file name is obtained from the command-line arguments.

    Returns:
    tuple: A tuple containing the loaded image (PIL.Image object) and the image file name (str).

    Raises:
    SystemExit: If no image file is provided and no command-line arguments are given.
    """
    if len(sys.argv) > 1:
        # Load the image passed as argument to the script
        filename = sys.argv[1]
    else:
        # Otherwise, use the image provided in the parameter, if any.
        filename = image

    if filename is None or not filename.endswith(('.jpg', '.png', '.jpeg')):
        print("No image file provided. Please provide an image file as an argument.")
        sys.exit(1)

    return Image.open(os.path.join(data_path, filename)), filename