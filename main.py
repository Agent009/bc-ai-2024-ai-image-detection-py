from lib.detect import detect as detect_image
from lib.classify import classify as classify_image

# ---------- Main script
if __name__ == '__main__':
    # ---------- Detect objects in an image
    # detect_image(image="CowChicken.png")

    # ---------- Classify objects in an image
    classify_image(image="CowChicken.png")
