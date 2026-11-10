from PIL import Image, ImageDraw
import os
import random

classes = ["10","20","50","100","200","500","2000"]

for cls in classes:

    folder = f"dataset/{cls}"
    os.makedirs(folder, exist_ok=True)

    for i in range(50):

        img = Image.new(
            "RGB",
            (224,224),
            (
                random.randint(0,255),
                random.randint(0,255),
                random.randint(0,255)
            )
        )

        draw = ImageDraw.Draw(img)

        draw.text(
            (80,100),
            f"Rs {cls}",
            fill=(255,255,255)
        )

        img.save(
            f"{folder}/{i}.jpg"
        )

print("Dataset Generated")
