import os
from PIL import Image, ImageDraw, ImageFont


def create_gif_from_photos(input_folder, output_file):
    # Get all image files from the input folder and sort them by name
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    image_files.sort()

    images = []
    durations = []
    base_duration = 300  # Start with 5000ms (5 seconds)
    min_duration = 10  # Minimum duration of 10ms
    max_duration = 300  # Maximum allowed duration in milliseconds

    for i, image_file in enumerate(image_files):
        # Open the image
        img = Image.open(os.path.join(input_folder, image_file)).convert("RGBA")

        # Create a drawing object
        draw = ImageDraw.Draw(img)

        # Prepare the font
        font_size = 40
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()

        # Prepare the text
        text = f"Frame: {i}"

        # Get text size
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        text_width = right - left
        text_height = bottom - top

        # Calculate position (top-right corner)
        position = (img.width - text_width - 10, 10)

        # Draw a semi-transparent background for the text
        text_bg = Image.new('RGBA', (text_width + 20, text_height + 20), (0, 0, 0, 128))
        img.paste(text_bg, (position[0] - 10, position[1] - 10), text_bg)

        # Draw the text
        draw.text(position, text, font=font, fill=(255, 255, 255, 255))

        images.append(img)

        # Calculate duration (exponential decrease)
        duration = max(min_duration, min(int(base_duration / (2 ** (i / 10))), max_duration))
        durations.append(duration)

    # Save as GIF
    images[0].save(
        output_file,
        save_all=True,
        append_images=images[1:],
        duration=durations,
        loop=0
    )

    return durations


def main():
    input_folder = "input"  # Replace with your input folder path
    output_file = "output.gif"  # Replace with your desired output file name

    durations = create_gif_from_photos(input_folder, output_file)

    print(f"GIF created successfully: {output_file}")
    print(f"Frame durations: {durations}")


if __name__ == "__main__":
    main()