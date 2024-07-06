import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN


def add_slide_with_notes(prs, layout, title, content, notes):
    slide = prs.slides.add_slide(layout)
    title_shape = slide.shapes.title
    title_shape.text = title
    body_shape = slide.shapes.placeholders[1]
    tf = body_shape.text_frame
    tf.text = content

    # Add speaker notes
    slide.notes_slide.notes_text_frame.text = notes

    return slide


def add_bullet_slide_with_notes(prs, layout, title, bullets, notes):
    slide = prs.slides.add_slide(layout)
    title_shape = slide.shapes.title
    title_shape.text = title
    body_shape = slide.shapes.placeholders[1]
    tf = body_shape.text_frame
    for bullet in bullets:
        p = tf.add_paragraph()
        p.text = bullet
        p.level = 0

    # Add speaker notes
    slide.notes_slide.notes_text_frame.text = notes

    return slide


def create_cnn_presentation():
    prs = Presentation()
    title_slide_layout = prs.slide_layouts[0]
    bullet_slide_layout = prs.slide_layouts[1]

    # Slide 1: Title
    slide = prs.slides.add_slide(title_slide_layout)
    title = slide.shapes.title
    subtitle = slide.placeholders[1]
    title.text = "Convolutional Neural Networks (CNNs)"
    subtitle.text = "A 20-Minute Introduction"
    slide.notes_slide.notes_text_frame.text = "Welcome everyone to this seminar on Convolutional Neural Networks, or CNNs. We'll explore how CNNs have revolutionized image processing in AI. By the end, you'll understand the basics of CNNs and their importance in modern AI applications."

    # Slide 2: Introduction
    add_bullet_slide_with_notes(prs, bullet_slide_layout, "Introduction", [
        "Neural Networks in AI",
        "CNNs: A Game-Changer in Image Processing"
    ],
                                "Neural networks are a fundamental part of modern AI. CNNs, in particular, have revolutionized image processing tasks, making significant breakthroughs in computer vision applications.")

    # Slide 3: What is a Convolution?
    add_bullet_slide_with_notes(prs, bullet_slide_layout, "What is a Convolution?", [
        "Mathematical operation",
        "Sliding window over data",
        "Feature extraction"
    ],
                                "At the heart of CNNs is the convolution operation. Imagine a small window sliding over an image, performing calculations at each step. This process helps in extracting important features from the input data. In image processing, it can detect edges, textures, and other patterns.")

    # Slide 4: Key Components of CNNs
    add_bullet_slide_with_notes(prs, bullet_slide_layout, "Key Components of CNNs", [
        "1. Convolutional Layers",
        "2. Pooling Layers",
        "3. Fully Connected Layers"
    ],
                                "CNNs consist of three main types of layers: 1) Convolutional layers apply filters to detect features. 2) Pooling layers reduce the spatial dimensions of the data. 3) Fully connected layers make the final decision based on the extracted features. This unique architecture allows CNNs to efficiently process image data.")

    # Slide 5: How CNNs Process Images
    add_slide_with_notes(prs, bullet_slide_layout, "How CNNs Process Images",
                         "[Diagram showing the flow of an image through CNN layers]",
                         "Let's walk through how a CNN processes an image: 1) The image enters the network. 2) Convolutional layers detect features like edges and shapes. 3) Pooling layers summarize these features. 4) This process repeats, detecting increasingly complex features. 5) Finally, fully connected layers use these features for classification. This hierarchical learning is what makes CNNs so powerful for image analysis.")

    # Slide 6: Advantages of CNNs
    add_bullet_slide_with_notes(prs, bullet_slide_layout, "Advantages of CNNs", [
        "Parameter sharing",
        "Spatial invariance",
        "Hierarchical feature learning"
    ],
                                "CNNs have several advantages over traditional neural networks: 1) Parameter sharing: The same filter is applied across the entire image, reducing the number of parameters. 2) Spatial invariance: CNNs can detect features regardless of their position in the image. 3) Hierarchical feature learning: Each layer builds upon the previous, learning more complex features. These properties make CNNs particularly suited for image-related tasks.")

    # Slide 7: Simple CNN Architecture
    add_slide_with_notes(prs, bullet_slide_layout, "Simple CNN Architecture",
                         "[Diagram of a basic CNN architecture]",
                         "Let's break down a simple CNN architecture: 1) Input layer: Receives the raw image data. 2) Convolutional layers: Apply filters to detect features. 3) Activation functions (like ReLU): Introduce non-linearity. 4) Pooling layers: Reduce spatial dimensions and computational load. 5) Fully connected layers: Interpret the extracted features. 6) Output layer: Produces the final prediction or classification. This structure allows the network to progressively learn and make decisions based on image content.")

    # Slide 8: Real-World Applications
    add_bullet_slide_with_notes(prs, bullet_slide_layout, "Real-World Applications", [
        "Image Classification",
        "Object Detection",
        "Facial Recognition",
        "Medical Image Analysis"
    ],
                                "CNNs have numerous real-world applications: 1) Image Classification: Identifying the content of images. 2) Object Detection: Locating and identifying multiple objects in an image. 3) Facial Recognition: Identifying or verifying a person from their face. 4) Medical Image Analysis: Detecting abnormalities in X-rays, MRIs, etc. These applications demonstrate the versatility and power of CNNs in solving complex visual tasks.")

    # Slide 9: Conclusion and Future Directions
    add_bullet_slide_with_notes(prs, bullet_slide_layout, "Conclusion and Future Directions", [
        "Recap: CNNs revolutionize image processing",
        "Ongoing research: Improved architectures and efficiency"
    ],
                                "To recap, CNNs have revolutionized image processing in AI by: 1) Efficiently extracting hierarchical features from images. 2) Providing spatial invariance and parameter sharing. 3) Enabling a wide range of applications in computer vision. Research continues to improve CNN architectures, making them more efficient and capable. The future of CNNs is exciting, with potential applications in video analysis, 3D image processing, and more.")

    # Slide 10: Thank You
    add_slide_with_notes(prs, bullet_slide_layout, "Thank You", "Questions?",
                         "Thank you for your attention. I hope this introduction to CNNs has been informative. I'm happy to take any questions you might have about CNNs or their applications.")

    # Save the presentation
    prs.save('CNN_Seminar_Presentation.pptx')


if __name__ == "__main__":
    create_cnn_presentation()
    print("Presentation created successfully as 'CNN_Seminar_Presentation.pptx'")