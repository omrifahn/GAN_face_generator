import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN

def add_slide(prs, layout, title, content):
    slide = prs.slides.add_slide(layout)
    title_shape = slide.shapes.title
    title_shape.text = title
    body_shape = slide.shapes.placeholders[1]
    tf = body_shape.text_frame
    tf.text = content
    return slide

def add_bullet_slide(prs, layout, title, bullets):
    slide = prs.slides.add_slide(layout)
    title_shape = slide.shapes.title
    title_shape.text = title
    body_shape = slide.shapes.placeholders[1]
    tf = body_shape.text_frame
    for bullet in bullets:
        p = tf.add_paragraph()
        p.text = bullet
        p.level = 0
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

    # Slide 2: Introduction
    add_bullet_slide(prs, bullet_slide_layout, "Introduction", [
        "Neural Networks in AI",
        "CNNs: A Game-Changer in Image Processing"
    ])

    # Slide 3: What is a Convolution?
    add_bullet_slide(prs, bullet_slide_layout, "What is a Convolution?", [
        "Mathematical operation",
        "Sliding window over data",
        "Feature extraction"
    ])

    # Slide 4: Key Components of CNNs
    add_bullet_slide(prs, bullet_slide_layout, "Key Components of CNNs", [
        "1. Convolutional Layers",
        "2. Pooling Layers",
        "3. Fully Connected Layers"
    ])

    # Slide 5: How CNNs Process Images
    add_slide(prs, bullet_slide_layout, "How CNNs Process Images",
              "[Diagram showing the flow of an image through CNN layers]")

    # Slide 6: Advantages of CNNs
    add_bullet_slide(prs, bullet_slide_layout, "Advantages of CNNs", [
        "Parameter sharing",
        "Spatial invariance",
        "Hierarchical feature learning"
    ])

    # Slide 7: Simple CNN Architecture
    add_slide(prs, bullet_slide_layout, "Simple CNN Architecture",
              "[Diagram of a basic CNN architecture]")

    # Slide 8: Real-World Applications
    add_bullet_slide(prs, bullet_slide_layout, "Real-World Applications", [
        "Image Classification",
        "Object Detection",
        "Facial Recognition",
        "Medical Image Analysis"
    ])

    # Slide 9: Conclusion and Future Directions
    add_bullet_slide(prs, bullet_slide_layout, "Conclusion and Future Directions", [
        "Recap: CNNs revolutionize image processing",
        "Ongoing research: Improved architectures and efficiency"
    ])

    # Slide 10: Thank You
    add_slide(prs, bullet_slide_layout, "Thank You", "Questions?")

    # Save the presentation
    prs.save('CNN_Seminar_Presentation.pptx')

if __name__ == "__main__":
    create_cnn_presentation()
    print("Presentation created successfully as 'CNN_Seminar_Presentation.pptx'")