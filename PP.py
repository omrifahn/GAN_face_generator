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


def create_gan_presentation():
    prs = Presentation()
    title_slide_layout = prs.slide_layouts[0]
    bullet_slide_layout = prs.slide_layouts[1]

    # Slide 1: Title
    slide = prs.slides.add_slide(title_slide_layout)
    title = slide.shapes.title
    subtitle = slide.placeholders[1]
    title.text = "Generative Adversarial Networks (GANs)"
    subtitle.text = "An Introduction to Creative AI"
    slide.notes_slide.notes_text_frame.text = "Welcome to this presentation on Generative Adversarial Networks, or GANs. We'll explore how these powerful models are pushing the boundaries of AI creativity."

    # Slide 2: Introduction
    add_bullet_slide_with_notes(prs, bullet_slide_layout, "Introduction", [
        "Brief overview of AI and machine learning",
        "Introduce the concept of generative models"
    ],
                                "We'll start with a quick overview of AI and machine learning, focusing on how they've evolved to include generative models. These are AI systems that can create new data, rather than just analyze existing data.")

    # Slide 3: What are GANs?
    add_bullet_slide_with_notes(prs, bullet_slide_layout, "What are GANs?", [
        "Basic definition and purpose",
        "The 'adversarial' concept"
    ],
                                "GANs are a type of generative model that consists of two neural networks competing against each other. The 'adversarial' in the name comes from this competition, which drives both networks to improve simultaneously.")

    # Slide 4: Architecture of GANs
    add_bullet_slide_with_notes(prs, bullet_slide_layout, "Architecture of GANs", [
        "Generator network",
        "Discriminator network",
        "How they work together"
    ],
                                "The GAN architecture consists of two main parts: the Generator and the Discriminator. The Generator creates synthetic data from random noise, while the Discriminator tries to distinguish between real and fake data. They work together in a competitive process that improves both networks.")

    # Slide 5: Generator Network
    add_bullet_slide_with_notes(prs, bullet_slide_layout, "Generator Network", [
        "Input: Random noise vector",
        "Output: Synthetic data (e.g., images)",
        "Typically uses transposed convolutional layers"
    ],
                                "The Generator takes random noise as input and produces synthetic data. In image generation tasks, it often uses transposed convolutional layers to upsample the input into a full-sized image.")

    # Slide 6: Discriminator Network
    add_bullet_slide_with_notes(prs, bullet_slide_layout, "Discriminator Network", [
        "Input: Real or generated data",
        "Output: Probability of input being real",
        "Usually uses convolutional layers"
    ],
                                "The Discriminator takes either real data or data produced by the Generator as input. It outputs a probability indicating whether it thinks the input is real or fake. For image tasks, it typically uses convolutional layers to downsample the input.")

    # Slide 7: Mathematical Framework
    add_slide_with_notes(prs, bullet_slide_layout, "Mathematical Framework",
                         "min_G max_D V(D, G) = E_x~p_data(x)[log D(x)] + E_z~p_z(z)[log(1 - D(G(z)))]",
                         "This is the core objective function of GANs. G is the Generator, D is the Discriminator, x represents real data, z is random noise, p_data is the distribution of real data, and p_z is the distribution of noise. This function encapsulates the minimax game between G and D.")

    # Slide 8: Breaking Down the Objective Function
    add_bullet_slide_with_notes(prs, bullet_slide_layout, "Breaking Down the Objective Function", [
        "G: Generator",
        "D: Discriminator",
        "x: Real data",
        "z: Random noise",
        "p_data: Distribution of real data",
        "p_z: Distribution of noise"
    ],
                                "Let's break down each component of the objective function. This will help us understand what each part represents and how they contribute to the overall training process.")

    # Slide 9: Training Process
    add_bullet_slide_with_notes(prs, bullet_slide_layout, "Training Process", [
        "1. Generate fake samples using G",
        "2. Train D on real and fake samples",
        "3. Train G to fool D"
    ],
                                "The training process involves alternating between training the Discriminator and the Generator. First, we generate fake samples, then train the Discriminator on both real and fake data. Finally, we train the Generator to produce data that can fool the Discriminator.")

    # Slide 10: Analogy - Counterfeiter and Detective
    add_slide_with_notes(prs, bullet_slide_layout, "Training Analogy",
                         "Counterfeiter (Generator) vs Detective (Discriminator)",
                         "To better understand the GAN training process, we can think of it as a game between a counterfeiter and a detective. The counterfeiter (Generator) tries to create fake currency, while the detective (Discriminator) tries to distinguish between real and fake currency. As they compete, both improve their skills.")

    # Slide 11: Applications of GANs
    add_bullet_slide_with_notes(prs, bullet_slide_layout, "Applications of GANs", [
        "Image generation",
        "Style transfer",
        "Data augmentation for machine learning"
    ],
                                "GANs have a wide range of applications. They're used for generating realistic images, transferring styles between images, and creating synthetic data to augment machine learning datasets. These are just a few examples of their capabilities.")

    # Slide 12: Conclusion and Future Prospects
    add_bullet_slide_with_notes(prs, bullet_slide_layout, "Conclusion and Future Prospects", [
        "GANs: Powerful tools for generative AI",
        "Ongoing research and improvements",
        "Potential future applications"
    ],
                                "In conclusion, GANs are powerful tools that have opened up new possibilities in generative AI. Research is ongoing to improve their stability and performance. The future looks bright, with potential applications in areas like drug discovery, virtual reality, and more.")

    # Save the presentation
    prs.save('GAN_Lecture_Presentation.pptx')


if __name__ == "__main__":
    create_gan_presentation()
    print("Presentation created successfully as 'GAN_Lecture_Presentation.pptx'")