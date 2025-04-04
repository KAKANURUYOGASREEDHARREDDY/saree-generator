import torch
import whisper
import gradio as gr
from diffusers import StableDiffusionPipeline

# Configuration
class CFG:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    generator = torch.Generator(device).manual_seed(seed)
    image_gen_steps = 50  # Higher steps for better quality
    image_gen_guidance_scale = 9
    image_gen_size = (512, 512)

# Load Models
saree_model = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2",
    torch_dtype=torch.float16,
    revision="fp16"
).to(CFG.device)

ghibli_model = StableDiffusionPipeline.from_pretrained(
    "nitrosocke/Ghibli-Diffusion",  # Ghibli-style artwork
    torch_dtype=torch.float16
).to(CFG.device)

speech_model = whisper.load_model("base")

# Function to generate images
def generate_image(prompt, model, negative_prompt=""):
    return model(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=CFG.image_gen_steps,
        generator=CFG.generator,
        guidance_scale=CFG.image_gen_guidance_scale
    ).images[0].resize(CFG.image_gen_size)

# Function to convert voice to image
def voice_to_image(audio):
    result = speech_model.transcribe(audio)
    base_prompt = result['text'].strip() + ", wearing a beautiful traditional saree"

    # Photorealistic Saree Image
    saree_prompt = f"{base_prompt}, intricate gold embroidery, silk fabric, photorealistic"
    saree_image = generate_image(
        saree_prompt, saree_model, negative_prompt="deformed face, bad anatomy, extra limbs"
    )

    # Ghibli-Style Saree Image
    ghibli_prompt = f"Ghibli style, {base_prompt}, anime artwork, studio lighting"
    ghibli_image = generate_image(
        ghibli_prompt, ghibli_model, negative_prompt="realistic, photograph, deformed"
    )

    return saree_image, ghibli_image

# Gradio Interface
interface = gr.Interface(
    fn=voice_to_image,
    inputs=gr.Audio(type="filepath"),
    outputs=[
        gr.Image(label="Traditional Saree", type="pil"),
        gr.Image(label="Ghibli Style Saree", type="pil")
    ],
    title="🎤✨ Voice-to-Saree Art Generator",
    description=(
        "Speak or upload an audio file to generate two types of saree images:<br>"
        "1️⃣ A photorealistic traditional saree.<br>"
        "2️⃣ A Studio Ghibli-style animated saree.<br><br>"
        "**Examples**: 'A woman standing by the river' or 'A dancer in a garden'."
    ),
    allow_flagging="never"
)

# Launch Gradio App
interface.launch()
