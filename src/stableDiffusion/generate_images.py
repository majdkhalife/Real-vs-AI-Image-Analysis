import torch
from diffusers import StableDiffusionPipeline
import os
from tqdm import tqdm
import random
import argparse
import ast
from pathlib import Path

def parse_prompt_file(file_path):
    """Parse the prompt file and extract components."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Create a namespace to execute the file content safely
    namespace = {}
    exec(content, namespace)
    
    # Extract components, using empty lists as fallbacks
    landscape_types = namespace.get('landscape_types', [])
    conditions = namespace.get('conditions', [])
    natural_elements = namespace.get('natural_elements', [])
    negative_prompt = namespace.get('negative_prompt', '')
    
    return landscape_types, conditions, natural_elements, negative_prompt

def generate_prompt(landscape_types, conditions, natural_elements):
    """Generate a single prompt by combining elements."""
    prompt_parts = [
        random.choice(landscape_types),
        random.choice(conditions),
        random.choice(natural_elements)
    ]
    return ", ".join(prompt_parts)

def setup_pipeline():
    """Initialize the Stable Diffusion pipeline."""
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        use_safetensors=True
    )
    pipe = pipe.to("cuda")
    return pipe

def generate_dataset(
    pipe,
    num_images,
    output_dir,
    dataset_name,
    landscape_types,
    conditions,
    natural_elements,
    negative_prompt,
    width=800,
    height=600,
    seed=None
):
    """Generate a dataset of images using structured prompts."""
    os.makedirs(output_dir, exist_ok=True)
    
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
    
    # Create a log file for all prompts
    log_file_path = os.path.join(output_dir, f"{dataset_name}_prompts_log.txt")
    with open(log_file_path, 'w') as log_file:
        # Generate images with progress bar
        for i in tqdm(range(num_images), desc="Generating images"):
            try:
                # Generate prompt
                prompt = generate_prompt(landscape_types, conditions, natural_elements)
                
                # Generate the image
                image = pipe(
                    prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=30,
                    guidance_scale=7.5,
                    width=width,
                    height=height
                ).images[0]
                
                # Save the image and prompt
                image_filename = f"{dataset_name}_{i:05d}.png"
                image_path = os.path.join(output_dir, image_filename)
                image.save(image_path)
                
                # Log the prompt
                log_file.write(f"{image_filename}: {prompt}\n")
                log_file.flush()  # Ensure prompt is written immediately
                
            except Exception as e:
                print(f"Error generating image {i}: {str(e)}")
                continue

def main():
    parser = argparse.ArgumentParser(description='Generate images from structured prompt file')
    parser.add_argument('prompt_file', type=str, help='Path to the prompt configuration file')
    parser.add_argument('--num-images', type=int, default=1000, help='Number of images to generate')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Get the dataset name from the prompt file name (without extension)
    dataset_name = Path(args.prompt_file).stem
    
    # Parse the prompt file
    landscape_types, conditions, natural_elements, negative_prompt = parse_prompt_file(args.prompt_file)
    
    # Set up the pipeline
    print("Setting up Stable Diffusion pipeline...")
    pipe = setup_pipeline()
    
    # Generate the dataset
    print("Starting image generation...")
    generate_dataset(
        pipe=pipe,
        num_images=args.num_images,
        output_dir=f"stableDiffusion_dataset",
        dataset_name=dataset_name,
        landscape_types=landscape_types,
        conditions=conditions,
        natural_elements=natural_elements,
        negative_prompt=negative_prompt,
        seed=args.seed
    )
    
    print("Done")

if __name__ == "__main__":
    main()
