

# **Text2Image_Generation_Huggingface**
This project demonstrates how to generate high-quality, photorealistic images from textual descriptions using Stable Diffusion models. It is implemented in Python with Hugging Face's `diffusers` library and PyTorch, showcasing a variety of prompts and configurations for image generation.


![Generated Example](generated_image_example.png)  
*Example output for "A futuristic cityscape with neon lights" (generated via `dreamlike-diffusion-1.0`)*

## 📝 Description
This project leverages pre-trained Stable Diffusion models from Hugging Face's `diffusers` library to generate high-quality images from text prompts. It demonstrates customization through parameter tuning (e.g., resolution, inference steps) and compares outputs from different model variants.

## ✨ Features
- Generate images from text prompts using multiple Stable Diffusion models.
- Adjustable parameters for fine-tuning outputs:
  - Image resolution (`height`, `width`)
  - Inference steps (`num_inference_steps`)
  - Batch generation (`num_images_per_prompt`)
  - Negative prompting to exclude unwanted elements
- Built-in visualization with `matplotlib`.

## 🛠️ Prerequisites
- **Python 3.10+**
- **GPU-enabled environment** (e.g., Google Colab, local CUDA setup)
- Libraries:
  ```bash
  torch==2.5.1+cu121  # CUDA 12.1 compatible
  diffusers>=0.31.0
  transformers>=4.46.2
  accelerate>=1.1.1
  matplotlib>=3.7.0
  Pillow>=11.0.0
  ```
## Clone the Repository
```bash
git clone https://github.com/Nazmul0005/Text2Image_Generation_HuggingFace.git
```
## 🚀 Installation
1. **Install PyTorch with CUDA** (adjust based on your CUDA version):
   ```bash
   pip install torch==2.5.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
   ```
2. **Install required libraries**:
   ```bash
   pip install diffusers transformers accelerate matplotlib Pillow
   ```

##  Usage

### 1. Load a Model
```python
from diffusers import StableDiffusionPipeline
import torch

# Choose from supported models:
model_id = "dreamlike-art/dreamlike-diffusion-1.0"  # Example model
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")  # Ensure GPU is used
```

### 2. Generate an Image
```python
prompt = "A futuristic cityscape with towering skyscrapers and neon lights"
image = pipe(prompt).images[0]
image.save("generated_image.png")
```

### 3. Customize Parameters
```python
def generate_image(pipe, prompt, params):
    images = pipe(prompt, **params).images
    # Visualization code (requires matplotlib)
    # ... (see full function in the notebook)

params = {
    "num_inference_steps": 50,
    "height": 640,
    "width": 512,
    "num_images_per_prompt": 2,
    "negative_prompt": "blurry, distorted, low quality"
}

generate_image(pipe, prompt, params)
```

## 🌟 Example Prompts
| Category      | Prompt Example                                                                 |
|---------------|--------------------------------------------------------------------------------|
| **Fantasy**   | `"A majestic unicorn galloping through wildflowers under a golden sunset."`    |
| **Cyberpunk** | `"A neon-lit cyberpunk city with flying cars and rain-soaked streets."`        |
| **Cinematic** | `"A girl sitting with her tiger, golden lighting, cinematic atmosphere."`      |
| **Historical**| `"A medieval marketplace with merchants, cobblestone streets, and horses."`    |

## Key Examples
Example 1: Generate a Grungy Woman Traveling Between Dimensions
```python
prompt = """dreamlikeart, a grungy young woman with rainbow hair, traveling between dimensions, dynamic pose, happy, soft eyes, and narrow chin, extreme bokeh, dainty figure, long hair straight down, torn kawali shirt, and baggy jeans"""

image = pipe(prompt).images[0]
plt.imshow(image)
plt.axis("off")
```
Example 2: Customize Parameters
```python
params = {
    "num_inference_steps": 50,
    "height": 640,
    "width": 512,
    "num_images_per_prompt": 2,
    "negative_prompt": "ugly, distorted, low quality"
}

generate_image(pipe, prompt, params)
```
## 🧠 Model Variants
Tested models include:
- [`dreamlike-art/dreamlike-diffusion-1.0`](https://huggingface.co/dreamlike-art/dreamlike-diffusion-1.0)
- [`stabilityai/stable-diffusion-x1-base-1.0`](https://huggingface.co/stabilityai/stable-diffusion-x1-base)
- [`stabilityai/stable-diffusion-2-1-base`](https://huggingface.co/stabilityai/stable-diffusion-2-1-base)

## 🛠️ Models Used

| Model ID                                   | Description                                      |
|--------------------------------------------|--------------------------------------------------|
| `dreamlike-art/dreamlike-diffusion-1.0`    | Artistic, dreamlike imagery generation           |
| `stabilityai/stable-diffusion-x1-base-1.0` | Base Stable Diffusion model for high-quality outputs |
| `stabilityai/stable-diffusion-2-1-base`    | Advanced model for diverse use cases            |
| `stabilityai/stable-diffusion-2-1`         | Enhanced version with better output precision   |


## ⚙️ Key Parameters
| Parameter                | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| `num_inference_steps`    | Number of denoising steps (higher = more detailed, slower). Default: `50`   |
| `height` & `width`       | Output resolution. Recommended: Multiples of 64 (e.g., 512x768).           |
| `num_images_per_prompt`  | Number of images to generate in one batch.                                  |
| `negative_prompt`        | Exclude undesired elements (e.g., `"ugly, distorted, text"`).               |

## 🔧 Functionality Highlights

- **Negative Prompting**: Specify a negative prompt to exclude undesired elements from the image.
- **Resolution Scaling**: Customize image dimensions with height and width parameters.
- **Iterative Improvements**: Experiment with different models and parameters to achieve the best results.

## 📜 License
- **Code**: This project is licensed under the GNU General Public License v3.0 - see the LICENSE file for details.
- **Models**: Check individual model licenses on Hugging Face Hub.

## 🙏 Acknowledgments
- Built with [Hugging Face Diffusers](https://huggingface.co/docs/diffusers).
- Models provided by [Stability AI](https://stability.ai/) and [Dreamlike Art](https://dreamlike.art/).

## 🤝 How to Contribute

**Contributions are welcome! Here's how you can contribute:**

1. **Fork the repository.**

2. **2.Create a feature branch:**
   ```bash
   git checkout -b feature-name
   ```
3. **Commit your changes:**
   ```bash
   git commit -m "Add some feature"
   ```

4. **Push to the branch:**
   ```bash
    git push origin feature-name
   ```

6. **Open a pull request.**



---

**Note**: For optimal performance, use a GPU environment like Google Colab. CPU inference is not recommended.  
``` 

This README.md includes proper markdown formatting, code blocks, tables, and placeholders for images. Replace `generated_image_example.png` with your own output samples.
