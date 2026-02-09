"""
YOLOE Prompt-Free App with Cached Vocabulary

This version uses pre-computed vocabulary embeddings for instant loading.
No text model is loaded at runtime - vocabulary is loaded from cache files.

Before running this app, you must run:
    python cache_vocab.py

This will create cached_vocab/ directory with pre-computed embeddings.
"""

import os
import torch
import numpy as np
import gradio as gr
import supervision as sv
from PIL import Image
from huggingface_hub import hf_hub_download
from ultralytics import YOLOE

# Configuration
CACHE_DIR = "cached_vocab"
VOCAB_FILE = "tools/ram_tag_list.txt"

# Model IDs
MODEL_IDS = [
    "yoloe-v8s",
    "yoloe-v8m",
    "yoloe-v8l",
    "yoloe-11s",
    "yoloe-11m",
    "yoloe-11l",
]

# Global cache for loaded vocabularies
# This ensures we only load from disk once per model
_vocab_cache = {}
_model_cache = {}


def load_cached_vocab(model_id, cache_dir=CACHE_DIR):
    """Load cached vocabulary from disk."""
    global _vocab_cache
    
    if model_id in _vocab_cache:
        return _vocab_cache[model_id]
    
    cache_path = os.path.join(cache_dir, f"{model_id}_vocab.pt")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"Cached vocabulary not found for {model_id}.\n"
            f"Please run 'python cache_vocab.py' first to generate cache files.\n"
            f"Expected file: {cache_path}"
        )
    
    print(f"Loading cached vocabulary for {model_id}...")
    cache = torch.load(cache_path, map_location="cpu")
    _vocab_cache[model_id] = cache
    print(f"  - Loaded {len(cache['texts'])} classes")
    return cache


def init_pf_model(model_id):
    """Initialize prompt-free model (cached)."""
    global _model_cache
    
    if model_id in _model_cache:
        return _model_cache[model_id]
    
    filename = f"{model_id}-seg-pf.pt"
    path = hf_hub_download(repo_id="jameslahm/yoloe", filename=filename)
    model = YOLOE(path)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    _model_cache[model_id] = model
    return model


def set_vocab_from_cache(model, cache, device):
    """Set vocabulary on model from cached weights."""
    head = model.model.model[-1]
    
    # Move vocab weights to device
    vocab_weights = [w.to(device) for w in cache["vocab_weights"]]
    
    # Create ModuleList and set weights
    from torch import nn
    vocab = nn.ModuleList()
    for i, cls_head in enumerate(head.cv3):
        assert isinstance(cls_head, nn.Sequential)
        # Replace the last layer with cached weights
        new_layer = nn.Linear(vocab_weights[i].shape[1], vocab_weights[i].shape[0], bias=False)
        new_layer.weight.data = vocab_weights[i]
        vocab.append(new_layer)
    
    # Set up the model with cached vocabulary
    # Warmup
    model.model(torch.empty(1, 3, 640, 640).to(device))
    
    # Set up LRPC heads
    from ultralytics.nn.modules.head import LRPCHead
    head.lrpc = nn.ModuleList(
        LRPCHead(cls, pf[-1], loc[-1], enabled=i!=2) 
        for i, (cls, pf, loc) in enumerate(zip(vocab, head.cv3, head.cv2))
    )
    
    # Remove original heads
    for loc_head, cls_head in zip(head.cv2, head.cv3):
        del loc_head[-1]
        del cls_head[-1]
    
    head.nc = len(cache["texts"])
    from ultralytics.utils import check_class_names
    model.model.names = check_class_names(cache["texts"])
    
    return model


def yoloe_pf_inference(image, model_id, image_size, conf_thresh, iou_thresh):
    """Run prompt-free inference with cached vocabulary."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load cached vocabulary and model
    vocab_cache = load_cached_vocab(model_id)
    model = init_pf_model(model_id)
    
    # Set vocabulary from cache
    set_vocab_from_cache(model, vocab_cache, device)
    
    # Configure model
    model.model.model[-1].is_fused = True
    model.model.model[-1].conf = 0.001
    model.model.model[-1].max_det = 1000
    
    # Run inference
    results = model.predict(source=image, imgsz=image_size, conf=conf_thresh, iou=iou_thresh)
    detections = sv.Detections.from_ultralytics(results[0])
    
    # Annotate results
    resolution_wh = image.size if isinstance(image, Image.Image) else (image.shape[1], image.shape[0])
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=resolution_wh)
    
    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence
        in zip(detections['class_name'], detections.confidence)
    ]
    
    annotated_image = image.copy() if isinstance(image, Image.Image) else Image.fromarray(image)
    annotated_image = sv.MaskAnnotator(
        color_lookup=sv.ColorLookup.INDEX,
        opacity=0.4
    ).annotate(scene=annotated_image, detections=detections)
    annotated_image = sv.BoxAnnotator(
        color_lookup=sv.ColorLookup.INDEX,
        thickness=thickness
    ).annotate(scene=annotated_image, detections=detections)
    annotated_image = sv.LabelAnnotator(
        color_lookup=sv.ColorLookup.INDEX,
        text_scale=text_scale,
        smart_position=True
    ).annotate(scene=annotated_image, detections=detections, labels=labels)
    
    return annotated_image


def create_app():
    """Create the Gradio app interface."""
    with gr.Blocks() as demo:
        gr.HTML(
            """
            <h1 style='text-align: center'>
            YOLOE: Real-Time Seeing Anything (Prompt-Free Mode)
            </h1>
            <h3 style='text-align: center; color: #666;'>
            Fast Loading Version with Cached Vocabulary
            </h3>
            """)
        
        gr.Markdown(
            """
            This version uses **pre-computed vocabulary embeddings** for instant loading.
            No text model is loaded at runtime - everything is cached for fast startup.
            
            **Note:** Before first use, run `python cache_vocab.py` to generate cache files.
            """
        )
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil", label="Input Image", interactive=True)
                
                model_id = gr.Dropdown(
                    label="Model",
                    choices=MODEL_IDS,
                    value="yoloe-v8l",
                )
                
                with gr.Row():
                    image_size = gr.Slider(
                        label="Image Size",
                        minimum=320,
                        maximum=1280,
                        step=32,
                        value=640,
                    )
                    conf_thresh = gr.Slider(
                        label="Confidence Threshold",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        value=0.25,
                    )
                    iou_thresh = gr.Slider(
                        label="IoU Threshold",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        value=0.70,
                    )
                
                detect_btn = gr.Button(value="Detect & Segment Objects", variant="primary")
                
                # Status display
                status_text = gr.Textbox(
                    label="Status",
                    value="Ready - Click 'Detect' to start",
                    interactive=False
                )
            
            with gr.Column():
                output_image = gr.Image(type="numpy", label="Annotated Image")
        
        # Example
        gr.Examples(
            examples=[
                ["ultralytics/assets/bus.jpg", "yoloe-v8l", 640, 0.25, 0.7],
            ],
            inputs=[input_image, model_id, image_size, conf_thresh, iou_thresh],
            label="Example",
            cache_examples=False,
        )
        
        def run_inference(image, model_id, image_size, conf_thresh, iou_thresh):
            if image is None:
                return None, "Error: Please upload an image first"
            
            try:
                # Check if cache exists
                cache_path = os.path.join(CACHE_DIR, f"{model_id}_vocab.pt")
                if not os.path.exists(cache_path):
                    return None, f"Error: Cache not found for {model_id}. Run 'python cache_vocab.py' first."
                
                status_text.value = f"Loading cached vocabulary for {model_id}..."
                result = yoloe_pf_inference(image, model_id, image_size, conf_thresh, iou_thresh)
                return result, f"Success! Using cached vocabulary for {model_id}"
            except Exception as e:
                import traceback
                traceback.print_exc()
                return None, f"Error: {str(e)}"
        
        detect_btn.click(
            fn=run_inference,
            inputs=[input_image, model_id, image_size, conf_thresh, iou_thresh],
            outputs=[output_image, status_text],
        )
    
    return demo


if __name__ == "__main__":
    # Check if cache exists
    if not os.path.exists(CACHE_DIR):
        print("=" * 70)
        print("WARNING: Cache directory not found!")
        print("=" * 70)
        print("\nBefore running this app, you need to generate cache files:")
        print("    python cache_vocab.py")
        print("\nThis will pre-compute vocabulary embeddings for all models.")
        print("=" * 70)
    else:
        cached_files = [f for f in os.listdir(CACHE_DIR) if f.endswith('_vocab.pt')]
        print(f"Found {len(cached_files)} cached vocabularies:")
        for f in cached_files:
            size_mb = os.path.getsize(os.path.join(CACHE_DIR, f)) / 1024 / 1024
            print(f"  - {f} ({size_mb:.2f} MB)")
    
    print("\nStarting Gradio app...")
    app = create_app()
    app.launch(share=False)
