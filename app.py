import gradio as gr
import torch
import timm
import numpy as np
from PIL import Image
import json
from torchvision import transforms as T

CLASS_LABELS = {
    0: "Mild Dementia",
    1: "Moderate Dementia",
    2: "Non Demented",
    3: "Very Mild Dementia",
    4: "Glioma",
    5: "Healthy",
    6: "Meningioma",
    7: "Pituitary",
}

ICD10_MAP = {
    "Mild Dementia":       {"code": "G30.9",  "desc": "Alzheimer's disease, unspecified"},
    "Moderate Dementia":   {"code": "G30.9",  "desc": "Alzheimer's disease, unspecified"},
    "Non Demented":        {"code": "Z03.89", "desc": "No abnormality detected"},
    "Very Mild Dementia":  {"code": "G30.0",  "desc": "Alzheimer's disease with early onset"},
    "Glioma":              {"code": "C71.9",  "desc": "Malignant neoplasm of brain, unspecified"},
    "Healthy":             {"code": "Z03.89", "desc": "No abnormality detected"},
    "Meningioma":          {"code": "D32.9",  "desc": "Benign neoplasm of meninges, unspecified"},
    "Pituitary":           {"code": "D35.2",  "desc": "Benign neoplasm of pituitary gland"},
}

CATEGORY_MAP = {
    "Mild Dementia":      "alzheimer",
    "Moderate Dementia":  "alzheimer",
    "Non Demented":       "normal",
    "Very Mild Dementia": "alzheimer",
    "Glioma":             "tumor",
    "Healthy":            "normal",
    "Meningioma":         "tumor",
    "Pituitary":          "tumor",
}

print("Loading Swin Transformer model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = timm.create_model(
    "swin_base_patch4_window7_224",
    pretrained=False,
    num_classes=8,
)

state = torch.load(
    "swin_brain_merged_best_model.pth",
    map_location=device,
    weights_only=False,
)

if isinstance(state, dict):
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    elif "state_dict" in state:
        state = state["state_dict"]
    state = {k.replace("module.", ""): v for k, v in state.items()}

model.load_state_dict(state, strict=False)
model.to(device)
model.eval()
print(f"Model loaded on {device}")

# Standard ImageNet transforms for Swin
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


def predict(image: Image.Image):
    if image is None:
        return json.dumps({"error": "No image provided"})

    if image.mode != "RGB":
        image = image.convert("RGB")

    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

    probs_np  = probs.cpu().numpy()
    pred_idx  = int(np.argmax(probs_np))
    pred_label = CLASS_LABELS[pred_idx]
    confidence = float(probs_np[pred_idx])
    icd = ICD10_MAP.get(pred_label, {"code": "Z03.89", "desc": "Unknown"})
    cat = CATEGORY_MAP.get(pred_label, "unknown")

    result = {
        "predicted_class":   pred_label,
        "class_index":       pred_idx,
        "confidence":        round(confidence, 4),
        "category":          cat,
        "icd10_code":        icd["code"],
        "icd10_description": icd["desc"],
        "all_probabilities": {CLASS_LABELS[i]: round(float(probs_np[i]), 4) for i in range(8)},
        "model":             "swin_base_patch4_window7_224",
        "device":            str(device),
    }
    return json.dumps(result)


with gr.Blocks(title="Brain MRI Classifier") as demo:
    gr.Markdown("## Brain MRI Classification — ClaimIQ\nSwin Transformer · 8 classes · Alzheimer stages + Brain tumors")
    with gr.Row():
        img_input   = gr.Image(type="pil", label="Upload MRI Scan")
        json_output = gr.Textbox(label="Result (JSON)", lines=18)
    gr.Button("Classify", variant="primary").click(predict, img_input, json_output)
    gr.Markdown("**API:** `POST /run/predict` with `{\"data\": [<image>]}`")

if __name__ == "__main__":
    demo.launch()