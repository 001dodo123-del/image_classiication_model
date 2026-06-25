import streamlit as st
import requests
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import io
import torch
from torchvision.ops import nms
from PIL import ImageEnhance, Image

# Streamlit UI
def main():
    st.title("Image Recognition App")

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        uploaded_bytes = uploaded_file.read()
        uploaded_file.seek(0)

        image_pil = Image.open(io.BytesIO(uploaded_bytes)).convert("RGB")
        st.image(image_pil, caption="Uploaded Image", use_column_width=True)

        confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.05)
        iou_threshold = st.slider("NMS IoU threshold", 0.0, 1.0, 0.45, 0.05)
        max_boxes = st.slider("Max boxes to display", 1, 20, 10)

        if st.button("Detect Objects"):
            files = {"file": (uploaded_file.name or "image.jpg", uploaded_bytes, uploaded_file.type or "image/jpeg")}
            response = requests.post("http://127.0.0.1:8000/detect_objects/", files=files)

            if response.status_code == 200:
                detection_results = response.json()
                boxes = detection_results.get("boxes", [])
                scores = detection_results.get("scores", [])
                classes = detection_results.get("classes", [])

                if len(boxes) == 0:
                    st.warning("The API returned no detections. Check that the FastAPI server is running and the correct endpoint is configured.")
                    return

                boxes_np = np.array(boxes, dtype=np.float32)
                scores_np = np.array(scores, dtype=np.float32)
                classes_np = np.array(classes, dtype=object)

                boxes_tensor = torch.from_numpy(boxes_np)
                scores_tensor = torch.from_numpy(scores_np)

                # Run Non-Maximum Suppression to remove duplicate overlapping boxes
                keep = nms(boxes_tensor, scores_tensor, float(iou_threshold))
                keep = keep[torch.argsort(scores_tensor[keep], descending=True)]
                keep = keep[:int(max_boxes)]

                boxes_np = boxes_np[keep.numpy()]
                scores_np = scores_np[keep.numpy()]
                classes_np = classes_np[keep.numpy()]

                image_np = np.array(image_pil)
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.imshow(image_np)

                brightness = ImageEnhance.Brightness(Image.fromarray(image_np))
                enhanced_image = brightness.enhance(1.1)
                ax.imshow(enhanced_image)

                detected = 0
                for box, score, cls in zip(boxes_np, scores_np, classes_np):
                    if score >= confidence_threshold:
                        detected += 1
                        xmin, ymin, xmax, ymax = box.tolist()
                        width = xmax - xmin
                        height = ymax - ymin
                        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor="r", facecolor="none")
                        ax.add_patch(rect)
                        ax.text(xmin, ymin - 5, f"{cls} ({score:.2f})", fontsize=9, color="r", backgroundcolor="white")

                ax.axis("off")
                st.pyplot(fig)
                st.write(f"API returned {len(boxes)} total predictions.")
                st.write(f"Top {len(boxes_np)} predictions after NMS.")
                st.write(f"Max score: {float(scores_np.max()):.2f}")
                st.write(f"Predictions above threshold: {detected}")

                if detected == 0:
                    st.warning(
                        "No objects detected above the selected confidence threshold. "
                        "Try lowering the threshold or using a clearer object image."
                    )
                else:
                    st.success(f"Detected {detected} object(s) above the threshold.")
            else:
                st.error(f"Error performing object detection: {response.status_code}")
                st.write(response.text)

if __name__ == "__main__":
    main()
