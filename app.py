import streamlit as st
from PIL import Image
from ultralytics import YOLO

# --- Page Configuration ---
st.set_page_config(page_title="Cow Skin Disease Analyzer", layout="centered")

st.title("Cow Skin Disease Detection")
st.write("Upload an image of the cow's skin to detect and learn about potential conditions.")

# --- Disease Descriptions ---
disease_descriptions = {
    "Healthy Skin": "The skin appears normal and healthy. No visible signs of Lumpy Skin Disease, Ringworm, or Lumpy Jaw were detected. Continue regular monitoring and good hygiene practices.",
    "LSD": "**Lumpy Skin Disease (LSD)** is a viral infection characterized by fever, enlarged lymph nodes, and firm, raised nodules (lumps) on the skin. It is highly contagious among cattle and can cause significant drops in milk production.",
    "Ringworm": "**Ringworm** is a contagious fungal infection of the skin. It typically presents as circular, grayish-white, crusty, and hairless patches, often starting around the head, neck, and shoulders.",
    "Lumpy Jaw": "**Lumpy Jaw (Actinomycosis)** is a bacterial infection that causes hard, immovable lumps, usually on the jawbone. While it primarily affects the bone, the severe localized swelling is highly visible on the animal's face and skin."
}

# --- Load Model ---
@st.cache_resource
def load_model():
    # Replace 'best.pt' with the actual path to your trained weights file
    return YOLO("./weight/best_200.pt")

try:
    model = load_model()
except Exception as e:
    st.error("Error loading the model. Please ensure your trained weights file (e.g., 'best.pt') is in the same directory as this script.")
    st.stop()

# --- Main Screen UI Elements ---
# Confidence slider on the main screen
conf_threshold = st.slider(
    "Confidence Threshold", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.50, 
    step=0.05,
    help="Adjust this slider to filter out less certain predictions."
)

uploaded_file = st.file_uploader("Upload Image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Uploaded Image", use_container_width=True)

    # Analyze Button
    if st.button("Analyze Image", type="primary"):
        with st.spinner("Analyzing..."):
            # Run inference
            results = model.predict(source=image, conf=conf_threshold)
            
            # Extract the plotted image array (YOLO returns BGR, Streamlit needs RGB)
            res_plotted = results[0].plot()[:, :, ::-1] 
            
            st.markdown("### Prediction Results")
            st.image(res_plotted, caption="Predicted Image with Bounding Boxes", use_container_width=True)

            # Extract unique detected classes to display descriptions
            detected_classes = set()
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                detected_classes.add(class_name)

            # Display Descriptions
            st.markdown("### Disease Description")
            if not detected_classes:
                st.warning(f"No classes detected above the {conf_threshold} confidence threshold.")
            else:
                for cls in detected_classes:
                    st.success(f"**Detected:** {cls}")
                    st.info(disease_descriptions.get(cls, "No description available for this class."))
