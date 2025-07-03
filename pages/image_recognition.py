import streamlit as st

from PIL import Image
import torch
import torch.nn.functional as F

from torchvision import transforms

@st.cache_resource
def load_model(model_name):
    model = torch.load(f'models/{model_name}.pth', map_location=torch.device('cpu'), weights_only=False)
    model.eval()
    return model

class_labels = ["other_activities", "safe_driving", "talking_phone", "texting_phone", "turning"]
model_name = "Modelo_Preentrenado_dataAug_v2_20250701_180011_full_model"
image_size = 224



transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],   #RGB mean values of Resnet dataset
                         [0.229, 0.224, 0.225])
])

st.set_page_config(layout="wide")


st.set_page_config(page_title="Image Prediction App")

st.title("🔍 Image Classification App")
st.write("Upload an image and the app will use a trained model to make a prediction.")

# File uploader for images
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

model = load_model(model_name)

# If an image is uploaded, display it
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    image = image.convert("RGB").resize((image_size, image_size))

    
    if st.button("Predict"):
        image_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = F.softmax(output, dim=1).squeeze()
            predicted_class = output.argmax(dim=1).item() 
        top_probs, top_idxs = torch.topk(probabilities, k=3)
        top_probs = top_probs.tolist()
        top_idxs = top_idxs.tolist()

        # Display result
        st.subheader("Top 3 Predictions:")
        for i in range(3):
            label = class_labels[top_idxs[i]]
            conf = top_probs[i]
            st.write(f"{i+1}. **{label}** – {conf:.2%} confidence")