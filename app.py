import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
import timm


# Load and display logo
logo_path = 'C:\\Lung Cancer Detection\\Assets\\setv_global_cover.jpeg'  # Using a raw string literal
logo = Image.open(logo_path)
st.image(logo, width=200)


# Define the model architecture
def create_model():
    model = timm.create_model('resnet50', pretrained=True)
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(2048, 256),
        torch.nn.Dropout(0.2),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 64),
        torch.nn.Dropout(0.2),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 32),
        torch.nn.Dropout(0.2),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 4),
        torch.nn.Softmax(dim=1)
    )
    return model

# Load the saved model
@st.cache(allow_output_mutation=True)
def load_model(model_path):
    model = create_model()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model('C:\Lung Cancer Detection\Model\model_ResNet50_best.h5')  # Update path

# Image preprocessing function
def process_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Prediction function
def predict(image, model):
    image = process_image(image)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return predicted.item()

# Class names
class_name = ['adenocarcinoma', 'large.cell.carcinoma', 'normal', 'squamous.cell.carcinoma']

# Streamlit UI
st.title("SETV Lung Cancer Detection")
st.write("Upload a CT Scan image for lung cancer detection.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg", "webp"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("Classifying...")
    label_index = predict(image, model)
    label = class_name[label_index]
    st.write(f'Prediction: {label}')
