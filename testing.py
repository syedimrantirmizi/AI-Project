import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

select_Model = {1 : "MobileNetV2", 2 : "resnet50", 3 : "DenseNet121"}

# Specify the model to use
model_name = select_Model[1]
model_path = f"vehicle_classifier_{model_name.lower()}.h5"

# Load the trained model
model = load_model(model_path)

# Manually define the class mapping
# Ensure the order matches the dataset structure (based on train_generator.class_indices)
index_to_class = {0: "Bus", 1: "Car", 2: "Truck", 3: "Motorcycle"}

# Function to preprocess, predict, and display a single image
def predict_and_display_image(img_path):
    # Load the image and resize it to match the model's input size
    img = image.load_img(img_path, target_size=(224, 224))  # Resize to 224x224
    img_array = image.img_to_array(img) / 255.0  # Normalize pixel values to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make predictions
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)  # Get index of the highest probability
    predicted_class = index_to_class[predicted_index]  # Map index to class name

    # Print the predicted class and confidence scores
    print("\n")
    print(f"Predicted Class: {predicted_class}")

    # Define the output directory based on the model used
    output_dir = f"./prediction/{model_name}/singleimagetest"
    os.makedirs(output_dir, exist_ok=True)

    # Save and display the prediction image
    plt.figure(figsize=(5, 5))
    plt.imshow(image.load_img(img_path, target_size=(224, 224)))  # Display the image
    plt.axis("off")  # Turn off the axes
    plt.title(f"Predicted: {predicted_class}", fontsize=20)  # Add the predicted class as the title
    plt.savefig(os.path.join(output_dir, "singleImage.png"))
    plt.show()
    plt.close()

    # Save and display the confidence scores
    confidence_text = "\n".join(
        [f"{index_to_class[idx]}: {predictions[0][idx] * 100:.2f}%" for idx in range(len(predictions[0]))]
    )
    plt.figure(figsize=(5, 5))
    plt.subplot(2, 1, 1)
    plt.axis("off")
    plt.title("Confidence Scores", fontsize=25, loc="center")  # Confidence scores above image
    plt.gcf().text(
        0.5, 0.3, confidence_text, fontsize=18, ha="center", transform=plt.gca().transAxes
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confidence_score.png"), bbox_inches="tight", dpi=300)
    plt.show()
    plt.close()

    print("\n")

# Path to the test image
test_image_path = "./testimage.jpeg"  # Replace with the path to your image

# Predict and display the result
predict_and_display_image(test_image_path)
