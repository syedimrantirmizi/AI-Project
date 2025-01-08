import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Load the trained model
model = load_model("vehicle_classifier.h5")

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
    
    # Display the image with the predicted class
    plt.figure(figsize=(5,5))
    plt.suptitle("Predictions")
    plt.imshow(image.load_img(img_path, target_size=(224, 224)))  # Display the image
    plt.axis("off")  # Turn off the axes
    plt.title(f"Predicted: {predicted_class}", fontsize=20)  # Add the predicted class as the title
    plt.savefig("./singleimagetest/singleImage.png")
    plt.show()
    plt.close()

    confidence_text = "\n".join(
    [f"{index_to_class[idx]}: {predictions[0][idx] * 100:.2f}%" for idx in range(len(predictions[0]))])
    plt.figure(figsize=(5,5))
    plt.subplot(2, 1, 1)
    plt.axis("off")
    plt.title(f"Confidence Scores", fontsize=25, loc='center')  # Confidence scores above image
    plt.gcf().text(
        0.5, 0.3, confidence_text, fontsize=18, ha="center", transform=plt.gca().transAxes
    )
    plt.tight_layout()
    plt.savefig("./singleimagetest/confidence_score.png", bbox_inches="tight", dpi=300)
    plt.show()
    plt.close()
    print("\n")
# Path to the test image
test_image_path = "./dataset/Car/car523.jpg"  # Replace with the path to your image

# Predict and display the result
predict_and_display_image(test_image_path)
