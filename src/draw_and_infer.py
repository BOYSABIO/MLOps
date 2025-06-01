# draw_and_infer.py

import os
import cv2
import numpy as np
import torch
from datetime import datetime
from src.inference.inference import load_trained_model, predict_digit

# Ensure predictions folder exists
os.makedirs("data/predictions", exist_ok=True)

def preprocess_canvas(canvas):
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    # Find bounding box of digit
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        digit = thresh[y:y+h, x:x+w]
        digit = cv2.resize(digit, (20, 20), interpolation=cv2.INTER_AREA)

        # Pad to 28×28
        padded = np.zeros((28, 28), dtype=np.uint8)
        x_offset = (28 - 20) // 2
        y_offset = (28 - 20) // 2
        padded[y_offset:y_offset+20, x_offset:x_offset+20] = digit
    else:
        padded = np.zeros((28, 28), dtype=np.uint8)

    norm = padded.astype("float32") / 255.0
    tensor = torch.tensor(norm).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 28, 28)
    return tensor, padded

def save_prediction_image(image_28x28, predicted_digit):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data/predictions/digit_{predicted_digit}_{timestamp}.png"
    cv2.imwrite(filename, image_28x28 * 255)
    print(f"[INFO] Saved prediction image: {filename}")

def main():
    model = load_trained_model(model_path="models/pytorch_mnist_model.pth", device="cpu")

    canvas = np.zeros((280, 280, 3), dtype=np.uint8)

    drawing = False
    last_point = None
    last_prediction = None

    def draw(event, x, y, flags, param):
        nonlocal drawing, last_point
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            last_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            cv2.line(canvas, last_point, (x, y), (255, 255, 255), 32)
            last_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False

    cv2.namedWindow("Draw a digit (SPACE to predict | ESC to quit | C to clear)")
    cv2.setMouseCallback("Draw a digit (SPACE to predict | ESC to quit | C to clear)", draw)

    while True:
        display = canvas.copy()

        # Display controls
        cv2.putText(display, "SPACE: Predict | ESC: Quit | C: Clear",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        if last_prediction is not None:
            cv2.putText(display, f"Prediction: {last_prediction}",
                        (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Draw a digit (SPACE to predict | ESC to quit | C to clear)", display)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break
        elif key == ord("c"):  # Clear
            canvas[:] = 0
            last_prediction = None
        elif key == 32:  # SPACE → Predict
            tensor, img28 = preprocess_canvas(canvas)
            prediction = predict_digit(model, tensor)
            last_prediction = int(prediction)
            save_prediction_image(img28, last_prediction)
            canvas[:] = 0  # Clear for next

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
