import requests

# API endpoint
url = "http://localhost:8000/predict"

# Path to your test image
image_path = "data\predictions\digit_3_20250612_170411.png"  # Replace with your image path

# Prepare the files for the request
files = {
    'file': ('test_image.png', open(image_path, 'rb'), 'image/png')
}

# Make the POST request
response = requests.post(url, files=files)

# Print the response
print("Status Code:", response.status_code)
print("Response:", response.json()) 