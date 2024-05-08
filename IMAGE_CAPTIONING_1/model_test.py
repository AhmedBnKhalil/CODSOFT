# Load the model
from keras.src.saving import load_model
from main import *
model = load_model(model_path)



# Generate caption for a new image
new_image_path = 'flickr30k_images'
new_features = extract_features(new_image_path, max_images=1)
new_image_id = list(new_features.keys())[0]
caption = generate_caption(model, tokenizer, new_features[new_image_id], max_length)
print("Generated Caption:", caption)