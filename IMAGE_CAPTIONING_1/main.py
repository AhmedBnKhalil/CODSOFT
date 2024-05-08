import os
import numpy as np
import pandas as pd
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, LSTM, Embedding, Add
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import corpus_bleu

# Path settings
images_path = 'flickr30k_images'
caption_path = 'results.csv'
model_path = 'saved_model.keras'

def load_doc(filename):
    with open(filename, 'r') as file:
        text = file.read()
    return text

def load_descriptions(filename):
    data = pd.read_csv(filename, delimiter='|')
    print("Data loaded from CSV:", data.head())  # Print the first few rows of the DataFrame
    data.columns = data.columns.str.strip()
    data['image_name'] = data['image_name'].str.strip()
    data['comment'] = data['comment'].astype(str).str.strip()

    descriptions = data.groupby('image_name')['comment'].apply(list).to_dict()

    if not descriptions:
        print("No descriptions found. Check the CSV file and delimiter.")

    for img_id, captions in descriptions.items():
        descriptions[img_id] = ['startseq ' + str(caption) + ' endseq' for caption in captions if caption != 'nan']

    return descriptions



def extract_features(directory, max_images=10):
    """Extract features for images that have descriptions."""
    base_model = VGG16(weights='imagenet', include_top=False)
    model = Model(inputs=base_model.input, outputs=Flatten()(base_model.output))  # Adding Flatten layer directly
    features = {}
    image_files = os.listdir(directory)[:max_images]
    for img_name in image_files:
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue  # Ensure only image files are processed
        filename = os.path.join(directory, img_name)
        try:
            img = load_img(filename, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            feature = model.predict(img_array)
            features[img_name.split('.')[0]] = feature  # Ensuring the feature is flattened
        except Exception as e:
            print(f"Failed to process {img_name}: {e}")
    return features


def define_model(vocab_size, max_length, use_cudnn=True):
    inputs1 = Input(shape=(25088,))  # This should be the flattened shape, ensure it matches extracted features
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256, use_cudnn=use_cudnn)(se2)
    decoder1 = Add()([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


def generate_caption(model, tokenizer, photo, max_length):
    """
    Generate a caption for a given image using the trained model.

    Args:
    - model (Keras Model): The trained deep learning model for image captioning.
    - tokenizer (Tokenizer): The tokenizer used to encode the captions.
    - photo (numpy array): The feature vector extracted from the image using a pre-trained VGG16 model.
    - max_length (int): The maximum length of the captions.

    Returns:
    - str: A caption generated for the given image.

    The function generates a caption for a given image by using the trained model and the tokenizer. It starts with a special token 'startseq' and iteratively generates words until it encounters the special token 'endseq'. The generated words are then joined together to form the final caption.
    """
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]  # remove 'startseq' and 'endseq'
    final = ' '.join(final)
    return final


def evaluate_model(model, descriptions, features, tokenizer, max_length):
    actual, predicted = [], []
    for key, desc_list in descriptions.items():
        feature_key = key.split('.')[0]  # Ensure this matches how features keys are formatted
        if feature_key not in features:
            continue  # Skip if no features are available for this key
        yhat = generate_caption(model, tokenizer, features[feature_key], max_length)
        references = [d.split() for d in desc_list]
        actual.append(references)
        predicted.append(yhat.split())
    bleu = corpus_bleu(actual, predicted)
    return bleu


def create_sequences(tokenizer, max_length, descriptions, features, vocab_size):
    """
    Create input-output pairs from the given descriptions and features.

    Args:
    - tokenizer (Tokenizer): The tokenizer used to encode the captions.
    - max_length (int): The maximum length of the captions.
    - descriptions (dict): A dictionary containing image identifiers as keys and lists of descriptions for each image as values.
    - features (dict): A dictionary containing image identifiers as keys and feature vectors extracted from the images as values.
    - vocab_size (int): The size of the vocabulary for the captions.

    Returns:
    - np.array: A numpy array containing the input sequences (X1).
    - np.array: A numpy array containing the input sequences with padding (X2).
    - np.array: A numpy array containing the output sequences (y).

    The function creates input-output pairs from the given descriptions and features. It walks through each image identifier and description, encodes the sequence, splits one sequence into multiple input-output pairs, pads the input sequences, encodes the output sequences, and stores the input and output sequences in separate numpy arrays.
    """
    X1, X2, y = [], [], []
    for key, desc_list in descriptions.items():
        feature_key = key.split('.')[0]  # Strip the extension if it exists
        if feature_key not in features:
            continue  # Skip descriptions for which no features are available
        for desc in desc_list:
            seq = tokenizer.texts_to_sequences([desc])[0]
            for i in range(1, len(seq)):
                in_seq, out_seq = seq[:i], seq[i]
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                X1.append(features[feature_key][0])
                X2.append(in_seq)
                y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)


def filter_descriptions(descriptions, directory):
    """Filter descriptions to include only those for which an image file exists."""
    valid_ids = {img_name.split('.')[0] for img_name in os.listdir(directory) if
                 img_name.lower().endswith(('.png', '.jpg', '.jpeg'))}
    filtered_descriptions = {img_id: desc for img_id, desc in descriptions.items() if img_id in valid_ids}
    return descriptions


def main():
    descriptions = load_descriptions(caption_path)
    descriptions = filter_descriptions(descriptions, images_path)
    features = extract_features(images_path, max_images=100)
    print("Number of descriptions loaded:", len(descriptions))
    print("Number of features extracted:", len(features))
    print("Feature shape check:", next(iter(features.values())).shape)
    print("Images directory exists:", os.path.exists(images_path))
    print("List of files in the directory:", os.listdir(images_path)[:10])  # print the first 10 files to verify

    def test_image_loading(image_path):
        try:
            img = load_img(image_path, target_size=(224, 224))
            img_array = img_to_array(img)
            print("Image loaded successfully")
        except Exception as e:
            print("Error loading image:", e)

    # Replace 'sample.jpg' with an actual image file name from the directory
    test_image_loading(os.path.join(images_path, '1000092795.jpg'))

    # Prepare tokenizer
    tokenizer = Tokenizer()
    all_captions = [desc for captions in descriptions.values() for desc in captions]
    tokenizer.fit_on_texts(all_captions)
    vocab_size = len(tokenizer.word_index) + 1
    max_length = max(len(tokenizer.texts_to_sequences([d])[0]) for d in all_captions)

    # Prepare data
    X1, X2, y = create_sequences(tokenizer, max_length, descriptions, features, vocab_size)
    print("Data prepared for training: X1 =", len(X1), "X2 =", len(X2), "y =", len(y))

    # Split the data into training and validation sets
    X1_train, X1_val, X2_train, X2_val, y_train, y_val = train_test_split(X1, X2, y, test_size=0.2, random_state=42)
    print("Shape of X1_train:", X1_train.shape)
    print("Shape of X1_val:", X1_val.shape)

    # Define the model
    model = define_model(vocab_size, max_length, use_cudnn=False)

    # Fit the model
    model.fit([X1_train, X2_train], y_train, epochs=3, batch_size=2048, validation_data=([X1_val, X2_val], y_val),
              verbose=2)

    # Save the model
    model.save(model_path)

    # Load the model
    # model = load_model(model_path)

    # Evaluate the model
    print("Evaluating model...")
    #bleu_score = evaluate_model(model, descriptions, features, tokenizer, max_length)
    #print("BLEU score on the dataset:", bleu_score)

    # Generate caption for a new image
    new_image_path = 'flickr30k_images'
    new_features = extract_features(new_image_path, max_images=1)
    new_image_id = list(new_features.keys())[0]
    caption = generate_caption(model, tokenizer, new_features[new_image_id], max_length)
    print("Generated Caption:", caption)


if __name__ == '__main__':
    main()
