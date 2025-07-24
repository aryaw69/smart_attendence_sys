import cv2
import face_recognition
import os 
import pickle 

def load_images_from_folder (folder_path):
    images = []
    names = []

    for file_name in os.listdir(folder_path):
        full_path = os.path.join(folder_path, file_name)
        img = cv2.imread(full_path)
        if img is not None:
            images.append(img)
            names.append(os.path.splitext(file_name)[0])
    
    return images, names

def encode_faces(images):
    encoded =[]
    for img in images:
        #openCV loads images in BGR format by default
        #and face_recognition library expects RGB, so we convert first 
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        enc = face_recognition.face_encodings(rgb)[0]
        #[0] - assumes each image has exactly one face
        encoded.append(enc)
    return encoded

if __name__ == "__main__":
    folder = "images"
    imgs, names = load_images_from_folder(folder)
    encodings = encode_faces(imgs)

    with open("encoding.pkl", "wb") as f:
        pickle.dump((encodings, names), f)

    print("Encodings saved to encodings.pkl")