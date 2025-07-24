import cv2
import face_recognition 
import numpy as np
import os
import pickle

def load_encodings(path='encodings.pkl'):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data['encodings'], data['names']

def recognize_faces(frames, known_encodings, known_names):
    #we are using reduce the original_frame size to small_frame (it reduces by 25%, both width and height). as face recognition doent need hi-res input 
    small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    #detect face and location in current frame 
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]
            else:
                name = "Unknown"
        
        else:
            name = "Unknown"
        
        names.append(name)

    return face_locations, names