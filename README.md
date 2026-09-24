# Facial-recognition-with-dlib
"""
ফেস আইডেন্টিফিকেশন সিস্টেম (নিবন্ধিত ব্যক্তিদের জন্য)

ইনস্টল:
    pip install opencv-python face_recognition numpy
    (face_recognition এর জন্য cmake ও dlib লাগে)

ফোল্ডার সাজানো:
    known_faces/
        rahim.jpg
        karim.jpg
    people.csv   <-- প্রতিটি ব্যক্তির ডিটেল

people.csv ফরম্যাট (প্রথম লাইন হেডার):
    file,name,id,phone,department
    rahim.jpg,Rahim Uddin,1001,01700000000,Accounts
    karim.jpg,Karim Ali,1002,01800000000,Sales

শুধুমাত্র সম্মতি দেওয়া ব্যক্তিদের ছবি ব্যবহার করবেন।
"""

import csv
import os

import cv2
import face_recognition
import numpy as np

FACES_DIR = "known_faces"
CSV_FILE = "people.csv"
TOLERANCE = 0.5  # কম = কড়া মিল, বেশি = ঢিলা মিল


def load_known_people():
    encodings, details = [], []
    with open(CSV_FILE, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            path = os.path.join(FACES_DIR, row["file"])
            if not os.path.exists(path):
                print(f"ছবি পাওয়া যায়নি: {path}")
                continue
            image = face_recognition.load_image_file(path)
            found = face_recognition.face_encodings(image)
            if not found:
                print(f"ছবিতে মুখ পাওয়া যায়নি: {path}")
                continue
            encodings.append(found[0])
            details.append(row)
    print(f"{len(details)} জন ব্যক্তি লোড হয়েছে")
    return encodings, details


def main():
    known_encodings, known_details = load_known_people()
    cam = cv2.VideoCapture(0)

    while True:
        ok, frame = cam.read()
        if not ok:
            break

        # দ্রুত প্রসেসিংয়ের জন্য ছোট করা
        small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        locations = face_recognition.face_locations(rgb_small)
        encodings = face_recognition.face_encodings(rgb_small, locations)

        for (top, right, bottom, left), enc in zip(locations, encodings):
            top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4
            lines = ["Unknown"]
            color = (0, 0, 255)

            if known_encodings:
                distances = face_recognition.face_distance(known_encodings, enc)
                best = int(np.argmin(distances))
                if distances[best] <= TOLERANCE:
                    d = known_details[best]
                    lines = [
                        d["name"],
                        f"ID: {d['id']}",
                        f"Phone: {d['phone']}",
                        f"Dept: {d['department']}",
                    ]
                    color = (0, 200, 0)

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            y = bottom + 20
            for line in lines:
                cv2.putText(frame, line, (left, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y += 24

        cv2.imshow("Face ID  (q = বন্ধ)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()



In this project, we will explore the use of the dlib library to detect faces in an image and perform facial recognition. Our aim is to build a system that can identify individual faces in an image and match them with known individuals.

Dlib is a machine learning library that provides state-of-the-art algorithms for computer vision tasks such as face detection, facial landmark detection, and facial recognition. It also provides a number of pre-trained models that can be used to quickly and accurately perform these tasks.

Once the faces have been detected, we will use dlib's facial recognition model to compare each face to a database of known individuals. For each face, we will use only a single image data to make the comparison, making our system simple and efficient.

You can find the pre-trained models used for face and facial landmark detections and others [here](https://github.com/davisking/dlib-models), or you can check the files above.

In this example our <u>database consists of only 5 people</u> (1 photo each): **Bill Gates , Elon Musk , Me (Mohamed Amine), Toby Maguire and Jeff Bezos**.
The output results are as follows :

<p align="center">
  <img src="https://github.com/mohamedamine99/Facial-recognition-with-dlib/blob/main/output_fig.png">
</p>
