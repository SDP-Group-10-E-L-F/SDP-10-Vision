import numpy as np
from PIL import Image
import tensorflow as tf
import mediapipe as mp
import imutils
import cv2
from arduino import BoardController

class_names = ['Empty','Longsleeve', 'Pants', 'T-Shirt']
# class_names = ['Longsleeve', 'Pants', 'T-Shirt']

batch_size = 32
img_height = 400
img_width = 400

new_model = tf.keras.models.load_model('ImgClassifier_Saved_Model/exp2/model2.h5')

def predict(model, frame):
    predictions = model.predict(frame)
    score = tf.nn.softmax(predictions[0])
    print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
    )
    return score

"""
Hand Detection Part
"""
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=4)
mpDraw = mp.solutions.drawing_utils

# Processing the input image
def process_image(img):
    # Converting the input to grayscale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(gray_image)

    # Returning the detected hands to calling function
    return results

def draw_bounding_box(img, results):
    """
    Args:
        img: <class 'numpy.ndarray'>
        results:

    Returns:
    """
    if results.multi_hand_landmarks:
        for hand_landmark, hand_classification in zip(results.multi_hand_landmarks, results.multi_handedness):
            img_height, img_width, _ = img.shape
            x = [int(landmark.x * img_width) for landmark in hand_landmark.landmark]
            y = [int(landmark.y * img_height) for landmark in hand_landmark.landmark]
            score = np.mean([float(classification.score) for classification in hand_classification.classification])
            score = "{:.2f}".format(round(score, 2))

            left = np.min(x)
            right = np.max(x)
            bottom = np.min(y)
            top = np.max(y)

            thick = int((img_height + img_width) // 400)

            line_width = max(round(sum(img.shape) / 2 * 0.003), 2)  # line width

            # Bouding box visualization
            cv2.rectangle(img,
                          (left - 10, top + 10),    # Top left coordinates
                          (right + 10, bottom - 10),    # Bottom right coordinates
                          (255, 0, 0),  # Color of the detection box
                          thickness=line_width,
                          lineType=cv2.LINE_AA)

            # Text info display on bounding box
            tf = max(line_width - 1, 1)  # font thickness

            # text width, height
            w, h = cv2.getTextSize(f'Hand {score}', 0, fontScale=line_width / 3, thickness=tf)[0]
            outside = (left - 10) - h >= 3
            p2 = (left - 10) + w, (top + 10) - h - 3 if outside else (top + 10) + h + 3
            cv2.rectangle(img, (left - 10, top + 10), p2, (255, 0, 0), -1, cv2.LINE_AA)  # filled
            cv2.putText(img,
                        f'Hand {score}', ((left - 10), (top + 10) - 2 if outside else (top + 10) + h + 2),
                        0,
                        line_width / 3,
                        (255, 255, 255),
                        thickness=tf,
                        lineType=cv2.LINE_AA)

def is_hand_detected(results):
    if results.multi_hand_landmarks and results.multi_handedness:
        # print("Hands Detected! Stop Folding")
        return True
    else:
        # print("Keep Folding")
        return False

def return_clothe_type(frame):
    hand_frame_arr = tf.expand_dims(tf.keras.utils.img_to_array(frame), 0)
    predictions = new_model.predict(hand_frame_arr)
    score = tf.nn.softmax(predictions[0])
    cloth_type = class_names[np.argmax(score)]
    return cloth_type, score


def arr2img(frame, width=400, height=400):
    """
    Args:
        frame: numpy.ndarray
    """
    frame = cv2.resize(frame, (width, height))
    img = Image.fromarray(frame, 'RGB')
    img.save('my.png')
    img.show()


if __name__ == '__main__':
    # controller = BoardController()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('cannot open the camera')
        exit()
    while cap.isOpened():
        ret, frame, = cap.read()

        # grab the dimensions of the image and calculate the center of the
        # image
        # (h, w) = frame.shape[:2]
        # (cX, cY) = (w // 2, h // 2)
        # # rotate our image by 45 degrees around the center of the image
        # M = cv2.getRotationMatrix2D((cX, cY), 270, 1.0)
        # frame = cv2.warpAffine(frame, M, (w, h))

        cloth_frame = frame
        hand_frame = frame
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # For hands detection
        hand_frame = imutils.resize(hand_frame, width=img_width, height=img_height)
        hand_results = process_image(hand_frame)
        draw_bounding_box(hand_frame, hand_results)

        # For clothes detection
        cloth_frame = cv2.resize(cloth_frame, (img_width, img_height))
        cloth_type, score = return_clothe_type(cloth_frame)

        # if not is_hand_detected(hand_results):
        #     if cloth_type == "T-shirt":
        #         controller.short_sleeve()
        #     elif cloth_type == "Longsleeve":
        #         controller.long_sleeve()
        #     elif cloth_type == "Pants":
        #         controller.trousers()
        # else:
        #     print("Hands Detected! Stop Folding")


        cv2.putText(cloth_frame, f'{cloth_type} : {round(np.max(score), 3)}',
                    (0, img_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    1,
                    cv2.LINE_AA)

        parallel = np.concatenate((cloth_frame, hand_frame), axis=0)

        cv2.imshow('Classification', parallel)

        # Program terminates when q key is pressed
        if cv2.waitKey(1) == ord('q'):
            cap.release()
            cv2.destroyAllWindows()

