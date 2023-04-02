import cv2
import mediapipe as mp
import imutils
import numpy as np
from time import time
import torch

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2)
mpDraw = mp.solutions.drawing_utils


# Processing the input image
def process_image(img):
    # Converting the input to grayscale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(gray_image)

    # Returning the detected hands to calling function
    return results


# Drawing landmark connections
def draw_hand_connections(img, results):
    if results.multi_hand_landmarks:
        print(len(results.multi_hand_landmarks))
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape

                # Finding the coordinates of each landmark
                cx, cy = int(lm.x * w), int(lm.y * h)

                # Printing each landmark ID and coordinates
                # on the terminal
                # print(id, cx, cy)

                # Creating a circle around each landmark
                cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
                # Drawing the landmark connections
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
        return img

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

            cv2.rectangle(img, (left - 10, top + 10), (right + 10, bottom - 10), (255, 0, 0), thickness=line_width, lineType=cv2.LINE_AA)
            tf = max(line_width - 1, 1)  # font thickness
            w, h = cv2.getTextSize(f'Hand {score}', 0, fontScale=line_width / 3, thickness=tf)[0]  # text width, height
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


def get_countours(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)[1]
    result = gray_img.copy()
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    for cntr in contours:
        x, y, w, h = cv2.boundingRect(cntr)
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
        print("x,y,w,h:", x, y, w, h)


def is_hand_detected(results):
    if results.multi_hand_landmarks:
        return True
    else:
        return False


def main():
    # Replace 0 with the video path to use a
    # pre-recorded video
    cap = cv2.VideoCapture(0)

    while True:
        # Taking the input
        success, image = cap.read()
        image = imutils.resize(image, width=500, height=500)
        results = process_image(image)
        #
        # for r in results.multi_handedness:
        #     for c in r.classification:
        #         print(c.score)



        # draw_hand_connections(image, results)
        draw_bounding_box(image, results)
        # get_countours(image)

        # Displaying the output
        cv2.imshow("Hand Detection", image)

        # Program terminates when q key is pressed
        if cv2.waitKey(1) == ord('q'):
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
