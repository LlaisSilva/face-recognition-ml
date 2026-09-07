import cv2

def draw_face_result(image, face, name):
    """Draw the face bounding box, student name """

    bbox= face.bbox.astype(int)

    x1, y1,x2,y2 = bbox

    if name.lower() =="unknown":
        color = (0,0,255)
    else:
        color=(0,255,0)

    cv2.rectangle(image, (x1,y1),(x2,y2), color, 2)

    label = f"{name}"

    (text_with, text_height), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6,2)

    cv2.rectangle(
        image,
        (x1,y1- text_height-baseline-10),
        (x1+text_with+10, y1),
        color,
        -1
    )

    cv2.putText(image,label, (x1+5,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

    return image