import cv2
import dlib
import time

def convert_and_trim_bounding_box(image, rectangle):
    startX = rectangle.left()
    startY = rectangle.top()
    endX = rectangle.right()
    endY = rectangle.bottom()

    startX = max(startX, 0)
    startY = max(startY, 0)
    endX = min(endX, image.shape[1])
    endY = min(endY, image.shape[0])

    w = endX - startX
    h = endY - startY

    return (startX, startY, w, h)

def harr_detector(image):
    haar_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    timer_start = time.time()
    rects = haar_detector.detectMultiScale(gray, scaleFactor=1.05,
                                        minNeighbors=5, minSize=(30, 30),
                                        flags=cv2.CASCADE_SCALE_IMAGE)
    timer_end = time.time()
    img_haar = image.copy()
    for (x, y, w, h) in rects:
        cv2.rectangle(img_haar, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    return img_haar, rects, timer_end - timer_start

def hog_detector(image):
    hog_detector = dlib.get_frontal_face_detector()
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    timer_start = time.time()
    hog_rects = hog_detector(rgb, 1)
    timer_end = time.time()
    rects = [convert_and_trim_bounding_box(image, rect) for rect in hog_rects]

    img_hog = image.copy()
    for (x, y, w, h) in rects:
        cv2.rectangle(img_hog, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return img_hog, rects, timer_end - timer_start

def cnn_detector(image):
    cnn_detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    timer_start = time.time()
    rects = cnn_detector(rgb, 1)
    timer_end = time.time()
    rects = [convert_and_trim_bounding_box(image, r.rect) for r in rects]

    img_cnn = image.copy()
    for (x, y, w, h) in rects:
        cv2.rectangle(img_cnn, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return img_cnn, rects, timer_end - timer_start


def print_info(rects, duration):
    print(f"Number of detections: {len(rects)}")
    print(f"Duration: {duration} [s]")   

def show_results(image, image_name):
    cv2.imshow(image_name, image)

PATH = "./out/"
def save_results(image, image_name):
    cv2.imwrite(f"{PATH}{image_name}.png",image)

def ground_read(path):
    GROUND = {}
    with open(path) as f:
        for _ in range(5): f.readline() # skip begining comments in file
        while True:
            line = f.readline()
            if not line:
                break
            file_name = line.split('/')[1]
            file_name = file_name.strip()
            numb_rectangles = f.readline()

            rectangles = [list(map(int, f.readline().split()[:4])) for _ in range(int(numb_rectangles))]                

            GROUND[file_name] = rectangles
    
    return GROUND

def ground_detect(image, image_name, ground):
    bound_box = ground[image_name]
    
    image_copy = image.copy()

    for (x, y, w, h) in bound_box:
        cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return image_copy, bound_box, 0