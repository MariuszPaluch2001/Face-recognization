from functions import *

def check_correctness_alg(correct_rects, result_rects):
    corr_predict = 0
    miss_predict = 0
    for rect in result_rects:
        rect_mid_x, rect_mid_y = rect[0] + rect[2] / 2, rect[1] + rect[3] / 2
        is_finded = False
        for (corr_startX, corr_startY, corr_w, corr_h) in correct_rects:
            if rect_mid_x > corr_startX and rect_mid_x < corr_startX + corr_w and \
                rect_mid_y > corr_startY and rect_mid_y < corr_startY + corr_h:
                corr_predict += 1
                is_finded = True
                break
        if not is_finded:
            miss_predict += 1

    return corr_predict, miss_predict, len(correct_rects) - corr_predict

def print_correcntess_info(correct_rects, result_rects):
    row_correct_predictions = f"{'Wykryte twarze' :<20}|"
    row_missed_predictions =  f"{'Pomylone twarze' :<20}|"
    row_not_finded_faces = f"{'Niewykryte twarze' :<20}|"

    for rects in result_rects:
        corr_predict, miss_predict, not_finded = check_correctness_alg(correct_rects, rects)
        row_correct_predictions += f"{corr_predict :8d}|"
        row_missed_predictions += f"{miss_predict :8d}|"
        row_not_finded_faces += f"{not_finded :8d}|"

    print(row_correct_predictions)
    print(row_missed_predictions)
    print(row_not_finded_faces)

def print_numb_bounding_boxs(rects):
    row = f"{'Liczba detekcji' :<20}|"
    for rect in rects:
        row += f"{len(rect) :8d}|"
    
    print(row)


def print_duration(times):
    row = f"{'Czas reakcji' :<20}|"
    for time in times:

        row += f"{time :8.2f}|"

    print(row)

if __name__ == "__main__":
    testing_images = ["2_Demonstration_Demonstration_Or_Protest_2_1", 
                      "2_Demonstration_Demonstration_Or_Protest_2_15", 
                      "38_Tennis_Tennis_38_452", 
                      "33_Running_Running_33_266",
                      "16_Award_Ceremony_Awards_Ceremony_16_25"
                      ]
    ground = ground_read("./obrazy/ground_truth_bbx.txt")

    for file_name in testing_images:
        image = cv2.imread(f"obrazy/{file_name}.jpg")
        image_ground ,curr_rects ,duration = ground_detect(image, f"{file_name}.jpg", ground)
        save_results(image_ground, f"ground_truth_{file_name}")
        print(f"Obraz: {file_name}")
        print(f"Poprawne detekcje: {len(curr_rects)}")
        images = []
        rects_list = []
        times = []
        for func, alg_name in ((harr_detector, "herr"), (hog_detector, "hog"), (cnn_detector, "cnn")):
            image_new, rects, duration = func(image)
            images.append(image_new)
            rects_list.append(rects)
            times.append(duration)
            save_results(image_new, f"{file_name}_{alg_name}")

        print(f"{'' :<20}|{'herr':^8}|{'hog':^8}|{'cnn':^8}|")
        print_numb_bounding_boxs(rects_list)
        print_correcntess_info(curr_rects, rects_list)
        print_duration(times)
        print("-"*48)
