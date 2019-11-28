import glob as glob
import cv2
import matplotlib.pyplot as plt
from scipy import math
import numpy as np

def preprocess(img_source):
    gray_img = cv2.cvtColor(img_source, cv2.COLOR_BGR2GRAY)
    (_, thresh) = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary_img = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return binary_img, contours


# detect board
def new_detect_boards(contours, too_small=0.00):
    max_w = 0
    max_h = 0
    boards_coords = []
    for c in contours:
        xarr = [point[0][0] for point in c]
        yarr = [point[0][1] for point in c]

        w = max(xarr) - min(xarr)
        h = max(yarr) - min(yarr)

        cx = sum(xarr) / len(c)
        cy = sum(yarr) / len(c)

        changes_done = False
        if w < max_w * too_small or h < max_h * too_small:
            continue
        for i in range(0, len(boards_coords)):
            # đã nằm trong vật thể được lưu trữ trong boards_coords
            bound_left = cx >= boards_coords[i][0] - boards_coords[i][2] / 2
            bound_right = cx <= boards_coords[i][0] + boards_coords[i][2] / 2
            bound_up = cy >= boards_coords[i][1] - boards_coords[i][3] / 2
            bound_down = cy <= boards_coords[i][1] + boards_coords[i][3] / 2
            if bound_left and bound_right and bound_up and bound_down:
                boards_coords[i][4] += 1
                changes_done = True
                break
            # bao quanh vật thể được lưu trữ trong boards_coords (swap)
            bound_left = boards_coords[i][0] >= cx - w / 2
            bound_right = boards_coords[i][0] <= cx + w / 2
            bound_up = boards_coords[i][1] >= cy - w / 2
            bound_down = boards_coords[i][1] <= cy + w / 2
            if bound_left and bound_right and bound_up and bound_down:
                boards_coords[i][0] = cx
                boards_coords[i][1] = cy
                boards_coords[i][2] = w
                boards_coords[i][3] = h
                boards_coords[i][4] += 1
                changes_done = True
                break
        if not changes_done:
            boards_coords.append([cx, cy, w, h, 0])
        if w > max_w:
            max_w = w
        if h > max_h:
            max_h = h
    res_boards_coords = []
    for b in boards_coords:
        if b[2] < max_w * too_small or b[3] < max_h * too_small:
            continue
        res_boards_coords.append(b)
    return res_boards_coords


def print_board_stage(board_stage):
    for i in range(0, len(board_stage)):
        print(board_stage[i][0] + " " + board_stage[i][1] + " " + board_stage[i][2])


def get_status(board_stage):
    isEnd = True

    # check 2 đường chéo
    if (board_stage[1][1] != "_"):
        if ((board_stage[0][0] == board_stage[1][1]) and (board_stage[1][1] == board_stage[2][2])):
            return board_stage[1][1] + " thắng!"
        if ((board_stage[0][2] == board_stage[1][1]) and (board_stage[1][1] == board_stage[2][0])):
            return board_stage[1][1] + " thắng!"

            # check các cột ngang
    for i in range(0, len(board_stage)):
        # check ending
        if (isEnd):
            for j in range(0, len(board_stage)):
                if (board_stage[i][j] == '_'):
                    isEnd = False
                    break
        if ((board_stage[i][0] == board_stage[i][1]) and (board_stage[i][0] == board_stage[i][2]) and (
                board_stage[i][0] != "_")):
            return board_stage[i][0] + " thắng!"
        if ((board_stage[0][i] == board_stage[1][i]) and (board_stage[0][i] == board_stage[2][i]) and (
                board_stage[0][i] != "_")):
            return board_stage[0][i] + " thắng!"
    if (isEnd):
        return "Hòa"
    return "Ván chưa kết thúc"


fig = plt.figure()
fig.subplots_adjust(left=0.0, top=1.0, right=1.0, bottom=0.0, wspace=0.0, hspace=0.0)
index = 0  # index of current
offset = 0  # offset for an image ID
allfiles = glob.iglob('input/*.jpg')
cap = cv2.VideoCapture(0)
old_board = [['_', '_', '_'], ['_', '_', '_'], ['_', '_', '_']]
while (True):
    # reads frames from a camera
    ret, image = cap.read()
    index += 1  # iteration counter

    # detect vunfg
    binary_img, contours = preprocess(image)
    boards_data = new_detect_boards(contours, too_small=0.25)

    # process contours
    processed_contours = [[] for _ in range(0, len(boards_data) + 1)]  # [[cx, cy, w, h, contours], [], [], ...]
    for c in contours:
        # calculate basic properties and board, object is located in
        xarr = [point[0][0] for point in c]
        yarr = [point[0][1] for point in c]

        center_x = sum(xarr) / len(c)
        center_y = sum(yarr) / len(c)
        obj_w = max(xarr) - min(xarr)
        obj_h = max(yarr) - min(yarr)

        board_id = len(boards_data)  # assume that object does not belong to any board
        for i in range(0, len(boards_data)):
            b = boards_data
            b_left = center_x >= b[i][0] - b[i][2] / 2
            b_right = center_x <= b[i][0] + b[i][2] / 2
            b_up = center_y >= b[i][1] - b[i][3] / 2
            b_down = center_y <= b[i][1] + b[i][3] / 2
            if b_left and b_right and b_up and b_down:
                board_id = i  # unless we find a match
                break
        processed_contours[board_id].append([center_x, center_y, obj_w, obj_h, c])

    # detect objects
    detected_objects = [[] for _ in range(0, len(boards_data))]
    for board_pc in range(0, len(processed_contours) - 1):
        for pc in processed_contours[board_pc]:
            # give up on objects too small for its board
            factor_edge = 0.05
            max_size = 0.28
            if pc[2] < boards_data[board_pc][2] * factor_edge or pc[3] < boards_data[board_pc][3] * factor_edge:
                continue
            if pc[2] > boards_data[board_pc][2] * max_size or pc[3] > boards_data[board_pc][3] * max_size:
                continue
            # calculate angle intervals
            angle_probes = 45
            angle_segment = [0 for _ in range(0, angle_probes)]
            for point in pc[4]:
                # if possible...
                # x - center_x
                dx = point[0][0] - pc[0]
                if dx == 0:
                    continue
                dy = point[0][1] - pc[1]
                # ..calculate atan and convert into degree measure
                rad_angle = math.atan(dy / dx) * 180 / math.pi
                if dx < 0 > dy:
                    rad_angle = 270 - rad_angle
                if dx < 0 <= dy:
                    rad_angle = 270 - rad_angle
                if dx >= 0 > dy:
                    rad_angle = 90 - rad_angle
                if dx >= 0 <= dy:
                    rad_angle = 90 - rad_angle
                angle_segment[int(rad_angle / (360 / angle_probes))] += 1

            object_avg = sum(angle_segment) / len(angle_segment)
            variance = 0
            for i in angle_segment:
                variance += (i - object_avg) ** 2
            variance = float(variance) / (len(angle_segment))
            # each object gets new field - variance
            pc.append(variance)
            # and the entire object is put into new list
            detected_objects[board_pc].append(pc)

    # begin board calculations
    board = detected_objects[0]
    # plot and decide board state
    if (len(board) > 0 and boards_data[0][2] / boards_data[0][3] < 1.3):
        not_dynamic_anymore_threshold = sum([c[5] for c in board]) / len(board) - 1
        board_state = [['_', '_', '_'], ['_', '_', '_'], ['_', '_', '_']]
        for c in board:
            xarr = [point[0][0] for point in c[4]]
            yarr = [point[0][1] for point in c[4]]
            if c[5] >= not_dynamic_anymore_threshold:
                mark = 'X'
                cv2.drawContours(image, [c[4]], 0, (255, 0, 0), 3)
            else:
                mark = 'O'
                cv2.drawContours(image, [c[4]], 0, (0, 0, 255), 3)
            # read the logical position form the physical position in an image
            x_diff = boards_data[0][0] - c[0]
            y_diff = boards_data[0][1] - c[1]
            w_perc = math.fabs(x_diff) / boards_data[0][2]
            h_perc = math.fabs(y_diff) / boards_data[0][3]
            row_detected = 0
            col_detected = 0
            # object is in the center if its not further than 10% of the board size
            if w_perc < 0.10:
                col_detected = 1
            else:
                if x_diff < 0:
                    col_detected = 2
                else:
                    col_detected = 0
            if h_perc < 0.10:
                row_detected = 1
            else:
                if y_diff < 0:
                    row_detected = 2
                else:
                    row_detected = 0
            board_state[row_detected][col_detected] = mark
        if(board_state != old_board):
            old_board = board_state
            print_board_stage(board_state)
            print("Trạng thái: " + get_status(board_state) + "\n")
    # Display an image in a window
    cv2.imshow('img', image)

    # Wait for Q key to stop
    k = cv2.waitKey(30)
    if (k == 113):
        break
# Close the window
cap.release()
# De-allocate any associated memory usage
cv2.destroyAllWindows()
