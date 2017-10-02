import cv2
import numpy as np
from matplotlib import pyplot as plt

A4WIDTH_MM = 210
DocSize = (176, 264)
template = cv2.imread('A_marker.png',0)
marker_shape = template.shape[::-1]

def detect_marker(Image):
    img_gray = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
    markerposarray = []
    for i in range(3):
        (minval, maxval, minloc, maxloc) = cv2.minMaxLoc(res)
        a = maxloc[0] + int(marker_shape[0]/2)
        b = maxloc[1] + int(marker_shape[1]/2)
        markerposarray.append(tuple([a,b]))
        cv2.circle(res, maxloc,20, (0.0), -1)
        print(a,b)
    return markerposarray

def sort_center_point(List):
    a = []
    temp = []
    result = []
    for i in range(3):
        a.append(sum(List[i]))

    for i in range(3):
        temp.append(a.index(max(a)))
        a.remove(max(a))

    for i in temp:
        result.append(List[i])

    return make_4th_point(result)

def make_4th_point(List):
    fourth_point = (List[1][0], List[0][1])
    sorted_result = []
    sorted_result.append(List[2])
    sorted_result.append(List[1])
    sorted_result.append(fourth_point)
    sorted_result.append(List[0])

    return sorted_result

def transform(image, points, dpmm):
    print(points)
    docpxls = (int(DocSize[0] * dpmm),int(DocSize[1]*dpmm))
    docrect = np.array([(0,0), (docpxls[0], 0), (docpxls[0], docpxls[1]), (0, docpxls[1])],'float32')
    transmat = cv2.getPerspectiveTransform(np.array(points, 'float32'), docrect)
    return cv2.warpPerspective(image, transmat, docpxls)

def main():
    img_rgb = cv2.imread('OCR_test.png')
    center_point_list =  detect_marker(img_rgb)
    center_point_list = sort_center_point(center_point_list)
    dpmm = min(img_rgb.shape[0:2]) / A4WIDTH_MM
    result = transform(img_rgb, center_point_list, dpmm)

    cv2.imwrite('corrected.jpg', result)
if __name__ == '__main__':
    main()
