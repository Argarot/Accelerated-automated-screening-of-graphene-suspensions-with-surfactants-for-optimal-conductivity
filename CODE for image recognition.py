# Import all the necessary packages
import numpy as np
import os
import cv2
import pandas as pd
import PySimpleGUI as sg

# Directory and Filename
MAIN_DIR = r"C:\Users\Argarot\images"
DANIIL_DIR = MAIN_DIR + "\calibration\\"
filename = 'IMG_20210727_181600.jpg'

# PySimpleGUI
sg.theme("LightGreen")

layout = [
    # 1. Threshold for Segmentation
    [
        sg.Text("Threshold for Segmentation:", size=(25, 1), justification="left"),
        sg.Slider(
            (0, 100),
            0,
            1,
            orientation="h",
            size=(40, 15),
            key="-SEGMENT THRESH SLIDER-",
        ),
    ],

    # 2. Morphological Transformations - Noise Reduction
    [
        sg.Text("Iterations for Opening:", size=(25, 1), justification="left"),
        sg.Slider(
            (0, 5),
            2,
            1,
            orientation="h",
            size=(40, 15),
            key="-OPEN ITER SLIDER-",
        ),
    ],
    [
        sg.Text("Iterations for Closing:", size=(25, 1), justification="left"),
        sg.Slider(
            (0, 5),
            2,
            1,
            orientation="h",
            size=(40, 15),
            key="-CLOSE ITER SLIDER-",
        ),
    ],

    # 3. Morphological Transformations - Dilation
    [
        sg.Text("Iterations for Dilation:", size=(25, 1), justification="left"),
        sg.Slider(
            (0, 5),
            2,
            1,
            orientation="h",
            size=(40, 15),
            key="-DILATE ITER SLIDER-",
        ),
    ],

    # 4. Options: DistanceTransform Mask Size
    [
        sg.Text("MaskSize for Distance Transform:", size=(25, 1), justification="left"),
        sg.Radio('3x3 MaskSize', 'Radio1', size=(10, 1), key="-3x3 MASK SIZE-", default=True),
        sg.Radio('5x5 MaskSize', 'Radio1', size=(10, 1), key="-5x5 MASK SIZE-"),
        sg.Radio('0x0 MaskSize', 'Radio1', size=(10, 1), key="-0x0 MASK SIZE-"),
    ],

    [sg.Button("OK", size=(10, 1))],
]

window = sg.Window("OpenCV Watershed Algorithm GUI", layout, location=(600, 400))

# __main__()
while (1):
    event, values = window.read(timeout=20)
    if event == "OK" or event == sg.WIN_CLOSED or event is None:
        break

    # Load the image
    img = cv2.imread(DANIIL_DIR + filename)
    img = cv2.resize(img, (600, 768))  
    img_copy = img.copy()
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # values from PysimpleGUI
    threshold = int(values["-SEGMENT THRESH SLIDER-"])

    opening_iter = int(values["-OPEN ITER SLIDER-"])

    closing_iter = int(values["-CLOSE ITER SLIDER-"])

    dilate_iter = int(values["-DILATE ITER SLIDER-"])

    if values["-3x3 MASK SIZE-"]:
        maskSize = 3
    elif values["-5x5 MASK SIZE-"]:
        maskSize = 5
    elif values["-0x0 MASK SIZE-"]:
        maskSize = 0

    # Image Thresholding (Inverse Binary + Otsu Method)
    ret, thresh = cv2.threshold(src=img_gray, thresh=127, maxval=255, type=cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    # Morphological Transformations / Noise Reduction
    kernel = np.ones((3,3),np.uint8)
    thresh = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel,iterations=opening_iter)     # Opening to reduce background noise
    thresh = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel,iterations=closing_iter)    # Closing to reduce foreground noise
    
    # Dilation: to find sure background area
    kernel = np.ones((3,3),np.uint8)
    sure_bg = cv2.dilate(thresh,kernel,iterations=dilate_iter) # Adjust until desired result is achieved

    # Calculate distance transformation
    dist_transform = cv2.distanceTransform(src=sure_bg,distanceType=cv2.DIST_L2,maskSize=maskSize)                                    # Adjust until desired result is achieved
    ret, sure_fg = cv2.threshold(src=dist_transform,thresh=(threshold/100 * dist_transform.max()),maxval=255,type=cv2.THRESH_BINARY)  # Adjust to get desired separation

    # Find unknown regions
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    # Markers Labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers+1             # Add one to all labels so that sure background is not 0, but 1
    markers[unknown==255] = 0       # Mark the region of unknown with zero

    # Watershed Algorithm to find markers
    markers = cv2.watershed(img,markers)
    markers_copy = markers.copy()
    markers_black = markers.astype(np.uint8)

    # Finding Contours on Markers
    contours, hierarchy = cv2.findContours(markers_copy, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # Find coordinates of the centroids, area of the droplets
    cx = []
    cy = []
    area_cv = []

    for i,c in enumerate(contours):
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        else:
            cx.append(int(M["m10"] / M["m00"])) # x-coordinates of centroids
            cy.append(int(M["m01"] / M["m00"])) # y-coordinates of centroids
            area_cv.append(cv2.contourArea(c))

    data = pd.DataFrame({'x': cx, 'y': cy, 'area': area_cv})
    data_contours = pd.DataFrame(list(contours))
    data_contours.rename(columns={0: 'contours'},inplace=True)
    data = data.merge(data_contours, left_index=True, right_index=True)
    # the following are the columns in 'data' df: x, y, area, contours

    data = data[(data['area'] < 15000.0) & (data['area'] > 100.0)]     # Filter contours with 100.0 < area < 1500.0  (adjustable)
    data.drop_duplicates(subset=['x','y'],inplace=True)               # Remove duplicate centroids

    # Sorting left-to-right, top-to-bottom
    data.sort_values(by=['y'], inplace = True)
    data.reset_index(inplace = True, drop = True)

        # Initialize the column 'row'
    data['row'] = np.ones(len(data), dtype='int64')    

        # Assign row number
    row_num = 1
    for i in np.arange(1,len(data)): 
        if (abs(data.loc[i,'y'] - data.loc[i-1,'y']) > 30):             # If diff in y-coordinates > 30: next row (adjustable) 
            row_num += 1
            data.loc[i,'row'] = row_num
        else:
            data.loc[i,'row'] = row_num

        # Sort by row first, then sort by x-axis
    data.sort_values(by=['row','x'], ascending=[True,True], inplace=True) 
    data.reset_index(inplace = True, drop = True)

    for i in np.arange(len(data)):        
        if i == 0:
            continue
        elif (abs(data.loc[i,'x'] - data.loc[i-1,'x']) < 5):           # If diff in x-coordinates < 5 (overlap of circles): remove (adjustable) 
            data.loc[i,'row'] = 999

    data = data[data['row'] != 999]
    data.reset_index(inplace = True, drop = True)

    # Sorted contours
    new_contours = list(data['contours'])

    # Draw contours and annotate on the image
    for i in list(data.index.values):
        cx = data['x'][i].astype('int64')
        cy = data['y'][i].astype('int64')
        cv2.drawContours(img, new_contours, -1, (0,0,255), 1) # red contours
        cv2.circle(img, (cx,cy), 2, (255,0,0), 1)             # blue circle of centroid
        cv2.putText(img, "{}".format(i),                      # black text to annotate
                (cx - 20, cy - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)

    cv2.imshow("Contours",img)
    cv2.imshow("Watershed Markers",markers_black)     
    k = cv2.waitKey(1) & 0xFF
    if k == 27:         # wait for ESC key to exit
        break
cv2.destroyAllWindows()

cv2.imshow("Final Result",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(os.path.join(DANIIL_DIR,"{}_contours.jpg".format(filename[:-4])),img)

print('\nNumber of contours: {}'.format(len(data)))

data.to_csv(os.path.join(DANIIL_DIR,"{}_data.csv".format(filename[:-4])))
print("{}_data.csv created.\n".format(filename[:-4]))

