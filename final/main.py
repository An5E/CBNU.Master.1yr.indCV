import cv2
import numpy as np
import matplotlib.pyplot as plt


def draw_img(img1, img2, img3, img4):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    gray4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
    
    # draw img by plt
    plt.figure(figsize=(10, 10))
    plt.subplot(221)
    plt.imshow(gray1, cmap='gray')
    plt.subplot(222)
    plt.imshow(gray2, cmap='gray')
    plt.subplot(223)
    plt.imshow(gray3, cmap='gray')
    plt.subplot(224)
    plt.imshow(gray4, cmap='gray')
    plt.show()
    
def draw_edge(img1, img2, img3 ,img4):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    gray4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
    
    # extract edge by canny
    edge1 = cv2.Canny(gray1, 100, 200)
    edge2 = cv2.Canny(gray2, 100, 200)
    edge3 = cv2.Canny(gray3, 100, 200)
    edge4 = cv2.Canny(gray4, 100, 200)
    
    # draw img by plt
    plt.figure(figsize=(10, 10))
    plt.subplot(221)
    plt.imshow(edge1, cmap='gray')
    plt.subplot(222)
    plt.imshow(edge2, cmap='gray')
    plt.subplot(223)
    plt.imshow(edge3, cmap='gray')
    plt.subplot(224)
    plt.imshow(edge4, cmap='gray')
    plt.show()
    
def findContours(contours, threshold=100):  # -> cv2.typing.MatLike:
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    for c in contours:
        
        if cv2.contourArea(c) > threshold:
            # rect = cv2.minAreaRect(c)
            # box = cv2.boxPoints(rect)
            
            # return np.int16(box)
            
            
            # ! ~ 
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            
            if len(approx) >= 12:
                return approx

def c0(img1, img2): # compare two circuit board 
    val_blur = 3
    
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # blurring to remove noise
    gray1 = cv2.GaussianBlur(gray1, (val_blur, val_blur), 0)
    gray2 = cv2.GaussianBlur(gray2, (val_blur, val_blur), 0)

    # extract edge by canny
    edge1 = cv2.Canny(gray1, 100, 200)
    edge2 = cv2.Canny(gray2, 100, 200)

    # draw edge1 on img1,2 by red color
    edge1_colored = cv2.cvtColor(edge1, cv2.COLOR_GRAY2BGR)
    edge1_colored[np.where((edge1_colored == [255, 255, 255]).all(axis=2))] = [0, 0, 255]
    img1 = cv2.addWeighted(img1, 1, edge1_colored, 1, 0)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

    edge2_colored = cv2.cvtColor(edge2, cv2.COLOR_GRAY2BGR)
    edge2_colored[np.where((edge2_colored == [255, 255, 255]).all(axis=2))] = [0, 0, 255]
    img2 = cv2.addWeighted(img2, 1, edge2_colored, 1, 0)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # draw img by plt
    plt.figure(figsize=(10, 10))
    plt.subplot(231)
    plt.imshow(img1)
    plt.subplot(232)
    plt.imshow(gray1, cmap='gray')
    plt.subplot(233)
    plt.imshow(edge1, cmap='gray')

    plt.subplot(234)
    plt.imshow(img2)
    plt.subplot(235)
    plt.imshow(gray2, cmap='gray')
    plt.subplot(236)
    plt.imshow(edge2, cmap='gray')
    plt.tight_layout()
    plt.show()

def c0_1(img1, img2): # to compare feature of two image
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # extract feature from image with threshold 
    sift = cv2.SIFT_create(contrastThreshold=0.12, edgeThreshold=2)
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # draw feature on image
    img1 = cv2.drawKeypoints(img1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img2 = cv2.drawKeypoints(img2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


    # compare img1 and img2 feature
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    # draw compare result by plt
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(10, 10))
    plt.subplot(131)
    plt.imshow(img1)
    plt.subplot(132)
    plt.imshow(img2)
    plt.subplot(133)
    plt.imshow(img3)
    plt.show()


    # draw img by plt
    # plt.figure(figsize=(10, 10))
    # plt.subplot(121)
    # plt.imshow(img1)
    # plt.subplot(122)
    # plt.imshow(img2)
    # plt.tight_layout()
    # plt.show()

def c1(img1, img2, img3 ,img4): # draw_edge'
    val_blur = 5
    
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    gray4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
    
    # blurring to remove noise
    gray1 = cv2.GaussianBlur(gray1, (val_blur, val_blur), 0)
    gray2 = cv2.GaussianBlur(gray2, (val_blur, val_blur), 0)
    gray3 = cv2.GaussianBlur(gray3, (val_blur, val_blur), 0)
    gray4 = cv2.GaussianBlur(gray4, (val_blur, val_blur), 0)

    # extract edge by canny
    edge1 = cv2.Canny(gray1, 100, 200)
    edge2 = cv2.Canny(gray2, 100, 200)
    edge3 = cv2.Canny(gray3, 100, 200)
    edge4 = cv2.Canny(gray4, 100, 200)
    
    contours1, _ = cv2.findContours(edge1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(edge2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours3, _ = cv2.findContours(edge3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours4, _ = cv2.findContours(edge4, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)



    pcb_contour = findContours(contours1,10)
    
    if pcb_contour is None:
        print("# cannot find pcb contour")
        return
    
    # draw pcb_contour
    print(pcb_contour)
    edge1 = cv2.drawContours(edge1, [pcb_contour], -1, (255, 0, 0), 2)
    
    # edge1 = cv2.drawContours(edge1, [pcb_contour], -1, (255, 0, 0), 2)

    # draw img by plt
    plt.figure(figsize=(10, 10))
    plt.subplot(221)
    plt.imshow(edge1)
    plt.subplot(222)
    plt.imshow(edge2, cmap='gray')
    plt.subplot(223)
    plt.imshow(edge3, cmap='gray')
    plt.subplot(224)
    plt.imshow(edge4, cmap='gray')
    plt.tight_layout()
    plt.show()
    
def stitch1(showplt: bool = False, saveimg: bool = False) -> cv2.typing.MatLike:
    # read img
    img1 = cv2.imread("./assets/images/stitch/sample-part1.jpg")
    img2 = cv2.imread("./assets/images/stitch/sample-part2.jpg")
    img3 = cv2.imread("./assets/images/stitch/sample-part3.jpg")
    img4 = cv2.imread("./assets/images/stitch/sample-part4.jpg")
    img5 = cv2.imread("./assets/images/stitch/sample-part5.jpg")
    img6 = cv2.imread("./assets/images/stitch/sample-part6.jpg")
    
    # image stitching
    stitcher = cv2.Stitcher_create()
    (status, stitched) = stitcher.stitch([img1, img2, img3, img4, img5, img6])
    
    # stitched = cv2.cvtColor(stitched, cv2.COLOR_BGR2RGB)
    
    # draw img by plt
    plt.figure(figsize=(10, 10))
    # plt.subplot(151)
    # plt.imshow(img1)
    # plt.subplot(152)
    # plt.imshow(img2)
    # plt.subplot(153)
    # plt.imshow(img3)
    # plt.subplot(154)
    # plt.imshow(img4)
    plt.subplot(111)
    plt.imshow(stitched)
    
    plt.tight_layout()
        
    if saveimg:
        plt.axis('off')
        plt.savefig("./assets/images/stitch/pcb-stitching.png", bbox_inches='tight', pad_inches=0.0)
    else:
        plt.title("PCB stitching with 6 parts")
    
    if showplt:
        plt.show()
        
        
    
    return stitched

def feature1(img1):
    # copy img1 to img2
    img2 = img1.copy()
    
    # img1 = cv2.imread("./assets/images/stitch/sample-part1.jpg")
    # img2 = cv2.imread("./assets/images/stitch/sample-part1.jpg")
    
    # extract features from images and matching
    
    sift = cv2.SIFT_create(contrastThreshold=0.15)
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
            
    # draw results
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(10, 10))
    plt.imshow(img3)
    plt.show()

def stroke1(img1):    
    img = img1.copy()
    
    # ! for DEBUG
    # draw rect on img
    pts1 = np.array([[0,0],[100,100],[100,0]],dtype=np.int32)
    pts2 = np.array([[0,0],[200,200],[200,0]],dtype=np.int32)
    cv2.polylines(img,[pts1,pts2], True,(255,0,0),2)
    
    
    # edge detection
    edge1 = cv2.Canny(img1, 100, 100)
    
    # contour detection
    contours, _ = cv2.findContours(edge1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # filter contours by edges
    for contour in contours:
        if cv2.contourArea(contour) < 1:
            cv2.drawContours(edge1, [contour], -1, (0, 0, 0), -1)
    
    
    
    # add draw rectangle function via mouse event callback
    x1, y1 = -1,-1
    clicked = False
    
    def draw_rectangle(event, x, y, flags, param):
        global x1, y1, clicked
        
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked = True
            x1, y1 = x, y
            # cv2.imshow("orig",img)
        elif event == cv2.EVENT_MOUSEMOVE:
            if clicked:
                cv2.rectangle(img, (x1, y1), (x, y), (0, 255, 0), 2)
                # x1,y1 = x,y
        elif event == cv2.EVENT_LBUTTONUP:
            clicked = False
            cv2.rectangle(img, (x1, y1), (x, y), (0, 255, 0), 2)
            
        print(clicked)
    
    cv2.namedWindow("orig")
    cv2.setMouseCallback("orig", draw_rectangle)
    
    while True:
        cv2.imshow("orig",img)
        
        if cv2.waitKey(20) & 0xFF == 27:
            break
        
    cv2.destroyAllWindows()
    
    # draw edge by plt
    # plt.figure(figsize=(10, 10))
    # plt.subplot(121)
    # plt.imshow(edge1)
    # plt.subplot(122)
    # plt.title("test")
    # plt.imshow(img)
    # plt.tight_layout()
    # plt.show()
    
    
    
    # draw edge by plt
    # plt.figure(figsize=(10, 10))
    # plt.subplot(121)
    # plt.imshow(edge1)
    # plt.subplot(122)
    # plt.title("test")
    # plt.imshow(img)
    # plt.tight_layout()
    # plt.show()
    
def dcheck1(img): # PCB 방향 인식
    img_reversed = [cv2.rotate(img, cv2.ROTATE_180), cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)]
    
    # identify image direction by feature matching
    
    # extract features from images and matching
    
    sift = cv2.SIFT_create(contrastThreshold=0.2)
    kp1, des1 = sift.detectAndCompute(img, None)
    
    bf = cv2.BFMatcher()
    
    # kp2, des2 is Array of keypoints and descriptors
    
    
    matches = []
    kp2, des2 = [],[]
    for ir in img_reversed:
        kp, des = sift.detectAndCompute(ir, None)
        match = bf.knnMatch(des1, des, k=2)
        matches.append(match)
        
        kp2.append(kp)
        des2.append(des)
    
        # compare kp1, des1 and kp2, des2 features
        
    good_matches = [[], [], []]
    for i, match in enumerate(matches):
        # good_matches = []
        for m, n in match:
            if m.distance < 0.75 * n.distance:
                good_matches[i].append(m)
    
    
    def chkmtd(_kp2, _gm) -> int:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in _gm ]).reshape(-1,1,2)
        dst_pts = np.float32([ _kp2[m.trainIdx].pt for m in _gm ]).reshape(-1,1,2)
        
        M,inliers = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC)
        
        if M is not None:
            cos_theta = M[0, 0]
            sin_theta = M[0, 1]
            
            angle_rad = np.arctan2(sin_theta, cos_theta)
            angle_deg = np.degrees(angle_rad)
            
            return np.round(angle_deg).astype(int)
    
    img3 = cv2.drawMatches(img, kp1, img_reversed[0], kp2[0], good_matches[0], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # draw results    
    img4 = cv2.drawMatches(img, kp1, img_reversed[1], kp2[1], good_matches[1], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    img5 = cv2.drawMatches(img, kp1, img_reversed[2], kp2[2], good_matches[2], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    print(chkmtd(kp2[0], good_matches[0]), chkmtd(kp2[1], good_matches[1]), chkmtd(kp2[2], good_matches[2]))
    
    mat3 = cv2.getRotationMatrix2D((img_reversed[0].shape[1]/2, img_reversed[0].shape[0]/2), chkmtd(kp2[0], good_matches[0]), 1)
    mat4 = cv2.getRotationMatrix2D((img_reversed[1].shape[1]/2, img_reversed[1].shape[0]/2), chkmtd(kp2[1], good_matches[1]), 1)
    mat5 = cv2.getRotationMatrix2D((img_reversed[2].shape[1]/2, img_reversed[2].shape[0]/2), chkmtd(kp2[2], good_matches[2]), 1)
    
    img3_corr = cv2.warpAffine(img_reversed[0], mat3, (img_reversed[0].shape[1], img_reversed[0].shape[0]))
    img4_corr = cv2.warpAffine(img_reversed[1], mat4, (img_reversed[1].shape[1], img_reversed[1].shape[0]) )
    img5_corr = cv2.warpAffine(img_reversed[2], mat5, (img_reversed[2].shape[1], img_reversed[2].shape[0]))
        
    
    
    plt.figure(figsize=(10, 10))
    plt.subplot(231)
    plt.imshow(img3)
    plt.subplot(232)
    plt.imshow(img4)
    plt.subplot(233)
    plt.imshow(img5)
    
    plt.subplot(234)
    plt.imshow(img3_corr)
    plt.subplot(235)
    plt.imshow(img4_corr)
    plt.subplot(236)
    plt.imshow(img5_corr)
    
    plt.show()
        
    
    
    
    
    # plt.figure(figsize=(20,5))
    # plt.subplot(141)
    # plt.imshow(img)
    # plt.subplot(142)
    # plt.imshow(img_reversed1)
    # plt.subplot(143)
    # plt.imshow(img_reversed2)
    # plt.subplot(144)
    # plt.imshow(img_reversed3)
    # plt.show()
    
    
    
def draw_sifted(img1, img2, img3, img4):
    # extract feature by SIFT
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    kp3, des3 = sift.detectAndCompute(img3, None)
    kp4, des4 = sift.detectAndCompute(img4, None)
    
    # draw img by plt
    plt.figure(figsize=(10, 10))
    plt.subplot(221)
    plt.imshow(cv2.drawKeypoints(img1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))
    plt.subplot(222)
    plt.imshow(cv2.drawKeypoints(img2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))
    plt.subplot(223)
    plt.imshow(cv2.drawKeypoints(img3, kp3, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))
    plt.subplot(224)
    plt.imshow(cv2.drawKeypoints(img4, kp4, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))
    plt.show()
    

def main():
    # 1
    img1 = cv2.imread("./assets/images/sample1.jpg")
    img2 = cv2.imread("./assets/images/sample2.jpg")
    # img3 = cv2.imread("./assets/images/sample3.jpg")
    # img4 = cv2.imread("./assets/images/sample4.jpg")
    
    # resize img to 400x400
    img1 = cv2.resize(img1, (img1.shape[1]//8, img1.shape[0]//8))
    img2 = cv2.resize(img2, (img2.shape[1]//8, img2.shape[0]//8))
    # img3 = cv2.resize(img3, (img3.shape[1]//8, img3.shape[0]//8))
    # img4 = cv2.resize(img4, (img4.shape[1]//8, img4.shape[0]//8))


    # draw_img(img1, img2, img3, img4)
    # draw_edge(img1, img2, img3, img4)
    # draw_sifted(img1, img2, img3, img4)
    
    
    # c0(img1, img2)
    

    # ? --- TESTING --- 
    # c0_1(img1, img2) # => result: smd 저항 같은 작은 부품들 식별하기에 해상도가 너무 낮음. 작은 부품 삽입된 부분 사진 필요할 듯, 
    
    # * ALGORITHM 2: FEATURE MATCHING
    # feature1(stitched)
    # * ALGORITHM 3: PARTS DETECTION AND STROKE
    stitched = cv2.imread("./assets/images/stitch/pcb-stitching.png")
    # stroke1(stitched)
    
    dcheck1(stitched) # 
    
    # ! --- DONE ---    
    # * ALGORITHM 1: IMAGE STITCHING
    # stitched = stitch1(True)

    # c1(img1, img2, img3, img4)
    
    
   
    # cv2.imshow('t',img)
    # cv2.waitKey(0)
    

if __name__ == "__main__":
    main()