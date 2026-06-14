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

def detectPCBandZoom(src, drawContour=False, showPlt=False) -> cv2.typing.MatLike: # PCB 외형 검출

    # detect big components on pcb
    
    
    def mtd2(img):
        # detect big components on pcb
        img21 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        img22 = cv2.GaussianBlur(img21, (5,5), 0)
        
        img23 = cv2.inRange(img22, 70, 120)
        
        img231 = cv2.inRange(img22, 100, 220)
        
        img232 = cv2.bitwise_and(img23, img231)

        # cv2.imshow(f"mtd2:img23 (before)",img23)
        # cv2.imshow(f"mtd2:img231",img231)  
        # cv2.imshow(f"mtd2:img232 (after)",img232)
        
        img24 = cv2.Canny(img23, 100, 200)
        
        mp_kernel = np.ones((21,21), np.uint8)
        img241 = cv2.morphologyEx(img24, cv2.MORPH_CLOSE, kernel=mp_kernel)
        
        # cv2.imshow(f"mtd2:img241",img241)    
        
        # find contours
        contours2, _ = cv2.findContours(img241, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
        # cv2.imshow(f"mtd2:img24",img24)
        pcb_contour2 = None
        for c in contours2:
            
            ca = cv2.contourArea(c)
            
            if ca >= 10000:
            
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.0001 * peri, False)
                
                if len(approx) >= 4:
                    pcb_contour2 = approx
                    break
    
    
        # draw contours
        # img251 = cv2.drawContours(img1_cp, contours2, -1, (255, 0, 0), 1)
        # cv2.imshow(f"mtd2:img251",img251)
        
        img25 = cv2.drawContours(img.copy(), [pcb_contour2], -1, (255, 0, 0), 2)    
        # cv2.imshow(f"mtd2:img25",img25) # PCB 영역 식별
        
        
        # PCB 식별 영역 확대
        # print(pcb_contour2)
        
        x,y,w,h = cv2.boundingRect(pcb_contour2)
        
        # cv2.imshow("mtd2:img25'",img1_cp[y:y+h, x:x+w])
        # img26 = cv2.resize()
    
        return img25, img[y:y+h, x:x+w], pcb_contour2
    
    img21, img22, contours = mtd2(src)
    
    if showPlt:
    
        plt.figure(figsize=(10,5))
        plt.subplot(131)
        plt.title("original")
        plt.imshow(src)
        
        plt.subplot(132)
        plt.title("processed#2-1")
        plt.imshow(img21)
        
        plt.subplot(133)
        plt.title("processed#2-2 (detect pcb & zoom)")
        plt.imshow(img22)
        
        plt.show()
    
    img22 = cv2.cvtColor(img22, cv2.COLOR_BGR2RGB)
        
    return img22, contours

def detectDarkComponents(src,showplt=False):
    img = src[0]
    
    # detect big components on pcb
    img11 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    
    # inrange ( dark components )
    mask = cv2.inRange(img11, np.array([0,0,0]),np.array([255,90,90])) # dark components
        
    if showplt:
        cv2.imshow("original",img)
        cv2.imshow("mask1 (dark)",mask)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return mask
    
    
def detectBrightComponents(src, showplt=False):
    img = src[0]
    ctr = src[1]
    
    # detect big components on pcb
    img11 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # inrange ( dark components )
    mask = cv2.inRange(img11, np.array([0,0,150]),np.array([255,90,255])) # bright components
    
    mask_ctr1 = np.zeros(img.shape, dtype=np.uint8)
    
    x,y,_,_ = cv2.boundingRect(ctr)
    
    ctr11 = ctr.reshape(-1,2)-[x,y]
    
    cv2.fillPoly(mask_ctr1, [np.array([ctr11],dtype=np.int32)], 255)
    mask_ctr1 = cv2.cvtColor(mask_ctr1, cv2.COLOR_RGB2GRAY)
    
    comp13 = cv2.bitwise_and(mask, mask_ctr1)
    
    if showplt:
        cv2.imshow("comp13 (m3&ctr1)",comp13)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return comp13

def getComponentArea(src, ctr, thres=1000):
    img = src
    img1 = img.copy()
    
    # get contour area, and filter area by threshold
    contours, _ = cv2.findContours(ctr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    comp_area = []
    
    for c in contours:
            
            ca = cv2.contourArea(c)
            
            # Dark: > 1000
            if ca > thres:
            
                peri = cv2.arcLength(c, False)
                approx = cv2.approxPolyDP(c, 0.0001 * peri, False)
                
                comp_area.append(approx)
    
    for c in comp_area:
        x,y,w,h = cv2.boundingRect(c)
        
        img1 = cv2.rectangle(img1, (x, y), (x+w, y+h),  (0, 0, 255), 2)
    
    
    # ---

    cv2.imshow("src(origin)",img)
    cv2.imshow("img1 (draw component rect)",img1)
    
    cv2.waitKey()
    cv2.destroyAllWindows()
    
    

# * ORIGIN CODE OF BRIGHTs, DARKs
def detectComponents(src1, src2):
    img1 = src1[0]
    img2 = src2[0]
    
    ctr1 = src1[1]
    ctr2 = src2[1]
    
    
    # detect big components on pcb
    img11 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    img21 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    
    
    # inrange ( dark components )
    thres2 = 65
    
    thres1 = ([0,0,0],[255,255,255])
    
    mask1 = cv2.inRange(img11, np.array([0,0,0]),np.array([255,90,90])) # dark components
    mask2 = cv2.inRange(img11, np.array([3,3,210]),np.array([255,255,255])) 
    mask3 = cv2.inRange(img11, np.array([0,0,150]),np.array([255,90,255])) # bright components
    
    mask_ctr1 = np.zeros(img1.shape, dtype=np.uint8)
    
    x,y,_,_ = cv2.boundingRect(ctr1)
    
    ctr11 = ctr1.reshape(-1,2)-[x,y]
    
    print(ctr11)
    
    cv2.fillPoly(mask_ctr1, [np.array([ctr11],dtype=np.int32)], 255)
    mask_ctr1 = cv2.cvtColor(mask_ctr1, cv2.COLOR_RGB2GRAY)
    cv2.imshow("mask_ctr1",mask_ctr1)
    
    # mask31 = cv2.bitwise_and(ctr2, mask3)
    # cv2.imshow("mask31",mask31)
    
    # comp1 = cv2.bitwise_or(mask1, mask3)
    comp1 = cv2.bitwise_or(mask1, mask3)
    comp13 = cv2.bitwise_and(mask3, mask_ctr1)
    comp12 = cv2.bitwise_and(comp1, mask_ctr1) 
    
    
    # for i in range(10):
        # img12 = cv2.inRange(img11, 5, thres2+i)
        # img22 = cv2.inRange(img21, 5, thres2+i)
        
        # cv2.imshow(f"img12, {i}",img12)
        # cv2.imshow(f"img22, {i}",img22)
    cv2.imshow("original",img1)
    cv2.imshow("mask1 (dark)",mask1)
    cv2.imshow("mask2",mask2)
    cv2.imshow("mask3 (bright)",mask3)
    
    cv2.imshow("comp1 (m1|m3)",comp1)
    cv2.imshow("comp12 (comp1&ctr1)", comp12)
    cv2.imshow("comp13 (m3&ctr1)",comp13)
    
    # cv2.imshow("comp2 (m1&mctr1)",comp2)
    
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # # blur
    # blurFactor = 3
    
    # img2 = cv2.GaussianBlur(img2, (blurFactor,blurFactor), 0)
    
    # # detect edge
    # img2 = cv2.Canny(img2, 100, 200)
    
    # # find contours
    # contours, _ = cv2.findContours(img2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # # draw contours
    # img2 = cv2.drawContours(img2, contours, -1, (255, 0, 0), 2)
    
    
    # plt.figure(figsize=(10,5))
    # plt.subplot(131)
    # plt.title("original")
    # plt.imshow(img1)
    # plt.subplot(132)
    # plt.title("processed#1")
    # plt.imshow(img2)
    
    # plt.show()

def detectSmallComponents():
    pass

def contour1():
    img1 = cv2.imread("./assets/images/pcb-stitching1.png")
    img2 = cv2.imread("./assets/images/pcb-stitching2.png")
    
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    
    # blur
    blurFactor = 3
    
    img1 = cv2.GaussianBlur(img1, (blurFactor,blurFactor), 0)
    img2 = cv2.GaussianBlur(img2, (blurFactor,blurFactor), 0)
    
    # detect edge
    img1 = cv2.Canny(img1, 100, 200)
    img2 = cv2.Canny(img2, 100, 200)
    
    # find contours
    contours1, _ = cv2.findContours(img1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(img2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # draw contours by contour size over #
    for i in range(len(contours1)):
        if cv2.contourArea(contours1[i]) > 1000:
            img1 = cv2.drawContours(img1, [contours1[i]], -1, (255, 0, 0), 2)
    
    for i in range(len(contours2)):
        if cv2.contourArea(contours2[i]) > 1000:
            img2 = cv2.drawContours(img2, [contours2[i]], -1, (255, 0, 0), 2)
    
    # GRAY TO RGB
    # img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(img1, cmap='gray')
    plt.subplot(122)
    plt.imshow(img2, cmap='gray')
    
    plt.show()

def stitch1(showplt: bool = False, saveimg: bool = False) -> cv2.typing.MatLike:
    # read img
    img1 = cv2.imread("./assets/images/sample2-1.jpg")
    img2 = cv2.imread("./assets/images/sample2-2.jpg")
    img3 = cv2.imread("./assets/images/sample2-3.jpg")
    img4 = cv2.imread("./assets/images/sample2-4.jpg")
    img5 = cv2.imread("./assets/images/sample2-5.jpg")
    
    # image stitching
    stitcher = cv2.Stitcher_create()
    (status, stitched) = stitcher.stitch([img1,img2, img3, img4, img5])
    
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
    # plt.subplot(154)
    plt.subplot(111)
    plt.imshow(stitched)
    
    plt.tight_layout()
        
    if saveimg:
        plt.axis('off')
        plt.savefig("./assets/images/pcb-stitching.png", bbox_inches='tight', pad_inches=0.0)
    else:
        plt.title("PCB stitching with 6 parts")
    
    if showplt:
        plt.show()
        
        
    
    return stitched

def feature1(src1, src2):
    # copy img1 to img2
    # img2 = img1.copy()
    
    img1 = src1 # cv2.imread("./assets/images/pcb-stitching1.png")
    img2 = src2 # cv2.imread("./assets/images/pcb-stitching2.png")
    
    # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # extract features from images and matching
    
    sift = cv2.SIFT_create(contrastThreshold=0.12, edgeThreshold=2)
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    print(kp1[0].pt)
    
    for _kp in kp1:
        cv2.circle(img1, (int(_kp.pt[0]), int(_kp.pt[1])), 5, (255, 0, 0), 2)
    # for kp in kp2:
    #     cv2.circle(img2, (kp.pt[0], kp.pt[1]), 5, (255, 0, 0), 2)
    
    # plt.figure(figsize=(10, 10))
    # plt.subplot(121)
    # plt.imshow(img1)
    # plt.subplot(122)
    # plt.imshow(img2)
    # plt.show()
    
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
    
def dcheck1(img, img2): # PCB 방향 인식
    img_reversed = [cv2.rotate(img2, cv2.ROTATE_180), cv2.rotate(img2, cv2.ROTATE_90_CLOCKWISE), cv2.rotate(img2, cv2.ROTATE_90_COUNTERCLOCKWISE)]
    
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
        
        print(M)
        
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
    
    # print(chkmtd(kp2[0], good_matches[0]), chkmtd(kp2[1], good_matches[1]), chkmtd(kp2[2], good_matches[2]))
    
    mat3 = cv2.getRotationMatrix2D((img_reversed[0].shape[1]/2, img_reversed[0].shape[0]/2), chkmtd(kp2[0], good_matches[0]), 1)
    mat4 = cv2.getRotationMatrix2D((img_reversed[1].shape[1]/2, img_reversed[1].shape[0]/2), -chkmtd(kp2[1], good_matches[1]), 1)
    mat5 = cv2.getRotationMatrix2D((img_reversed[2].shape[1]/2, img_reversed[2].shape[0]/2), -chkmtd(kp2[2], good_matches[2]), 1)
    
    img3_corr = cv2.warpAffine(img_reversed[0], mat3, (img_reversed[0].shape[1], img_reversed[0].shape[0]))
    img4_corr = cv2.warpAffine(img_reversed[1], mat4, (img_reversed[1].shape[1], img_reversed[1].shape[0]) )
    img5_corr = cv2.warpAffine(img_reversed[2], mat5, (img_reversed[2].shape[1], img_reversed[2].shape[0]))
        
    
    
    plt.figure(figsize=(15, 10))
    plt.subplot(231)
    plt.imshow(img3)
    plt.title("PCB direction (Golden : Target 180')")
    plt.subplot(232)
    plt.imshow(img4)
    plt.title("PCB direction (Golden : Target CW 90')")
    plt.subplot(233)
    plt.imshow(img5)
    plt.title("PCB direction (Golden : Target CCW 90')")
    
    plt.subplot(234)
    plt.imshow(img3_corr)
    plt.title("PCB direction' (Target)")
    plt.subplot(235)
    plt.imshow(img4_corr)
    plt.subplot(236)
    plt.imshow(img5_corr)
    
    plt.tight_layout()
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
    # img1 = cv2.imread("./assets/images/sample1.jpg")
    # img2 = cv2.imread("./assets/images/sample2.jpg")
    # img3 = cv2.imread("./assets/images/sample3.jpg")
    # img4 = cv2.imread("./assets/images/sample4.jpg")
    
    # resize img to 400x400
    # img1 = cv2.resize(img1, (img1.shape[1]//8, img1.shape[0]//8))
    # img2 = cv2.resize(img2, (img2.shape[1]//8, img2.shape[0]//8))
    # img3 = cv2.resize(img3, (img3.shape[1]//8, img3.shape[0]//8))
    # img4 = cv2.resize(img4, (img4.shape[1]//8, img4.shape[0]//8))


    # draw_img(img1, img2, img3, img4)
    # draw_edge(img1, img2, img3, img4)
    # draw_sifted(img1, img2, img3, img4)
    
    
    # c0(img1, img2)
    

    # ? --- TESTING --- 
    # c0_1(img1, img2) # => result: smd 저항 같은 작은 부품들 식별하기에 해상도가 너무 낮음. 작은 부품 삽입된 부분 사진 필요할 듯, 
    
    # * DRAW CONTOUR
    # contour1()
    # detectBigComponents()
    
    # * ( DETECT PCB & ZOOM ) --> ( DETECT DARK/BRIGHT COMPONENTS )
    img1 = cv2.imread("./assets/images/pcb-stitching1.png")
    img2 = cv2.imread("./assets/images/pcb-stitching2.png")
    zoomed1, contours1 = detectPCBandZoom(img1) 
    zoomed2, contours2 = detectPCBandZoom(img2)
    
    dc1 = detectDarkComponents((zoomed1, contours1))
    dc2 = detectDarkComponents((zoomed2, contours2))
    
    bc1 = detectBrightComponents((zoomed1, contours1))
    bc2 = detectBrightComponents((zoomed2, contours2))
    
    # * ( GET COMPONENT AREA )
    # getComponentArea(zoomed1,dc1)
    # getComponentArea(zoomed1,bc1,200)
    
    getComponentArea(zoomed2,dc2)
    getComponentArea(zoomed2,bc2,200)
    
    
    # ---
    # cv2.imshow("dc1",dc1)
    # cv2.imshow("dc2",dc2)
    # cv2.imshow("bc1",bc1)
    # cv2.imshow("bc2",bc2)
    
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    
    
    
    
    
    # * ALGORITHM 2: FEATURE MATCHING
    # stitched = cv2.imread("./assets/images/pcb-stitching1.png")
    # feature1()
    # * ALGORITHM 3: PARTS DETECTION AND STROKE
    # stitched = cv2.imread("./assets/images/stitch/pcb-stitching.png")
    # stroke1(stitched)
    

    
    # ! --- DONE ---    
    # * ALGORITHM 1: IMAGE STITCHING
    # stitched = stitch1(True, True)
    
    # * ALGORITHM ? : CHECK PCB DIRECTION
    img1 = cv2.imread("./assets/images/pcb-stitching1.png")
    img2 = cv2.imread("./assets/images/pcb-stitching2.png")
    dcheck1(img1,img2) # 
    
    # * DETECT PCB & ZOOM
    # zoomed1 = detectPCBandZoom(img1) 
    # zoomed2 = detectPCBandZoom(img2)

    # * DETECT BIG COMPONENTS (WIP)
    
    # * DETECT SMALL COMPONENTS (WIP)

    # * FEATURE MATCHING
    # feature1(zoomed1, zoomed2)

    # c1(img1, img2, img3, img4)
    
    
   
    # cv2.imshow('t',img)
    # cv2.waitKey(0)
    

if __name__ == "__main__":
    main()