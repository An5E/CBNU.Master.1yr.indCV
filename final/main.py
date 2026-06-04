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
    c0_1(img1, img2) # => result: smd 저항 같은 작은 부품들 식별하기에 해상도가 너무 낮음. 작은 부품 삽입된 부분 사진 필요할 듯, 

    # c1(img1, img2, img3, img4)
    
    
   
    # cv2.imshow('t',img)
    # cv2.waitKey(0)
    

if __name__ == "__main__":
    main()