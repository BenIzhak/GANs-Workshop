import cv2


def getEdgeMap(img, lwT, upT):
    edges = cv2.Canny(img, lwT, upT)
    return edges

def convertToBandW(img, i):
    im_bw = cv2.threshold(img, 125, 255, cv2.THRESH_BINARY)[1]
    im_bw = (255 - im_bw)
    cv2.imwrite('edges\\' + 'e-' + str(i) + '.png', im_bw)
    
def convertToGrayscale(img, i):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('gray\\' + 'g-' + str(i) + '.png',gray_img)

def convertImg(startIndex, endIndex, mode = 0 ,lwT = 120, upT = 300):
    '''
    To get the edge map set mode to 0 and create a directory named edges.
    To get the grayscale image set mode to 1 and create a directory named gray.
    In both cases the script, the images and the new directory should be in the
    same directory.
    '''
    for i in range(startIndex, endIndex + 1):
        path = 'frog-' + str(i) + '.png'
        img = cv2.imread(path)
        if(mode == 0):
            edges = getEdgeMap(img, lwT, upT)
            convertToBandW(edges, i)
        if(mode == 1):
            convertToGrayscale(img, i)
        
        
convertImg(1, 7796, 0 ,lwT = 120, upT = 300)