import numpy as np

def hough_transform(binary_img, theta_step=0.1):
    w, h = binary_img.shape
    diag = int(round(np.sqrt(w*w + h*h)))
    rho = np.linspace(-diag, diag, num= diag*2)
    theta = np.arange(-np.pi/2, np.pi/2, theta_step)
    A = np.zeros((len(rho), len(theta)))

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    for iy in range(binary_img.shape[1]):
        for ix in range(binary_img.shape[0]):
            # print(binary_img[iy,ix])
            if binary_img[ix,iy] == 1:
                for itheta in range(len(theta)):
                    rho_curr = ix*cos_t[itheta] + iy*sin_t[itheta]
                    irho = np.amax(np.nonzero(rho_curr>=rho)) if rho_curr<=0 else np.amin(np.nonzero(rho_curr<=rho))
                    A[irho, itheta] += 1

    return A

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    img = np.array([[1,0,0,0,0],
                    [0,1,0,0,0],
                    [0,0,1,0,0],
                    [0,0,0,1,0],
                    [0,0,0,0,1]]).astype(np.int16)
    
    A = hough_transform(img)
    plt.imshow(A, cmap='gray')
    