import numpy as np

def distanceRightEye(c):
        eR_36 = c[35]
        eR_37 = c[36]
        eR_38 = c[37]
        eR_39 = c[38]
        eR_40 = c[39]
        eR_41 = c[40]
        x1 = np.linalg.norm(np.array([eR_37])-np.array([eR_41]))
        x2 = np.linalg.norm(np.array([eR_38])-np.array([eR_40]))
        return ((x1 + x2) / 2)

def distanceLeftEye(c):
    eL_42 = c[41]
    eL_43 = c[42]
    eL_44 = c[43]
    eL_45 = c[44]
    eL_46 = c[45]
    eL_47 = c[46]
    x1 = np.linalg.norm(np.array([eL_43])- np.array([eL_47]))
    x2 = np.linalg.norm(np.array([eL_44])- np.array([eL_46]))
    return ((x1 + x2) / 2)

def eyePoints():
    return [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]

def distanceBetweenMouth(c):
    m_60 = c[59]
    m_61 = c[60]
    m_62 = c[61]
    m_63 = c[62]
    m_64 = c[63]
    m_65 = c[64]
    m_66 = c[65]
    m_67 = c[66]
    x1 = np.linalg.norm(np.array([m_61])-np.array([m_67]))
    x2 = np.linalg.norm(np.array([m_62])-np.array([m_66]))
    x3 = np.linalg.norm(np.array([m_63])-np.array([m_65]))
    return ((x1+x2+x3)/3)


def heightFace(c):
    bottom = np.min(c[:,1])
    top = np.min(c[48:68,1])
    return top-bottom




def mouthPoints():
    return [60,61,62,63,64,65,66,67]