"""Library for processing pictures and segments"""
import glob
import os.path
import cv2
import numpy
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import filters

def read_to_monochrome(path):
    """read png to monochrome raster"""
    img = plt.imread(path)
    pict_mono = (img[:, :, 0] + img[:, :, 1] + img[:, :, 2]) / 3    # * img[:, :, 3]
    return pict_mono

def get_phase2D(A,B,radius):
    """get phase on 2D arrays with standard correlation"""
    A=A-np.min(A)
    B=B-np.min(B)
    if np.any(A!=0) and np.any(B!=0):
    # if True:
        M=A.shape[0]
        N=A.shape[1]
        maxval=0
        indmaxval=[0,0]
        for tau_m in range(-radius,radius+1):
            # choosing m range
            if tau_m<=0:
                m_srt = -tau_m
                m_end = M
            else:
                m_srt = 0
                m_end = M-tau_m
            for tau_n in range(-radius,radius+1):
                # choosing n range
                if tau_n<=0:
                    n_srt = -tau_n
                    n_end = N
                else:
                    n_srt = 0
                    n_end = N-tau_n
                # calculate correlation value on selected parts of areas
                A1 = A[m_srt:m_end,n_srt:n_end]
                B1 = B[(m_srt+tau_m):(m_end+tau_m),(n_srt+tau_n):(n_end+tau_n)]
                corrmn=np.sum(A1*B1)
                # actualise maximum value
                if corrmn>maxval:
                    maxval=corrmn
                    indmaxval=[tau_m+M-1,tau_n+N-1]
        # print(maxval)
        return np.array(indmaxval)-np.array([M,N])+1
    else:
        # print('zeros')
        return np.array([0,0])

def get_phase2D_var(A, B, radius):
    """get phase on 2D arrays with correlation depended on minimum of variation"""
    # A=A-np.min(A)
    # B=B-np.min(B)
    if np.any(A!=0) and np.any(B!=0):
        M=A.shape[0]
        N=A.shape[1]
        minval=M*N
        indmval=[0,0]
        for tau_m in range(-radius,radius+1):
            # choosing m range
            if tau_m<=0:
                m_srt = -tau_m
                m_end = M
            else:
                m_srt = 0
                m_end = M-tau_m
            for tau_n in range(-radius,radius+1):
                # choosing n range
                if tau_n<=0:
                    n_srt = -tau_n
                    n_end = N
                else:
                    n_srt = 0
                    n_end = N-tau_n
                # calculate correlation value on selected parts of areas
                A1 = A[m_srt:m_end, n_srt:n_end]
                B1 = B[(m_srt + tau_m):(m_end + tau_m), (n_srt + tau_n):(n_end + tau_n)]
                corrmn = np.sum(np.power( A1 - B1 ,2))#/((-m_srt+m_end))
                # actualise minimum value
                if corrmn<minval:
                    minval=corrmn
                    indmval=[tau_m+M-1,tau_n+N-1]
        # print(minval)
        if minval==0:
            return[0,0]
        else:
            return np.array(indmval)-np.array([M,N])+1
    else:
        # print('zeros')
        return [0,0]


class Image():
    def __init__(self,numpy_array):
        """set raster of numpy array class"""
        self.raster = numpy_array
    def draw_on_plot(self):
        plt.imshow(self.raster, cmap='gray', vmin=0, vmax=1)
    def get_shape(self):
        return self.raster.shape
    def normalise(self):
        """normalise each pixel value to 0-1 range"""
        max_val = np.max(self.raster)
        min_val = np.min(self.raster)
        if max_val-min_val>0.0001:
            self.raster = 0 + 1 / (max_val - min_val) * (self.raster - min_val)
    def get_quantile_value(self, prob):
        """return value of quantile"""
        temp_list = np.reshape(self.raster,(self.raster.size,1)).tolist()
        target = len(temp_list) * prob
        temp_list = sorted(temp_list)
        return temp_list[int(target)]
    def get_quantile_value_fast(self, prob):
        """return value of quantile"""
        return numpy.quantile(self.raster,prob)
    def do_convolution(self, con_matrix, normalize=True):
        """apply the mask"""
        rnum, cnum = self.raster.shape
        new_pict = np.zeros((rnum, cnum))
        for r in range(0, rnum - 0):
            for c in range(0, cnum - 0):
                rgs = [0, 0, 0, 0]
                cenpos = [1, 1]
                sumel = 0
                # print(r,c)
                if r == 0:
                    # print('r==0')
                    rgs[0] = 1
                if r == rnum - 1:
                    # print('r==rnum')
                    rgs[2] = 1
                if c == 0:
                    # print('c==0')
                    rgs[1] = 1
                if c == cnum - 1:
                    # print('c==cnum')
                    rgs[3] = 1
                for i in range(0 + rgs[0], con_matrix.shape[0] - rgs[2]):
                    for j in range(0 + rgs[1], con_matrix.shape[1] - rgs[3]):
                        sumel += con_matrix[i, j]
                        new_pict[r, c] += con_matrix[i, j] * self.raster[r - cenpos[0] + i, c - cenpos[1] + j]
        self.raster = new_pict
        # return new_pict
    def do_convolution_fast(self, con_matrix, normalize=True):
        """apply the mask"""
        self.raster = filters.convolve(self.raster, con_matrix, mode='constant', cval=0.0)
        if normalize:
            self.normalise()
    def split(self, border):
        """make binarization of raster"""
        new_pict = np.zeros(self.raster.shape)
        jmax = new_pict.shape[1]
        for i in range(0, new_pict.shape[0]):
            for j in range(0, jmax):
                if self.raster[i, j] < border:
                    # print(i,j,pict[i,j],'<',border)
                    new_pict[i, j] = 0
                else:
                    # print(i,j,pict[i,j],'>',border)
                    new_pict[i, j] = 1
        self.raster = new_pict
    def cog(self):
        """return center of area"""
        self.s = np.sum(self.raster)
        self.sx = np.sum(np.arange(0,self.raster.shape[0])*np.sum(self.raster,axis=1))
        self.sy = np.sum(np.arange(0,self.raster.shape[1])*np.sum(self.raster,axis=0))
        if self.s != 0:
            return self.sy / self.s, self.sx / self.s, False
        else:
            return self.raster.shape[0]/2, self.raster.shape[1]/2, True
    def cog4(self):
        """return center of area at 4 power"""
        self.s = np.sum(np.power(self.raster,4))
        self.sx = np.sum(np.arange(0,self.raster.shape[0])*np.sum(np.power(self.raster,4),axis=1))
        self.sy = np.sum(np.arange(0,self.raster.shape[1])*np.sum(np.power(self.raster,4),axis=0))
        # print(self.s)
        if self.s != 0:
            return self.sy/self.s, self.sx/self.s, False
        else:
            return self.raster.shape[0]/2, self.raster.shape[1]/2, True # do wywalenia

class RecordVideo():
    def __init__(self):
        self.video_folder = 'records/video*'
        self.video_name=(sorted(glob.glob(self.video_folder),key=os.path.getmtime))[-1].split('_')
        self.lastname=self.video_name[-1].split('.')
        self.lastname='.'.join([str((int(self.lastname[-2])+1)),self.lastname[-1]])
        self.video_name='_'.join(self.video_name[:-1]+[self.lastname])
        self.filename = 'records/temp.jpg'
        plt.savefig(self.filename)#,pad_inches = 0,bbox_inches='tight')
        self.frame = cv2.imread(self.filename)
        self.height, self.width, self.layers = self.frame.shape
        self.video = cv2.VideoWriter(self.video_name, cv2.VideoWriter_fourcc(*'XVID'), 1, (self.width, self.height))
    def add_new_frame(self):
        plt.savefig(self.filename)#,pad_inches = 0,bbox_inches='tight')
        self.frame = cv2.imread(self.filename)
        self.video.write(self.frame)
    def save(self):
        cv2.destroyAllWindows()
        self.video.release()



# def get_next_points(x,y,k):
#
#     # PM = np.mgrid[-10:10,-10:10].reshape(2, -1).T
#     # DM = np.zeros(shape=(PM.shape))
#
#     def displacement(x,y,k=1):
#         return x+math.sin((x**2+y**2)/10)*0.1*x,y+math.cos((x**2+y**2)/10)*0.1*y
#
#     # for p in range(0,PM.shape[0]):
#     #     DM[p][0],DM[p][1]=strainF(PM[p][0],PM[p][1])
#
#     nPM = PM+DM
#
#     # for p in range(0,PM.shape[0]):
#     #     plt.plot(PM[p][0],PM[p][1],'k.')
#     #     plt.plot(nPM[p][0],nPM[p][1],'k.',markersize=10)
#     # plt.show()
#
#     return nPM, DM
