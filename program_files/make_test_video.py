"""Script makes video and saves start and end positions of points fixed with blooms"""
import numpy as np
import matplotlib.pyplot as plt
import random
import myDIClibrary as mdl
from blooms import Bloom

PLTSHOW=False
RECORDVIDEO=True
NUMER_PROBY = 8
RLIM=[100,100]

FOLDER = 'start_end_points'
FOLDER_START_END = 'start_end_points_txt'

if RECORDVIDEO:
    plt.plot(0,0)
    myVideo=mdl.RecordVideo()

start_points = np.mgrid[20:80:19,20:80:9].reshape(2, -1).T
np.savetxt(FOLDER_START_END+'/startpointsshape'"{:02d}".format(NUMER_PROBY)+'.txt', start_points.shape)

listOfPLame=[]
for id_plama in range(0,start_points.shape[0]):
    listOfPLame.append(Bloom(start_points[id_plama][0], start_points[id_plama][1], 5))

def displacement(x,y,k=1):
    # return x+math.sin((x**2+y**2)/10)*0.01*x,y+math.cos((x**2+y**2)/10)*0.01*y
    return 50+(x-50)*1.02,50+(y-50)*1.01

for klatka in range(0,10):
    print(klatka)
    plt.clf()
    plt.gcf().set_size_inches(6, 6)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    for id_plama in range(0,start_points.shape[0]):
        if True:
            listOfPLame[id_plama].Xg += random.randint(-1, 1)
            listOfPLame[id_plama].Yg += random.randint(-1, 1)
            # listOfPLame[id_plama].Xg,listOfPLame[id_plama].Yg = displacement(listOfPLame[id_plama].Xg,listOfPLame[id_plama].Yg,klatka/10)
            for i in range(listOfPLame[id_plama].num_of_circles):
                plt.gca().add_patch(listOfPLame[id_plama].plotgggg(i))
            plt.xlim(0, RLIM[0])
            plt.ylim(0, RLIM[1])
    plt.plot(0,0,'k.')
    plt.plot(0,100,'k.')
    plt.plot(100,0,'k.')
    plt.plot(100,100,'k.')
    plt.axis('equal')
    if RECORDVIDEO:
        myVideo.add_new_frame()
    path = FOLDER+'/probe_'+"{:02d}".format(NUMER_PROBY)+'_frame_'+"{:04d}".format(klatka)+'.png'
    plt.savefig(path,pad_inches = 0,bbox_inches='tight',dpi=100)
    print(path)
    if PLTSHOW:
        plt.show()

if RECORDVIDEO:
    myVideo.save()

end_points = np.zeros(shape=start_points.shape)

for point in range(0,end_points.shape[0]):
    end_points[point]=(listOfPLame[point].Xg,listOfPLame[point].Yg)

# print(start_points)
# print(end_points)

start_points=start_points.astype(float)
start_points[:,0],start_points[:,1]=(start_points[:,1])/RLIM[1],start_points[:,0]/RLIM[0]
end_points=end_points.astype(float)
end_points[:,0],end_points[:,1]=(end_points[:,1])/RLIM[1],end_points[:,0]/RLIM[0]

np.savetxt(FOLDER_START_END+'/startpoints'"{:02d}".format(NUMER_PROBY)+'.txt', start_points, fmt='%.3f')
np.savetxt(FOLDER_START_END+'/endpoints'"{:02d}".format(NUMER_PROBY)+'.txt', end_points, fmt='%.3f')

print('\n'+FOLDER_START_END+'/startpoints'"{:02d}".format(NUMER_PROBY)+'.txt')
print(FOLDER_START_END+'/endpoints'"{:02d}".format(NUMER_PROBY)+'.txt')

