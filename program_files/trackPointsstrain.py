"""Main script for tracking points"""
import glob as glob
import numpy as np
import matplotlib.pyplot as plt
import os
import myDIClibrary as mdl
import nodesStrain


def lookForCages():
    """return paths IDs taking into account"""
    paths_range = range(int(VIDEO_RANGE[0] * len(PATHS)), int(len(PATHS) * VIDEO_RANGE[1]), 1)
    return paths_range

def probeProperties():
    """get properties of pictures"""
    im_start = mdl.Image(mdl.read_to_monochrome(PATHS[paths_range[0]]))
    im_start = image_processing(im_start)
    [X_limit,Y_limit] = im_start.get_shape()
    return X_limit,Y_limit

def createInitialPoints():
    """Create map of points at given rectangular range"""
    X_START, X_END, Y_START, Y_END = int(AREA_RANGE[0] * X_limit), int(AREA_RANGE[1] * X_limit) - SQUARE_SIZE, \
                                     int(AREA_RANGE[2] * Y_limit), int(AREA_RANGE[3] * Y_limit) - SQUARE_SIZE
    DEF_GRID_s = np.mgrid[Y_START:Y_END:STEP_SIZE, X_START:X_END:STEP_SIZE]
    start_points = DEF_GRID_s.reshape(2, -1).T
    return start_points, (DEF_GRID_s)

def trackPoints(RECORD_VIDEO=True,PROBE=False, CALC_STRAIN=False):
    """Return end positions of given initial points"""

    # reference frame
    im_start = mdl.Image(mdl.read_to_monochrome(PATHS[paths_range[0]]))
    im_start = image_processing(im_start)
    im_act=im_start

    # initialize of tracking points
    actual_points = start_points.copy()

    # record video to file
    if RECORD_VIDEO and (not PROBE):
        myVideo=mdl.RecordVideo()

    numplot=0
    print('Chosen cages:',' - '.join([str(item) for item in paths_range[0:-1:(len(paths_range)-2)]]))

    for id_path in paths_range[1:]:

        numplot+=1
        plt.clf()
        plt.axes().set_aspect('equal')
        print(str(numplot) + '/' + str(len(paths_range) - 1))

        # chosing reference and actual frame
        im_bef = im_start                                               # im_bef = copy(im_act) or average of previous
        im_act = mdl.Image(mdl.read_to_monochrome(PATHS[id_path]))
        im_act = image_processing(im_act,METHOD)

        # determine new points positions
        for ind, point in enumerate(actual_points):

            # actual point position
            y,x = point[0], point[1]
            # points on reference frame
            ys,xs = start_points[ind]           # on start frame
            # determine movement of point
            if METHOD == 'var':
                phase = mdl.get_phase2D_var(im_bef.raster[xs:xs + SQUARE_SIZE, ys:ys + SQUARE_SIZE],
                                            im_act.raster[x:x + SQUARE_SIZE, y:y + SQUARE_SIZE],
                                            SEARCH_RADIUS)[::-1]
            elif METHOD == 'mult':
                phase = mdl.get_phase2D(im_bef.raster[xs:xs + SQUARE_SIZE, ys:ys + SQUARE_SIZE],
                                            im_act.raster[x:x + SQUARE_SIZE, y:y + SQUARE_SIZE],
                                            SEARCH_RADIUS)[::-1]
            # new position of point
            actual_points[ind] += [phase[0], phase[1]]
            ########################################

        if RECORD_VIDEO or PROBE:
            im_act.draw_on_plot()
            plt.title('Frame: '+str(id_path)+'\nmethod: '+nodesStrain.plotting_type_of_strain)
            for ind, point in enumerate(actual_points):
                y, x = point[0], point[1]
                plt.plot(y + SQUARE_SIZE / 2, x + SQUARE_SIZE / 2, 'b.', alpha=0.8, markersize=5)
                # plt.plot([y, y, y + SQUARE_SIZE, y + SQUARE_SIZE, y],
                #          [x, x + SQUARE_SIZE, x + SQUARE_SIZE, x, x],
                #          'g-', alpha=0.6)

        if CALC_STRAIN:
            # print(actual_points.shape)
            # get deformation of elements
            DEF_GRID_ACT = nodesStrain.matrix_DIC_to_strain(actual_points[:], DEF_GRID_0.shape) + int(SQUARE_SIZE / 2)

            min_val, max_val = -0.000001, 0.000001      # for plotting if start strain is 0

            for el in ELEMENTS:
                # el.plot_points(DEF_GRID_ACT)
                # plot element on actual grid
                el.plot_element(DEF_GRID_ACT)
                # set element strains
                el.strain(DEF_GRID_0,DEF_GRID_ACT)
                # choose strain to show
                el.set_strain_val()
                to_show = el.strain_val
                # finding extremums
                if to_show< min_val:
                    min_val=to_show*1.0001
                if to_show> max_val:
                    max_val=to_show*1.0001
            # plot colormap of strain values
            for el in ELEMENTS:
                el.plot_strain(DEF_GRID_ACT, min_val, max_val )
            # show min and max values
            plt.text(0.01, 0.02, 'min: '+"{:.1f}".format(min_val*100)+' %', fontsize=7, ha='left', va='bottom',transform=plt.gca().transAxes,color='blue')
            plt.text(0.01, 0.05, 'max: '+"{:.1f}".format(max_val*100)+' %', fontsize=7, ha='left', va='bottom',transform=plt.gca().transAxes,color='red')

        if RECORD_VIDEO and (not PROBE):
            myVideo.add_new_frame()
        else:
            plt.show()

    if RECORD_VIDEO and (not PROBE):
        myVideo.save()

    return actual_points

#####################################################################################################

def image_processing(img,method = 'var'):
    """do image processing"""
    # changing resolution
    img.raster=img.raster[1::divider,1::divider]

    # cut raster
    sxs,sxe,sys,sye = 0.2 , 0.8 , 0.2 , 0.8
    sx,sy=img.raster.shape
    img.raster=img.raster[int(sx*sxs):int(sx*sxe),int(sy*sys):int(sy*sye)]

    if method == 'var':
        # normalize to 0-1
        img.normalise()
    elif method == 'mult':
        # do convolution with mask
        # MASK = np.array([[1,1,1],[1,2,1],[1,1,1]])
        # img.do_convolution(MASK)
        MASK = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
        img.do_convolution_fast(MASK)
        # MASK = np.array([[1,2,1],[2,2,2],[1,2,1]])
        # img.do_convolution_fast(MASK)

        # split by quntile
        img.split(img.get_quantile_value_fast(0.92))
        # img.raster=img.raster*(-1)+1
    else:
        print('Wrong method')
    return img

########################### control parameters #############################################
# setting parameters
KEY_PATH= r'videos/frames/vn011_frame_*.png'                    # paths with frames in order
# KEY_PATH = r'start_end_points\proba_08_klatka_*.png'
PATHS = sorted(glob.glob(KEY_PATH), key=os.path.getmtime)[::1]  # reorganization
print(PATHS)
VIDEO_RANGE = [ 0.1, 0.5 ]                  # range of video
AREA_RANGE = [0.1 , 0.9 , 0.1 , 0.9]        # set tracking area
divider=1                                   # reduce resolution

SQUARE_SIZE = int(30/divider)               # size of cell around tracking point
SEARCH_RADIUS = int(0.1*SQUARE_SIZE)        # radius of searched area (moving cell)
STEP_SIZE = int(SQUARE_SIZE*1)              # period between cells

METHOD = 'mult'                             # "var" or "mult"
nodesStrain.plotting_type_of_strain = 'eqv' # type of measured strain

#####################################################################################################

# getting paths for frames
paths_range = (lookForCages())
X_limit,Y_limit = probeProperties()
print(X_limit,Y_limit)

# making tracking points and elements
start_points, DEF_GRID_0 = createInitialPoints()
DEF_GRID_0 = DEF_GRID_0 + int(SQUARE_SIZE/2)
ELEMENTS=[]
for i in range(0,(DEF_GRID_0.shape[1]-1)*(DEF_GRID_0.shape[2]-1)):
    ELEMENTS.append(nodesStrain.Element(DEF_GRID_0.shape, nodesStrain.CONSTR, i))

# getting end position of points
tracked_end_points = trackPoints(PROBE=False, CALC_STRAIN=True)
np.savetxt("DIC_wr.txt",tracked_end_points)


######### track movie created by make_video_test.py and compare results with given positions of points #########

# NUMER_PROBY = 8
# FOLDER_START_END='start_end_points_txt'
# start_point_loaded=np.loadtxt(FOLDER_START_END+'/startpoints'"{:02d}".format(NUMER_PROBY)+'.txt').astype(float)
# end_point_loaded=np.loadtxt(FOLDER_START_END+'/endpoints'"{:02d}".format(NUMER_PROBY)+'.txt').astype(float)
# fx,fy = mdl.Image(mdl.read_to_monochrome(PATHS[paths_range[0]])).raster.shape
#
# start_points = (start_point_loaded)      # wygenerowany przez Cb
# start_points[:,1],start_points[:,0]=(start_points[:,0])*X_limit/divider*(-1)+X_limit-SQUARE_SIZE/2,(start_points[:,1])*Y_limit/divider-SQUARE_SIZE/2
# start_points=start_points.astype(int)
#
# end_points = (end_point_loaded)      # wygenerowany przez Cb
# end_points[:,1],end_points[:,0]=end_points[:,0]*X_limit/divider*(-1)+X_limit-SQUARE_SIZE/2,end_points[:,1]*Y_limit/divider-SQUARE_SIZE/2
# end_points=end_points.astype(int)

######### make correction of started points to nearest centers of areas #########

# im = mdl.Image(mdl.read_to_monochrome(PATHS[paths_range[0]]))
# print(im.raster.shape)
# fim = mdl.Image(mdl.read_to_monochrome(PATHS[paths_range[0]]))
# scale = 6
# # spts = np.mgrid[20:80:9,20:80:9].reshape(2, -1).T*scale
# spts = start_points
# if True:
#     to_remove=[]
#     for nop in range(0,spts.shape[0]):
#         tim = mdl.Image(fim.raster[spts[nop,1]:(spts[nop,1]+SQUARE_SIZE),spts[nop,0]:(spts[nop,0]+SQUARE_SIZE)])
#         tim.raster=tim.raster*(-1)+1
#         # tim.draw_on_plot()
#         tx, ty , out = tim.cog4()
#         tcog=[tx,ty]
#         if out:
#             to_remove.append(nop)
#             # print(spts[nop])
#         plt.plot(tcog[0],tcog[1],'yo')
#         # plt.show()
#         spts[nop]=spts[nop]+np.array([tcog])-int(SQUARE_SIZE/2)
#         tim = mdl.Image(fim.raster[spts[nop,0]:(spts[nop,0]+SQUARE_SIZE),spts[nop,1]:(spts[nop,1]+SQUARE_SIZE)])
#         # tim.raster=tim.raster*(-1)+1
#         # tim.draw_on_plot()
#         # plt.show()
#     spts=np.delete(spts,to_remove,axis=0)
#     end_points = np.delete(end_points, to_remove,axis=0)
# print('to remove', to_remove)
# start_points = copy(spts)

######## some statistic ############

# lengths=np.sqrt(np.sum(np.power(end_points-tracked_end_points,2),axis=1))
# print(lengths)
#
# odch_stand=np.std(end_points-tracked_end_points)
# print(odch_stand)





