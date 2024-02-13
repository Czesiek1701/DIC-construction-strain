"""cript makes png frames from recorded by camera video"""
import cv2

number_of_record = "011"
vidcap = cv2.VideoCapture(r'videos/vn_'+number_of_record+'.mp4')  # path to video

success,image = vidcap.read()
print(type(image))
count = 0
print(success)
while success:
  new_frame_path = "videos/frames/vn" + number_of_record + "_frame_%d.png" % count
  cv2.imwrite(new_frame_path, image)     # save frame as JPEG file
  print(new_frame_path)
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1