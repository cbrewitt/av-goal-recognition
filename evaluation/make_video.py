import cv2
import os
import matplotlib.pyplot as plt

image_folder = '../images/video_road'
video_name = 'video.avi'

images = ['{}.png'.format(idx) for idx in range(0, 174)]
frame = cv2.imread(os.path.join(image_folder, images[0]))

x = 415
y = 9
w = 872
h = 743
frame = frame[y:y+h, x:x+w]

height, width, layers = frame.shape


tree_images = ['{}.png'.format(idx) for idx in range(0, 174)]
tree_image_folder = '../images/video_tree/'
tree_frame = cv2.imread(os.path.join(tree_image_folder, tree_images[0]))
tree_rescale = h / tree_frame.shape[0]
new_tree_w = int(tree_frame.shape[1] * tree_rescale)


video = cv2.VideoWriter(video_name, 0, 25, (width + new_tree_w,height))
print((width + new_tree_w,height))


v_idx = 0
for image, tree_image in zip(images, tree_images):
    v_idx += 1
    img = cv2.imread(os.path.join(image_folder, image))
    tree_img = cv2.imread(os.path.join(tree_image_folder, tree_image))

    if img is not None and tree_img is not None:

        tree_img = cv2.resize(tree_img, (new_tree_w, h))
        img = img[y:y + h, x:x + w]
        img = cv2.hconcat([img, tree_img])
        # plt.imshow(img)
        # plt.show()
        # print(img.shape)
        #
        # video.write(img)
        cv2.imwrite('../images/video/img{}.png'.format(v_idx), img)

# cv2.destroyAllWindows()
# video.release()

#ffmpeg -r 25 -i img%01d.png -vcodec mpeg4 -b:v 5000k -y movie.mp4