obj = VideoReader('Elephant.avi');
video = read(obj);
frames = obj.NumFrames;
for x = 1 : frames
    imwrite(video(:,:,:,x),strcat('frame-',num2str(x),'.png'));
end