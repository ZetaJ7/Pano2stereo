# Generating Pano Stereopairs from single pano and depth estimation
---
## Pipeline for Stereoscopic stitching work(Each frame)
1. Using DepthAnythingV2 to estimate depth (TODO: Using FlashDepth to get continuous depth and real-time efficency)
2. Convert image coords(xy) into sphere coords(R,theta,phi)
3. Generate Pano Stereo pairs from sphere coords and depth, according to Circular Projection
4. Convert left and right sphere coords back to image coords(xy)
5. Repair blank regions caused by stereo reprojection
6. (optional) Generate Red-cyan image, stereo image(left-right) and stereo videos
---
## Using StereoscoPy to Generate Red-Cyan stereoscopic pairs
install: pip install stereoscopy
CIL: StereoscoPy -S 5 0 -a -m color --cs red-cyan --lc rgb .\left_repaired.png .\right_repaired.png red_cyan.jpg\
-A: autoalignment(should be off)\
-a: anaglyph output\
-S: xy shift for left/right image\
-m: method\
-cs: color scheme (should be red-cyan)\
--lc: luma coding (should be rgb)

### Generate StereoPairs image and video from left and right by ffmpeg
ffmpeg -i .\left_repaired.png -i .\right_repaired.png -filter_complex "[0:v]scale=512:256[img1];[img1][1:v]hstack" stereo.jpg
ffmpeg -loop 1 -i .\stereo.jpg -c:v libx264 -t 60 -pix_fmt yuv420p stereo_output.mp4