# Pipeline for Stereoscopic stitching work
1. Visualize the matches from superpoint+lightglue, learn the matches and points from the results;\
   1. Extract points and matches from results
2. Visualize the results from segmentation, learn the results and how to extract regions;\
   1. Extract semantic region mask from SAM2's results
   2. Make a class:Region to store region mask and points inside
3. Match points to semantic region, build class to save them;\
   1. 
4. Learn how to use points to controll image warping;\
5. Learn DLT/MDLT to estimate optimal homography;\
6. Learn DLT to solve energy minimization problem, write class to realize with Homography, Disparity, smoothness;\
7. Stereo Stitching;\
# Using StereoscoPy to Generate Red-Cyan stereoscopic pairs
install: pip install stereoscopy
---
CIL: StereoscoPy -S 5 0 -a -m color --cs red-cyan --lc rgb imgs_ss/11040035_l.png imgs_ss/11040035_r.png results/anaglyph_color_rc.jpg\
-A: autoalignment(should be off)\
-a: anaglyph output\
-S: xy shift for left/right image\
-m: method\
-cs: color scheme (should be red-cyan)\
--lc: luma coding (should be rgb)