Reconstructing images with KNN

STEPS:

database of images with GIST descriptors
data-augmented (stretching, zooming, contrast)

analyze target image with canny edge detection
enforce closed loops, merge together regions that are too small
for each region in target image:
  find closest match from region + neighborhood --> database using KNN on GIST descriptors
  template-match each region with top 100 (or so) to find closest match
  merge matches from database into region
  (smooth around edges, possibly?)
