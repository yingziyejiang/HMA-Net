# Sample data

Three de-identified test-split examples, included so that the expected input format can be
reproduced without access to the full cohort. Names, hospital identifiers and DICOM metadata were
removed before transfer to the research server; the images here are the same de-identified files
used in the study.

```
data_sample/
  images/                 full orthopantomograms (grayscale JPEG)
  labels_obb5/            one file per image, one line per detected or annotated tooth
  sample_index.csv        age, sex and split for these three subjects
```

`labels_obb5/*.txt` columns:

```
fdi cx cy w h angle_deg
```

`fdi` is the FDI tooth number, `cx cy w h` are the oriented bounding-box centre and size
normalised by image width/height, and `angle_deg` is the rotation in degrees. Missing teeth have no
line. Class ids cover 52 positions (32 permanent, 20 deciduous).

`sample_index.csv` uses the same schema as the training manifest:

```
img_path,label_path,age,gender,split
```

`gender` is 0 for female and 1 for male. The full cohort (4,293 orthopantomograms, ages 3-18) is not distributed here; the manuscript
states how it can be requested.
