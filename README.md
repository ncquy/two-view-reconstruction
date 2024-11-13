# Two-view Geometry: Two-view Reconstruction

This project reconstruct a 3D building structure from two images (Wadham College dataset).

## Table of Contents
- [Requirements](#requirements)
- [Results and Report](#results-and-report)
- [Acknowledgments](#acknowledgments)

## Requirements
- Python 3.x
- OpenCV
- NumPy
- Matplotlib

1. Clone the repository:

   ```bash
   git clone https://github.com/ncquy/two-view-reconstruction.git
   cd two-view-reconstruction

2. Install the required packages:
   ```bash
   pip install -r requirements.txt


## Results and Report
1. 2D point visualization on two images.
<p align='center'>
  <img width="600px" src="https://github.com/ncquy/two-view-reconstruction/blob/main/results/2d_points_on_img.png"/>
  <br/>
  <i> 2D point visualization on two images.</i>
</p>

2. The essential matrix between two images:
<p align='center'>
  <img width="400px" src="https://github.com/ncquy/two-view-reconstruction/blob/main/results/essential_matrix.png" />
  <br/>
  <i> The essential matrix between two images.</i>
</p>

3.   The rotation and translation, R and t:
<p align='center'>
  <img width="400px" src="https://github.com/ncquy/two-view-reconstruction/blob/main/results/r_t.png" />
  <br/>
  <i> The rotation and translation, R and t.</i>
</p>

4. 3D visualization of reconstructed 3D points in more than two viewpoints:
<p align='center'>
  <img width="400px" src="https://github.com/ncquy/two-view-reconstruction/blob/main/results/3d_points.png" />
  <img width="400px" src="https://github.com/ncquy/two-view-reconstruction/blob/main/results/3d_points_2.png" />
  <br/>
  <i> 3D visualization of reconstructed 3D points in more than two viewpoints.</i>
</p>

### Authors
* [Nguyen Cong Quy](https://github.com/ncquy)

