# Instance Segmentation / Object Detection
## Proposal-based methods
## Proposal-free methods
### Semi-convolutional operators
1. Instance coloring.
2. y = phi(x) + (u, v)
3. min(y - cy). Attractive force only.
# SLAM/SfM
## Basics
### Direct vs indirect
1. Direct: minimize the color error of the projection.
2. Indirect: minimize the geometry error of the projection.

### Feature extraction
1. A feature point is often represented by an oriented planar texture patch.
2. Surf
3. Fast

### Three paradigms
1. (Extended) Kalman filter
2. Particle filter
3. Graph-based

### Bayes filter
1. Prediction step (motion model): bel(xt)’ = ∫p(xt | ut, xt-1) bel(xt-1)dxt-1
2. Correction step (sensor/observation model): bel(xt) = \phi p(zt | xt) bel(xt)

### (Extended) Kalman filter
1. A Bayes filter.
2. Optimal for linear models with Gaussian distributions.
3. Prediction step: xt (state) = At * xt-1 + Bt * ut (observation) + epsilon.
4. Correction step: zt (predicted observation) = Ct * xt + delta.
5. noise smoothing (improve noisy measurements) + state estimation (for state feedback) + recursive (computes next estimate using only most recent measurement).
6. Marginal and conditional of Gaussian are still Gaussian.
7. Extended: local linearilization (at the current best-estimated point) of non-linear functions. (The inverse operation is the bottleneck.)
8. Unscented: sampling techniques to find an approximated Gaussian distribution.

### Grid map
1. Discrete maps into cells (occupied or free space).
2. Non parametric model.
3. Assumptions: Cells are binary, static, and independent; Poses are known.
4. Binary bayes filter (for static state). Correction step only.

### Bundle adjustment
1. Levenberg-Marquardt (LM) algorithm.
## Indirect methods
### MonoSLAM
1. Used in a small volume (a room) in a long term.
2. First real-time monocular SLAM system.
3. Probabilistic filtering of a joint state consisting of camera and scene feature position estimates.

### PTAM
1. Separate tracking and mapping into two parallel threads.
2. Mapping is based on keyframes processed using batch techniques(bundle adjustment).
3. The map is densely initialized from a stereo pair (5-point algorithm).
4. New points are initialized by epipolar search.
5. A large number of points are mapped.
6. Bundle adjustment + robust n-point pose estimation.

### KinectFusion
1. PTAM + dense reconstruction
2. Projective TSDF (easy to parallelize but correct exactly only at the surface.
3. Moving average for surface (TSDF) update.
4. Surface measurement (V, N) -> Projective TSDF -> V, N -> pose (frame to model).

### DynamicFusion
1. Coarse 6D warp-field to model the dynamic motion.
2. Estimation of the volumetric model-to-frame warp field -> fusion of the live frame depth map into the canonical space -> adaption of the warp-field to capture new geometry.

### EKF-SLAM
1. Estimate pose and landmark locations (represented in the state space).
2. Assumption: known correspondences.

## Direct methods
### LSD-SLAM
1. Pose-graph of keyframes with semi-dense depth maps.
2. Filtering over a large number pixelwise small-baseline stereo comparisons.
3. Tracking with sim(3) (detecting scale-drift explicitly).
4. Initialized with a random depth map and large variance.
5. Re-weighted Gauss-Newton optimization.

### DSO (Direct Sparse Odometry)
1. Direct + Sparse
2. Points are well-distributed. Divide the image into 32x32 blocks and select one pixel inside each block with large gradient.

### DTAM (Dense Tracking and Mapping)
1. Incrementally construct cost volume and minimize energy for dense mapping.
2. Dense tracking.

### CodeSLAM
1. Use a sparse code to represent depth.
2. Linear depth decoder (no ReLU). Jacobian of the decoder w.r.t the code can be computed.

# Stereo
### Global optimization
1. SGM (Semi-global matching) / LoopyBP: Generalize there's an axact solution for a chain.
2. Graph cuts: generalize there's an exact solution if d has only two values.
3. TRW-S

### PMVS
1. Uniform coverage by finding top-k local maxima from each image block (32 x 32 blocks).
2. Matching -> expansion -> filtering -> polygonal surface reconstruction.

# Correspondence
### LIFT: Learned Invariant Feature Transform
1. Detector + orientation + descriptor

### GeoDesc
1. Integrate geometry constraints (based on GT surface normals).

### Learning Good Correspondences
1. Input a list of all possible matching pairs (coordinates only) and predict if each pair is valid or not.

## 3D correspondences
### PPFNet: Global Context Aware Local Features for Robust 3D Point Matching
1. Point pair features
2. PPF-FoldNet: rotation invariant

# Scene Understanding
## Scene completion
### Semantic Scene Completion from a Single Depth Image
1. SUNCG
2. TSDF input.

### Layered scene decomposition
1. Layered Scene Decomposition via the Occlusion-CRF
2. Layer-structured 3D Scene Inference via View Synthesis


# Image Synthesis / GAN
### Tricks
1. https://github.com/linxi159/GAN-training-tricks

# 3D Learning
## Point cloud
### Tengent Convolutions for Dense Prediction in 3D
1. Project nearby points onto the tangent plane.

### VoxelNet
1. Divide point cloud into voxels and process points inside each voxel using a PointNet.

## 2D-3D
### Deep Continuous Fusion for Multi-Sensor 3D Object Detection

## Graph
### FeaStNet
1. Learn the weight for each neighbor point (similarity).
2. Compute the weighted summation of features (non-local module).


