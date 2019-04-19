# SLAM
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
3. Prediction step: xt (state) = At * xt-1 + Bt * ut (observation) + epsilon
4. Correction step: zt (predicted observation) = Ct * xt + delta
5. noise smoothing (improve noisy measurements) + state estimation (for state feedback) + recursive (computes next estimate using only most recent measurement)
6. Marginal and conditional of Gaussian are still Gaussian.
7. Extended: local linearilization (at the current best-estimated point) of non-linear functions. (The inverse operation is the bottleneck.)
8. Unscented: sampling techniques to find an approximated Gaussian distribution.

### Grid map
1. Discrete maps into cells (occupied or free space).
2. Non parametric model.
3. Assumptions: Cells are binary, static, and independent; Poses are known.
4. Binary bayes filter (for static state). Correction step only.

## Indirect methods
### MonoSLAM
1. Used in a small volume (a room) in a long term.
2. First real-time monocular SLAM system.
3. Probabilistic filtering of a joint state consisting of camera and scene feature position estimates

### PTAM
1. Separate tracking and mapping into two parallel threads.
2. Mapping is based on keyframes processed using batch techniques(bundle adjustment).
3. The map is densely initialized from a stereo pair (5-point algorithm).
4. New points are initialized by epipolar search.
5. A large number of points are mapped.
6. Bundle adjustment + robust n-point pose estimation

### KinectFusion
1. PTAM + dense reconstruction
2. Projective TSDF (easy to parallelize but correct exactly only at the surface.
3. Moving average for surface (TSDF) update.
4. Surface measurement (V, N) -> Projective TSDF -> V, N -> pose (frame to model)

### EKF-SLAM
1. Estimate pose and landmark locations (represented in the state space).
2. Assumption: known correspondences

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

