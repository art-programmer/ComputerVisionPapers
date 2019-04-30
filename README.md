# Semantic Segmentation
### Conditional Random Fields as Recurrent Neural Networks
### ReSeg: A recurrent Neural Network-based Model for Semantic Segmentation

## Co-segmentation
### CoSegNet: Deep Co-Segmentation of 3D Shapes with Group Consistency Loss
1. Propose novel group consistency loss for unsupervised part-segmentation. Use inconsistent dataset to train a shape refiner.

# Instance Segmentation / Object Detection
## Conventional methods
### Normalized cut
1. Cut / Assoc(A) + Cut / Assoc(B). Cut, Assoc(A), and Assoc(B) are summation of edge weights.


## Proposal-based methods
### ron reverse connection with objectness prior networks
1. Multi-scale + reverse connection

### Frustum PointNets for 3D Object Detection from RGB-D Data
1. 2D proposals + PointNet

### Reinforcement learning to choose proposals
### Cascade Object Detection with Deformable Part Models
### SSD: Single Shot MultiBox Detector


## Proposal-free methods


## Graph-based methods
### Iterative Visual Reasoning Beyond Convolutions
### Semantic Object Parsing with Graph LSTM

### Semi-convolutional operators
1. Instance coloring.
2. y = phi(x) + (u, v)
3. min(y - cy). Attractive force only.

### Bottom-up Instance Segmentation using Deep Higher-Order CRFs



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

### Fusion++: Volumetric Object-Level SLAM
1. Object-based map representation. Use Mask R-CNN to predict object-level TSDF for initialization.
2. Predict foreground probability for rendering.


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

### Manhattan-world Stereo
1. Dominant axes + hypothesis planes + optimization
# Correspondence

### LIFT: Learned Invariant Feature Transform
1. Detector + orientation + descriptor

### GeoDesc
1. Integrate geometry constraints (based on GT surface normals).

### Learning Good Correspondences
1. Input a list of all possible matching pairs (coordinates only) and predict if each pair is valid or not.

### NetVLAD: CNN Architecture for Weakly Supervised Place Recognition
1. V(k) = sum(w_i(x_i - c_k), where K is the number of clusters.

### Efficient Deep Learning for Stereo Matching
1. Inner product layer at the end.
2. Classify possible disparities.


## 3D correspondences
### PPFNet: Global Context Aware Local Features for Robust 3D Point Matching
1. Point pair features
2. PPF-FoldNet: rotation invariant



# Scene Understanding
### Naural Scene De-rendering
1. Reinforce algorithm based on rendered images
### Learning to Parse Wireframes in Images of Man-Made Environments
1. Junctions + post-processing
### Im2Pano3D
1. Single image -> 360 degree panorama.
### Single-View 3D Scene Parsing by Attributed Grammar
### Indoor Segmentation and Support Inference from RGBD Images


## Scene completion
### Semantic Scene Completion from a Single Depth Image
1. SUNCG
2. TSDF input.

### ScanComplete: Large-Scale Scene Completion and Semantic Segmentation for 3D Scans
### Layered scene decomposition
1. Layered Scene Decomposition via the Occlusion-CRF
2. Layer-structured 3D Scene Inference via View Synthesis


## Single-image Reconstruction
### Multi-view Consistency as Supervisory Signal for Learning Shape and Pose Prediction
### Factoring Shape, Pose, and Layout from the 2D Image of a 3D Scene
### Automatic Photo Pop-up
### Unfolding an Indoor Origami World
### Perspective Transformer Nets: Learning Single-View 3D Object Reconstruction without 3D Supervision
1. Image -> voxel -> projection (supervision)

### Pixels, Voxels, and Views: A Study of Shape Representations for Single View 3D Object Shape Prediction
1. Multi-surface generalizes better than voxel-based representations. It also looks better (high resolution). It can also capture some thin structures, though its post-processing step (surface reconstruction) might discard them.
2. Viewer-centered gneralizes better than object-centered. It has good shape prediction but poor pose prediction. Object-centered tend to memorizes the observed meshes, and its learned features can be used for object recognition.
3. The model trained to predict shape and pose can be finetuned for object recognition. Maybe it will generalize better.


## Depth estimation
### SURGE: Surface Regularized Geometry Estimation from a Single Image
1. CNN + DenseCRF

### Monocular Depth Estimation using Neural Regression Forest



# Machine Learning
## Architectures
### Neural Module Networks


## Image Synthesis / GAN
### Tricks
1. https://github.com/linxi159/GAN-training-tricks

### Adversarial Generator-Encoder Networks
### View Synthesis by Appearance Flow
### DRAW: A Recurrent Neural Network for Image Generation


## Transfer learning
### Cross Model Distillation for Supervision Transfer
1. Similarity loss between internal features.
2. Paired images of the same scene with different modalities.

### Generic 3D Representation via Pose Estimation and Matching
1. Porxy 3D tasks: object-centric camera pose estimation and wide baseline feature matching.

### Unsupervised Domain Adaption by Backpropagation
1. Domain confusion.


## Unsupervised learning
### Unsupervised Visual Representation Learning by Context Prediction.
1. Predict relative location between patches.


## One-shot learning
### Learning Feed-Forward One-Shot Learners
1. Train a learnet to predict the model parameters.
2. Assess the model on another exemplar to predict if the new exemplar is of the same class with the one used by the learnet.


## Graph
### GraphGAN: Generating Graphs via Random Walk
### Defromable Graph Matching


## Attention
### Look Closer to See Better: Recurrent Attention Convolutional Neural Network for Fine-grained Image Recognition
### Recurrent Models of Visual Attention
### Show, Attend and Tell: Neural Image Caption Generation with Visual Attention


## Visualization
### Understanding Deep Image Representations by Inverting Them
1. Similar to DeepDream.

### Object Detectors Emerge in Deep Scene CNNs
1. Simplifying the input images.
2. Visualizing the receptive fields of units and their activation patterns.
3. Indentifying the semantics of internal units.

### Deep Convolutional Inverse Graphics Network
1. AutoEncoder. The latent code is divided into segments.
2. Only one attribute changes in each mini-batch.



# 3D Learning
## Point cloud
### Tengent Convolutions for Dense Prediction in 3D
1. Project nearby points onto the tangent plane.

### VoxelNet
1. Divide point cloud into voxels and process points inside each voxel using a PointNet.

### Point Convolutional Neural Networks by Extension Operators
1. Apply extension operators to convert point cloud to volumetric representations (using basis functions).
2. Process the volumetric representation and sample back to point cloud.

### PointCNN
1. K-neareat neighbor. Lift each neighbor into new feature, and concatenate lifted features with the current one.
2. Learn a KxK transformation matrix to permute, and a standard convolution to process.

### Recurrent Slice Networks for 3D Segmentation on Point Clouds
1. Divide the point cloud into slices and use recurrent network to process slices sequentially.

### Attentional ShapeContextNet for Point Cloud Recognition
1. Non-local modules.

### Escape from Cells: Deep Kd-Networks for the Recognition of 3D Point Cloud Models


## Voxels
### CSGNet
1. Rendering + RL


## 2D-3D
### Deep Continuous Fusion for Multi-Sensor 3D Object Detection


## Graph
### FeaStNet
1. Learn the weight for each neighbor point (similarity).
2. Compute the weighted summation of features (non-local module).

### Scan2Mesh: From Unstructured Range Scans to 3D Meshes
1. Predict 100 vertices (set generation), read features from voxel grids, and use graph neural network to predict edges.
2. Find face candidates in the dual graph, and use graph network to predict face existence.
3. Generate training data using mesh simplification (https://github.com/kmammou/v-hacd).

## Boxes
### GRASS
1. Recursive network (merging two parts into one node).
2. Train an autoencoder and map the context feature to the latent representation for decoding.

### Learning Shape Abstractions by Assembling Volumetric Primitives
1. Voxel -> boxes
2. Consistency loss + coverage loss. Reinforce algorithm to allow an arbitrary number of primitives.



# Tracking / Localization
### Detect to Track and Track to Detect
### Lost Shopping! Monocular Localization in Large Indoor Spaces
### Learning Transformation Synchronization
1. Predict weights between every pair of views, which are used for estimating the absolute pose.
2. Repeat the process iteratively.



# Human Pose Estimation
### Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields
### Lifting from the Deep: Convolutional 3D Pose Estimation from a Single Image
### Structured Feature Learning for Pose Estimation
### End-to-End Learning of Deformable Mixture of Parts and Deep Convolutional Neural Networks for Human Pose Estimation
1. Message passing
