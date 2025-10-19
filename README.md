# Face Recognition Methods Survey

A comprehensive single-file demonstration implementing multiple face recognition approaches using the Labeled Faces in the Wild (LFW) dataset.

## Overview

This project showcases various face recognition techniques, ranging from traditional keypoint-based methods to modern holistic and hybrid approaches. It provides a complete pipeline from face detection to recognition using multiple algorithms.

## Features

The implementation includes **7 different face recognition methods**:

### 1. Local Keypoints-Based (ORB)
- Uses ORB (Oriented FAST and Rotated BRIEF) detector
- Extracts and matches keypoint descriptors
- Brute-force matching with ratio test

### 2. Local Appearance-Based (LBP)
- Local Binary Patterns with histogram pooling
- Grid-based feature extraction
- SVM classifier for recognition

### 3. Local Appearance-Based (HOG)
- Histogram of Oriented Gradients
- Captures edge and gradient structure
- SVM classifier for recognition

### 4. Holistic Linear (PCA/Eigenfaces)
- Principal Component Analysis for dimensionality reduction
- k-Nearest Neighbors classifier
- Classic eigenfaces approach

### 5. Holistic Linear (LDA/Fisherfaces)
- Linear Discriminant Analysis
- Supervised dimensionality reduction
- k-Nearest Neighbors classifier

### 6. Holistic Non-Linear (Kernel PCA + SVM)
- Kernel PCA with RBF kernel
- Non-linear feature extraction
- SVM classifier with RBF kernel

### 7. Hybrid Approach
- Combines LBP local features with PCA holistic features
- Best of both worlds: texture and global structure
- SVM classifier for final prediction

## Requirements

Install dependencies using pip:

```bash
pip install numpy opencv-python scikit-image scikit-learn matplotlib
```

### Library Versions
- `numpy`: Array operations and numerical computing
- `opencv-python`: Face detection (Haar cascade) and image processing
- `scikit-image`: LBP and HOG feature extraction
- `scikit-learn`: ML algorithms (PCA, LDA, SVM, k-NN) and LFW dataset
- `matplotlib`: Visualization (optional)

## Dataset

The script uses the **Labeled Faces in the Wild (LFW)** dataset, which is automatically downloaded via scikit-learn:
- Contains thousands of face images of public figures
- Pre-aligned and cropped faces
- Configurable minimum faces per person (default: 100)

## Usage

### Basic Usage

```python
# Run the complete pipeline
python face_methods_survey.py
```

### Step-by-Step Usage

```python
# 1. Load the dataset
imgs, labels, names = load_lfw_dataset(min_faces_per_person=100, resize=0.5)

# 2. Train all models
models = train_all_models(imgs, labels, names)

# 3. Test on a new image
test_img_path = "path/to/your/image.jpg"
results = predict_all_methods(models, test_img_path)
```

### Custom Image Testing

Update the `test_img_path` variable in the main block:

```python
test_img_path = "path/to/your/test_image.jpg"
```

## Pipeline Architecture

Each method follows this pipeline:

```
Input Image → Face Detection (Haar Cascade) → Feature Extraction → Classification → Prediction
```

### Face Detection
- OpenCV Haar Cascade classifier
- Detects and crops the largest face
- Resizes to 128x128 pixels for consistency

### Feature Extraction
Method-specific feature extraction (ORB, LBP, HOG, PCA, etc.)

### Classification
- k-NN for PCA and LDA methods
- SVM for appearance-based and hybrid methods
- Descriptor matching for ORB method

## Output

The script provides detailed predictions from all methods:

```
[ORB] Matched label idx 5  matches=42 => Person_Name
[LBP-SVM] Pred: Person_Name  prob=0.856
[HOG-SVM] Pred: Person_Name  prob=0.923
[PCA-kNN] Pred: Person_Name
[LDA-kNN] Pred: Person_Name
[KPCA-SVM] Pred: Person_Name  prob=0.891
[Hybrid LBP+PCA SVM] Pred: Person_Name  prob=0.934
```

## Results Dictionary Structure

```python
{
    'orb': {'label_index': int, 'matches': int},
    'lbp': {'label_index': int, 'prob': float},
    'hog': {'label_index': int, 'prob': float},
    'pca': {'label_index': int},
    'lda': {'label_index': int},
    'kpca_svm': {'label_index': int, 'prob': float},
    'hybrid': {'label_index': int, 'prob': float}
}
```

## Method Comparison

| Method | Type | Advantages | Use Cases |
|--------|------|------------|-----------|
| ORB | Keypoints | Robust to rotation/scale | Partial faces, occlusions |
| LBP | Local texture | Fast, illumination-invariant | Real-time applications |
| HOG | Local gradient | Good for edges/shapes | General face recognition |
| PCA | Holistic linear | Simple, fast | Controlled conditions |
| LDA | Holistic linear | Supervised, discriminative | Good labeled data |
| Kernel PCA | Holistic non-linear | Captures complex patterns | Complex variations |
| Hybrid | Combined | Best overall performance | Production systems |

## Configuration Options

### Dataset Parameters
```python
load_lfw_dataset(
    min_faces_per_person=100,  # Minimum images per identity
    resize=0.5                  # Image resize factor
)
```

### Feature Extraction Parameters
```python
# LBP
extract_lbp_hist(face, P=8, R=1, grid_x=8, grid_y=8)

# HOG
extract_hog_descriptor(face, pixels_per_cell=(16,16), orientations=9)

# PCA
train_pca_classifier(X_train, y_train, n_components=150)
```

## Performance Notes

- **Training time**: 2-5 minutes on standard hardware
- **Inference time**: < 1 second per image for all methods
- **Memory usage**: ~500MB-1GB depending on dataset size
- **Accuracy**: Varies by method (typically 70-95% on LFW)

## Troubleshooting

### No face detected
- Ensure good lighting and frontal face pose
- Try adjusting `min_size` parameter in `detect_face_opencv()`

### Low accuracy
- Increase `min_faces_per_person` for more training data
- Adjust classifier parameters (n_neighbors, C, gamma)

### Memory issues
- Reduce dataset size with higher `min_faces_per_person`
- Lower PCA/KPCA `n_components`

## Use in Google Colab

This script is Colab-friendly. Simply:
1. Upload the script to Colab
2. Upload your test image
3. Update `test_img_path` to match your uploaded image path
4. Run all cells

## License

This code is provided for educational and research purposes. The LFW dataset has its own usage terms.

## References

- LFW Dataset: http://vis-www.cs.umass.edu/lfw/
- Eigenfaces: Turk & Pentland (1991)
- Fisherfaces: Belhumeur et al. (1997)
- LBP: Ojala et al. (2002)
- HOG: Dalal & Triggs (2005)

## Author

Face recognition methods survey demonstration script.

## Contributing

Feel free to extend this implementation with additional methods or improvements!
