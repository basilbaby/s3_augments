# Augmenter

A Flask-based web application that provides preprocessing and augmentation capabilities for both text and image files.

## Features

### Text Processing
1. **Preprocessing**
   - Whitespace Removal
   - Lowercase Conversion
   - Standardizes text formatting

2. **Augmentation**
   - Synonym Replacement (up to 3 words)
   - Maintains original word casing
   - Preserves text structure

### Image Processing
1. **Preprocessing (Denoising)**
   - Noise Reduction
   - Edge Enhancement
   - Gaussian Blur
   - Image Smoothing

2. **Augmentation**
   - Color Jittering
   - Brightness Adjustment (±50%)
   - Contrast Enhancement (±80%)
   - Saturation Modification (±70%)
   - Hue Variation (±10%)
   - Image Sharpening

## Installation

1. Clone the repository:
