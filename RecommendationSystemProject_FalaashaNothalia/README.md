# 🛍️ Visual Product Recommendation System

> **Image-based fashion product discovery using Deep Learning, Transfer Learning, and Siamese Networks**

A deep-learning powered **visual product recommendation system** that retrieves visually similar fashion products from a catalog using an input image — **without requiring text-based search**.

The project progressively develops and evaluates three recommendation approaches:

**ResNet50 Baseline → Fine-Tuned CNN → Siamese Network with Triplet Loss**

The final system converts product images into learned visual embeddings and uses **cosine similarity** to retrieve the Top-K most similar products.

---

## ✨ What Does This Project Do?

Imagine uploading a picture of a shirt, pair of shoes, or other fashion product and asking:

> *"Show me products that look similar to this."*

Instead of relying on keywords such as *"blue shirt"* or *"casual shoes"*, the system learns a visual representation of the image and searches the product catalog in **embedding space**.

### Core Pipeline

```text
             Input Product Image
                      │
                      ▼
             Image Preprocessing
                 224 × 224
                      │
                      ▼
             ResNet50 Feature
                Extraction
                      │
                      ▼
             Learned Embedding
                  128-D
                      │
                      ▼
             Cosine Similarity
                      │
                      ▼
              Top-K Retrieval
                      │
                      ▼
          Visually Similar Products
```

---

# 🎯 Project Objectives

The system is designed to:

- Accept a fashion product image as input
- Extract meaningful deep visual features
- Represent products as compact embeddings
- Retrieve visually similar products
- Compare conventional pretrained features with adapted representations
- Improve similarity learning using a Siamese/triplet-learning approach
- Evaluate retrieval quality quantitatively and qualitatively
- Provide an interactive Streamlit interface for image-based search

---

# 🧠 Three-Stage Recommendation Architecture

The notebook deliberately builds the system in three stages so that the effect of each improvement can be evaluated.

## 1️⃣ Baseline — Pretrained ResNet50

A pretrained **ResNet50** is used as a feature extractor.

The classification head is removed and the resulting feature representation is used as the product embedding.

Similarity is then computed using cosine distance.

```text
Product Image
     ↓
ResNet50
     ↓
Deep Feature Embedding
     ↓
Cosine Similarity
     ↓
Top-K Products
```

This establishes the baseline against which the later approaches are compared.

---

## 2️⃣ Fine-Tuned CNN

The second stage adapts the pretrained ResNet50 to the selected fashion categories.

Most of the pretrained backbone remains frozen while the final layers are fine-tuned.

```text
Pretrained ResNet50
        ↓
Freeze Most Layers
        ↓
Fine-Tune Final Layers
        ↓
Fashion-Specific Features
        ↓
Similarity Search
```

This allows the feature representation to become more relevant to the selected product domain.

---

## 3️⃣ Siamese Network + Triplet Loss ⭐

The final stage focuses directly on **similarity learning**.

Instead of simply adapting a classifier, the network learns an embedding space where similar products are closer together and dissimilar products are farther apart.

Training uses:

```text
Anchor
   │
   ├── Positive → Same Category
   │
   └── Negative → Different Category
```

The shared embedding network produces representations for these images, and **triplet loss** optimizes the embedding space.

Conceptually:

```text
             ┌──────────────┐
Anchor ─────►│              │
Positive ───►│ Shared CNN   │──► Embeddings
Negative ───►│              │
             └──────────────┘
                    │
                    ▼
              Triplet Loss
                    │
                    ▼
       Better Similarity Representation
```

The final embedding dimension used by the recommendation pipeline is **128**.

---

# 📦 Dataset

The project uses the **Fashion Product Images (Small)** dataset containing:

- `styles.csv` — product metadata
- `images/` — corresponding product images

The source dataset contains approximately **44K product records**.

To keep training efficient and balanced, the notebook creates a controlled subset rather than using the entire dataset.

### Selected Categories

The six highest-frequency `articleType` categories used in the notebook are:

| Category | Images |
|---|---:|
| Tshirts | 300 |
| Shirts | 300 |
| Casual Shoes | 300 |
| Watches | 300 |
| Sports Shoes | 300 |
| Kurtas | 300 |
| **Total** | **1,800** |

Images whose corresponding files are missing are removed before processing.

This produces a balanced catalog of **1,800 product images**.

---

# 🖼️ Image Preprocessing

Images are processed for compatibility with ResNet50.

### Configuration

```text
Image Size:       224 × 224
Color Processing: RGB
Normalization:    ResNet50 / ImageNet preprocessing
```

The same preprocessing pipeline is applied during both catalog embedding generation and query-image inference.

---

# 🔎 Similarity Search

Once the product catalog has been embedded, recommendations do not require retraining.

The process is:

```text
Query Image
    ↓
Embedding Model
    ↓
128-D Query Embedding
    ↓
Compare with Catalog Embeddings
    ↓
Cosine Distance
    ↓
Nearest Products
    ↓
Top-K Recommendations
```

The notebook uses **NearestNeighbors with cosine distance** for retrieval.

Precomputed catalog embeddings make inference considerably more efficient because the entire catalog does not need to be embedded again for every query.

---

# 📊 Evaluation

The project does not rely only on visual inspection.

The recommendation approaches are evaluated using:

### Precision@K

Measures the proportion of retrieved products that are considered relevant among the Top-K results.

\[
Precision@K =
\frac{\text{Relevant items in Top-K}}{K}
\]

### Recall@K

Measures how many of the available relevant products are retrieved.

\[
Recall@K =
\frac{\text{Relevant items retrieved}}{\text{Total relevant items}}
\]

### Qualitative Evaluation

The notebook also compares recommendation outputs visually across:

- Baseline ResNet50
- Fine-tuned CNN
- Siamese embedding model

### Performance Evaluation

Inference and embedding-generation time are also considered.

This provides both **quantitative and qualitative evidence** for comparing the approaches.

---

# 🖥️ Interactive Streamlit Application

The notebook also contains the deployment component for an interactive recommendation interface.

The Streamlit application provides:

- 📤 Product image upload
- 🔢 Configurable Top-K recommendations
- 🧠 Siamese embedding generation
- 🔍 Cosine-similarity based retrieval
- 🖼️ Product visualization
- 📈 Similarity scores
- 🏷️ Product metadata such as category, gender and colour

### Application Flow

```text
Upload Image
     ↓
Preprocess Image
     ↓
Load Learned Embedding Model
     ↓
Generate 128-D Embedding
     ↓
Search Precomputed Catalog
     ↓
Retrieve Top-K Products
     ↓
Display Recommendations
```

The application can be launched from the Kaggle notebook using Streamlit and exposed through a temporary public tunnel for demonstration.

---

# 🗂️ Important Files

The recommendation application relies on the following generated artifacts:

```text
siamese_embedding_branch.weights.h5
        │
        └── Learned Siamese embedding model weights

catalog_metadata.csv
        │
        └── Product metadata and image paths

catalog_embeddings.npy
        │
        └── Precomputed catalog embeddings

app.py
        │
        └── Streamlit recommendation interface
```

---

# 🧰 Technology Stack

| Component | Technology |
|---|---|
| Programming | Python |
| Deep Learning | TensorFlow / Keras |
| CNN Backbone | ResNet50 |
| Similarity Learning | Siamese / Triplet Learning |
| Loss Function | Triplet Loss |
| Numerical Computing | NumPy |
| Data Processing | Pandas |
| Image Processing | OpenCV / PIL |
| Similarity Search | Scikit-learn |
| Visualization | Matplotlib |
| Application UI | Streamlit |
| Environment | Kaggle Notebook |
| Dataset | Fashion Product Images (Small) |

---

# 📓 Notebook Structure

The Kaggle notebook follows this general sequence:

```text
01. Setup
      ↓
02. Dataset Preparation
      ↓
03. Dataset Exploration
      ↓
04. Image Preprocessing
      ↓
05. ResNet50 Baseline
      ↓
06. Baseline Similarity Search
      ↓
07. Transfer Learning / Fine-Tuning
      ↓
08. Siamese Network
      ↓
09. Triplet-Loss Training
      ↓
10. Embedding Generation
      ↓
11. Recommendation Retrieval
      ↓
12. Precision@K / Recall@K
      ↓
13. Model Comparison
      ↓
14. Save Model + Embeddings
      ↓
15. Streamlit Application
```

---

# 🚀 How to Run

## Option 1 — Run the Kaggle Notebook

1. Open the notebook in Kaggle.
2. Ensure the **Fashion Product Images (Small)** dataset is attached.
3. Enable GPU acceleration for training.
4. Run the notebook cells sequentially.
5. Allow the baseline, fine-tuning and Siamese stages to complete.
6. The notebook generates the model weights and catalog embeddings.
7. Run the Streamlit deployment cells.
8. Open the generated public application URL.

---

## Option 2 — Use the Generated Application Artifacts

The Streamlit application requires:

```text
app.py
siamese_embedding_branch.weights.h5
catalog_metadata.csv
catalog_embeddings.npy
```

The image paths stored in the metadata must also point to the corresponding product images.

---

# ⚙️ Configuration

Important notebook parameters include:

```python
SEED = 42

IMG_SIZE = (224, 224)

TOP_N_CATEGORIES = 6

IMAGES_PER_CATEGORY = 300
```

The resulting dataset therefore contains approximately:

```text
6 categories × 300 images
= 1,800 images
```

---

# 💡 Why Use a Subset?

The original Fashion Product Images dataset contains tens of thousands of products and many categories with relatively few samples.

For this project, the goal is not to train on the entire commercial catalog.

Instead, a balanced subset makes it possible to:

- Keep experiments manageable on a Kaggle GPU
- Reduce Siamese pair/triplet generation cost
- Avoid severe category imbalance
- Make model comparison faster
- Demonstrate the complete recommendation methodology clearly

The subset therefore represents a controlled experimental catalog rather than the complete dataset.

---

# 📌 Important Evaluation Note

The quantitative retrieval evaluation defines relevance using the product's **`articleType` category**.

Therefore, Precision@K and Recall@K measure **category-level retrieval relevance**, while the visual comparisons provide an additional qualitative assessment of actual visual similarity.

This distinction is important because two products from the same category may not necessarily look identical.

---

# 🔬 Key Learning Outcomes

This project demonstrates the complete workflow of a deep-learning recommendation system:

- Dataset selection and controlled subsetting
- Image preprocessing
- Transfer learning
- CNN feature extraction
- Embedding generation
- Cosine similarity search
- Nearest-neighbour retrieval
- Siamese/triplet similarity learning
- Quantitative retrieval evaluation
- Qualitative model comparison
- Precomputed embedding-based inference
- Interactive deployment using Streamlit

---

# 🌟 Final System

The completed system combines:

**Computer Vision + Deep Learning + Metric Learning + Similarity Search + Interactive Deployment**

into a single image-based recommendation pipeline.

```text
                         ┌─────────────────────┐
                         │   Fashion Dataset   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Balanced Subset     │
                         │ 1,800 Images        │
                         └──────────┬──────────┘
                                    │
                  ┌─────────────────┼─────────────────┐
                  │                 │                 │
                  ▼                 ▼                 ▼
             ResNet50         Fine-Tuned CNN      Siamese
             Baseline                              Network
                  │                 │                 │
                  └─────────────────┼─────────────────┘
                                    │
                                    ▼
                           128-D Embeddings
                                    │
                                    ▼
                           Cosine Similarity
                                    │
                                    ▼
                              Top-K Results
                                    │
                                    ▼
                           Streamlit Interface
```

---

## 📚 Reference

The project is based on the project brief specifying an image-based recommendation engine using pretrained CNN embeddings, cosine similarity/FAISS retrieval, transfer learning, a Siamese network, dataset subsetting, and Precision@K / Recall@K evaluation.

---

### 👩‍💻 Project Status

**Core recommendation pipeline:** ✅ Complete  
**Baseline model:** ✅ Complete  
**Transfer learning:** ✅ Complete  
**Siamese / triplet learning:** ✅ Complete  
**Embedding-based retrieval:** ✅ Complete  
**Quantitative evaluation:** ✅ Complete  
**Interactive Streamlit application:** ✅ Implemented  
**Kaggle demonstration:** ✅ Supported

> **A complete visual recommendation pipeline — from raw fashion images to an interactive image-search experience.**