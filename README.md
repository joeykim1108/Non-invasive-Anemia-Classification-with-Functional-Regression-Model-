# Non-invasive Anemia Classification Using Smartphone Fingernail Images

[![Language](https://img.shields.io/badge/Language-R-blue.svg)](https://www.r-project.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the R code and methodology for the Master's thesis, "Non-invasive Anemia Classification with Functional Regression Model Leveraging Smartphone Fingernail Images."

## Abstract

Anemia is a highly prevalent disease characterized by a low number of circulating red blood cells. Current testing procedures are often invasive, posing risks to vulnerable groups. This project explores a non-invasive diagnostic method that uses smartphone images of fingernails, where color is primarily influenced by blood hemoglobin. We developed a functional data regression model that analyzes 51x51 pixel RGB matrices from fingernail photos of 246 subjects. By treating the image pixels as two-dimensional functions, our model preserves the spatial structure of the image to maximize predictive power. The proposed model demonstrates strong predictive performance, achieving an Area Under the Curve (AUC) of **0.947** with RGB data and **0.952** with HSI data.

## The Problem

Standard anemia tests, like a complete blood count, require drawing blood. This invasive procedure presents challenges, especially for infants, where it can be linked to neuro-developmental issues, and for individuals in low-resource settings. There is a clear need for a reliable, accessible, and non-invasive diagnostic tool.

## Our Approach

This project leverages the fact that fingernail beds lack melanocytes, meaning their color is a direct representation of underlying blood capillaries. The core of this work is a **functional regression model** which offers a significant advantage over simple color averaging.

1.  **Data Collection:** A prototype smartphone app collected fingernail images from 246 subjects at Emory University Hospital.
2.  **Image Processing:** Each image was processed into 51x51 pixel matrices for its Red, Green, and Blue (RGB) color channels. These were also converted to the Hue, Saturation, and Intensity (HSI) color space.
3.  **Functional Data Analysis (FDA):** Instead of treating each pixel as an independent variable, we model the entire 51x51 grid as a continuous 2D function. This retains the image's structural and spatial information.
4.  **Prediction:** We use Functional Principal Component Analysis (FPCA) to reduce the dimensionality of the image functions and feed the resulting scores into a logistic regression model to classify a patient as anemic or non-anemic.

## Key Results

The model's performance was evaluated using the Area Under the Curve (AUC) of the Receiver Operating Characteristic (ROC). An AUC of 1.0 represents a perfect classifier.

| Data Type | Area Under the Curve (AUC) |
| :-------- | :------------------------- |
| **RGB**   | 0.947                      |
| **HSI**   | **0.952**                  |

These results indicate a high degree of accuracy and validate the potential of this method as a clinical screening tool.

## Repository Structure
.
├── R/
│ ├── 01_data_processing.R # Script for RGB to HSI conversion and data cleaning
│ └── 02_fda_analysis.R # Script for FPCA, regression modeling, and evaluation
│
├── data/
│ └── (Raw data not included due to patient privacy)
│
├── output/
│ └── (Folder for generated plots and results)
│
└── README.md # This file


## How to Reproduce the Analysis

To run this analysis, you will need R and the following packages.

1.  **Clone the repository:**
    ```bash
    git clone [URL to your repository]
    ```

2.  **Install required R packages:**
    ```R
    install.packages(c("fda", "funData", "MFPCA", "caret", "pROC", "dplyr", "plotly"))
    ```

3.  **Run the R scripts:**
    Open the R project and run the scripts in the `R/` folder in the following order:
    *   `01_data_processing.R`: This script loads the raw data, performs the color space conversions, and prepares the data for analysis.
    *   `02_fda_analysis.R`: This script performs the functional data analysis, trains the logistic regression models, and evaluates their performance.

    *(Note: Since the original data is private, the scripts have been written with placeholder data to demonstrate the workflow.)*

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Author & Acknowledgments

*   **Author:** [Jospeh Kim]
*   This work was completed as part of a Master of Science thesis at Yonsei University. I would like to thank my advisors and the staff at Emory University Hospital for their invaluable support.
