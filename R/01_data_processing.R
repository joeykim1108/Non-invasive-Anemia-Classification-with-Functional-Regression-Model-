#
# SCRIPT: 01_data_processing.R
#
# PURPOSE: This script handles the preprocessing of image data. Its primary
#          function is to convert RGB (Red, Green, Blue) color values from
#          fingernail images into the HSI (Hue, Saturation, Intensity) color space.
#          It also calculates average color values across images.
#
# AUTHOR: [Your Name]
# DATE: [Date of Last Edit]
#

# =============================================================================
#  1. RGB to HSI Conversion Function
# =============================================================================

#' Convert a matrix of RGB pixels to HSI.
#'
#' This function takes a matrix where each row is a pixel and columns are R, G, B
#' and converts the values to the HSI color space.
#'
#' @param rgb_matrix A numeric matrix with 3 columns (R, G, B).
#'        RGB values are assumed to be in the [0, 255] range.
#' @return A numeric matrix with 3 columns (Hue, Saturation, Intensity).
#'         - Hue is in degrees [0, 360].
#'         - Saturation is scaled to [0, 255].
#'         - Intensity is in the range [0, 255].

convert_rgb_to_hsi <- function(rgb_matrix) {
  # Ensure input is a matrix
  if (!is.matrix(rgb_matrix)) {
    rgb_matrix <- as.matrix(rgb_matrix)
  }
  
  # Vectorized calculation for faster processing
  R <- rgb_matrix[, 1]
  G <- rgb_matrix[, 2]
  B <- rgb_matrix[, 3]
  
  # Intensity (I)
  I <- (R + G + B) / 3
  
  # Saturation (S)
  # Suppress warnings for cases where R+G+B = 0 (black pixel)
  suppressWarnings({
    min_rgb <- pmin(R, G, B)
    S <- 1 - (3 / (R + G + B)) * min_rgb
  })
  S[is.nan(S)] <- 0 # Handle the case of black pixels (0/0 = NaN)
  S <- S * 255 # Scale to [0, 255]
  
  # Hue (H)
  numerator <- 0.5 * ((R - G) + (R - B))
  denominator <- sqrt((R - G)^2 + (R - B) * (G - B))
  
  # Suppress warnings for the acos calculation
  suppressWarnings({
    theta <- acos(numerator / denominator)
  })
  
  H <- theta * (180 / pi)
  H[G < B] <- 360 - H[G < B] # Adjust Hue based on B and G values
  H[denominator == 0] <- 0   # Set Hue to 0 if R=G=B (grayscale)
  
  # Combine into a single matrix
  hsi_matrix <- cbind(Hue = H, Saturation = S, Intensity = I)
  
  return(hsi_matrix)
}


# =============================================================================
#  2. Data Loading (Example Placeholder)
# =============================================================================
# In your actual script, you would load your raw data here.
# For this example, I am creating placeholder data structures that match
# the ones you seem to be using (lists of matrices).

# NOTE: Replace this section with your actual data loading (e.g., from .csv or .RData)
print("Loading placeholder data...")

# Placeholder for 51x51 data (list of 221 images)
# Each element of the list is a 51x51 matrix for that color channel.
num_images <- 221
data.51.R <- lapply(1:num_images, function(x) matrix(runif(51*51, 0, 255), 51, 51))
data.51.G <- lapply(1:num_images, function(x) matrix(runif(51*51, 0, 255), 51, 51))
data.51.B <- lapply(1:num_images, function(x) matrix(runif(51*51, 0, 255), 51, 51))

# Placeholder for other required data
# This assumes 'rgbhsi' is a data frame with metadata.
CBCHGB_level <- runif(num_images, 9, 15)
rgbhsi <- data.frame(
  HGb = CBCHGB_level,
  brightness = runif(num_images),
  shutterspeed = runif(num_images),
  exposure = runif(num_images),
  diagnosis = sample(0:1, num_images, replace = TRUE)
)
print("Placeholder data loaded.")


# =============================================================================
#  3. Process 51x51 Image Data
# =============================================================================
# This section processes each 51x51 image, converting it from RGB to HSI.

# Initialize lists to store the results
data.51.hue <- list()
data.51.saturation <- list()
data.51.intensity <- list()

for (i in 1:length(data.51.R)) {
  # Combine the R, G, B matrices for the i-th image into a 3-column matrix
  # Each row represents a pixel: [R, G, B]
  rgb_pixels <- cbind(as.vector(data.51.R[[i]]),
                      as.vector(data.51.G[[i]]),
                      as.vector(data.51.B[[i]]))
  
  # Convert the entire image's pixels at once
  hsi_pixels <- convert_rgb_to_hsi(rgb_pixels)
  
  # Reshape the HSI vectors back into 51x51 matrices and store them
  data.51.hue[[i]] <- matrix(hsi_pixels[, "Hue"], nrow = 51, ncol = 51)
  data.51.saturation[[i]] <- matrix(hsi_pixels[, "Saturation"], nrow = 51, ncol = 51)
  data.51.intensity[[i]] <- matrix(hsi_pixels[, "Intensity"], nrow = 51, ncol = 51)
}

print("51x51 image data converted to HSI.")


# =============================================================================
#  4. Calculate Average HSI Values per Image
# =============================================================================
# Use sapply for a more concise way to apply the 'mean' function to each list element

data.avghue <- sapply(data.51.hue, mean, na.rm = TRUE)
data.avgsat <- sapply(data.51.saturation, mean, na.rm = TRUE)
data.avgint <- sapply(data.51.intensity, mean, na.rm = TRUE)

# Create a final data frame with average HSI values and Hemoglobin levels
df_hsi_avg <- data.frame(
  hue = data.avghue,
  saturation = data.avgsat,
  intensity = data.avgint,
  HGb = CBCHGB_level # Assuming CBCHGB_level is loaded and available
)

print("Average HSI values calculated.")
head(df_hsi_avg)

# =============================================================================
#  5. Save Processed Data
# =============================================================================
# It's good practice to save the processed data so you don't have to
# re-run this script every time you do analysis.

# save(data.51.hue, data.51.saturation, data.51.intensity, df_hsi_avg, rgbhsi,
#      file = "data/processed_hsi_data.RData")

print("Data processing script finished.")