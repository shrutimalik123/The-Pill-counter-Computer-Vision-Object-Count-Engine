# 📸 The Pill Counter: Linear Regression Vision Engine

An interactive Supervised Learning simulation designed to teach **Simple Linear Regression**, **Ordinary Least Squares (OLS)**, and **Continuous Target Estimation** from scratch. You play as a Lead Pharmacy Automation and Robotics Engineer constructing an overhead computer vision camera module that automatically estimates the exact physical pill count on a sorting tray based on the total continuous pixel area detected.

## 🎓 Learning Objectives

This project focuses on teaching:
* **Simple Linear Regression:** Modeling a direct, continuous linear relationship between an independent feature ($X$) and a dependent numeric target ($y$).
* **Ordinary Least Squares (OLS):** The mathematical optimization routine used to find the best-fitting trendline by minimizing the sum of squared residuals.
* **Continuous Inference:** Generating scalar numerical outputs across an infinite range instead of sorting inputs into rigid, categorical classification bins.
* **Residual Analysis:** Evaluating model performance by examining vertical deviations between true observed training markers and geometric line predictions.

---

## ✨ Features

* **Computer Vision Automation Scenario:** Contextualizes geometric regression models within a real-world central-fill pharmacy automation and quality control framework.
* **Transparent OLS Calculations:** Breaks down the statistical calculation steps for both slope ($\beta_1$) and intercept ($\beta_0$) from scratch using standard loops.
* **Post-Inference Quantization:** Integrates continuous regression estimates cleanly with realistic physical constraints by rounding predictions to whole-pill increments.
* **Zero External Dependencies:** Built entirely with base Python logic and elementary algebraic expressions—no matrix libraries or heavy frameworks required.

---

## 🚀 How to Run the Game

### 1. Prerequisites
You only need **Python 3** installed.

### 2. Setup and Execution
1.  **Clone the Repo:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/pill-counter-linear-regression.git](https://github.com/YOUR_USERNAME/pill-counter-linear-regression.git)
    cd pill-counter-linear-regression
    ```
2.  **Save the Code:** Save the provided script as `pill_counter.py`.
3.  **Run the Script:**
    ```bash
    python pill_counter.py
    ```

### 3. Gameplay Instructions
1.  **Examine Calibration Runs:** Review the incoming calibration dataset showing how historical pixel blob areas correlate with hand-counted pill baselines.
2.  **Observe OLS Model Fitting:** Watch the system calculate sample means and fit an optimal line equation: $\text{Count} = (\text{Slope} \times \text{Area}) + \text{Intercept}$.
3.  **Process a Live Scan Intake:** Monitor the vision engine as the robot drops an unknown batch yielding a visual signature of $415.0$ pixel area units.
4.  **Verify Downstream Weight Match:** Check if the rounded continuous prediction passes the final scale verification check to clear the bottle for automated shipping.

---

## 🧠 Code Structure Highlights

### Ordinary Least Squares Fitting
The script loops through the reference arrays to compute deviations from the sample mean, establishing the slope gradient and intercept height.

```python
# OLS: Slope m = sum((x - mean_x) * (y - mean_y)) / sum((x - mean_x)^2)
numerator = 0.0
denominator = 0.0

for s in calibration_data:
    x_dev = s["pixel_area"] - mean_x
    y_dev = s["actual_count"] - mean_y
    numerator += x_dev * y_dev
    denominator += x_dev ** 2

m = numerator / denominator
b = mean_y - (m * mean_x)

