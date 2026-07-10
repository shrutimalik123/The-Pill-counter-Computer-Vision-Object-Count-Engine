def linear_regression_counter_game():
    # 1. Scenario: Computer Vision Tray Verification
    print("--- 📸 THE PILL COUNTER: LINEAR REGRESSION VISION ENGINE 📸 ---")
    print("Mission: Estimate the exact number of pills on a tray from visual pixel area.")
    print("Goal: Apply Ordinary Least Squares (OLS) to calculate an optimal fitting line.")

    # 2. Historical Calibration Matrix (Continuous Feature to Continuous Target)
    # Feature (X): Sensed Pixel Area (arbitrary units)
    # Target (y): Actual Number of Physical Tablets counted by hand
    calibration_data = [
        {"pixel_area": 120.0, "actual_count": 10.0},
        {"pixel_area": 240.0, "actual_count": 20.0},
        {"pixel_area": 370.0, "actual_count": 30.0},
        {"pixel_area": 480.0, "actual_count": 40.0},
    ]

    print("\n--- 🖥️ CAMERA VISION CALIBRATION LOGS ---")
    for idx, sample in enumerate(calibration_data):
        print(f"Run {idx+1}: Visual Area = {sample['pixel_area']} units -> Count = {sample['actual_count']} pills")

    # 3. The Math: Calculating Means for X and Y
    n = len(calibration_data)
    mean_x = sum(s["pixel_area"] for s in calibration_data) / n
    mean_y = sum(s["actual_count"] for s in calibration_data) / n

    # 4. Ordinary Least Squares (OLS) to find Slope (m) and Intercept (b)
    # Slope Formula: m = sum((x - mean_x) * (y - mean_y)) / sum((x - mean_x)^2)
    # Intercept Formula: b = mean_y - (m * mean_x)
    numerator = 0.0
    denominator = 0.0
    
    for s in calibration_data:
        x_dev = s["pixel_area"] - mean_x
        y_dev = s["actual_count"] - mean_y
        numerator += x_dev * y_dev
        denominator += x_dev ** 2

    m = numerator / denominator
    b = mean_y - (m * mean_x)

    print("\n--- 📊 STEP 1: CALCULATING ORDINARY LEAST SQUARES ---")
    print(f"Calculated Mean Sensed Area (Mean X): {mean_x:.2f}")
    print(f"Calculated Mean Pill Count (Mean Y): {mean_y:.2f}")
    print(f"Resulting Model Equation: Predicted Count = ({m:.4f} * Area) + ({b:.4f})")

    # 5. Incoming Real-Time Scan Query
    # A robot drops a batch on the tray, and the overhead camera captures 415.0 units of area.
    test_area_x = 415.0
    print(f"\n--- 🚨 AUTOMATION INFRASTRUCTURE INTAKE ---")
    print(f"Incoming Scanner Feed -> Detected Pixel Blob Area (X): {test_area_x}")

    # 6. Model Prediction (Inference)
    # Linear Equation: y = mx + b
    predicted_count = (m * test_area_x) + b
    
    # Round to nearest whole pill since fractions of pills shouldn't exist in standard dispensing
    final_count_decision = round(predicted_count)

    print(f"\n--- 🔄 MODEL INFERENCE PIPELINE ---")
    print(f"Raw Predicted Target Continuous Output (y): {predicted_count:.2f}")
    print(f"Post-Processed Output (Nearest Integer): {final_count_decision} Tablets")

    # 7. Quality Control Verification Boundary
    # Ground Truth target for this scan test run is 35 tablets
    actual_truth = 35
    print(f"Downstream Scale Verification Weight Check: {actual_truth} Tablets")

    if final_count_decision == actual_truth:
        print("\n🏆 SUCCESS: Your regression loop hit the exact pill count validation target!")
        print("The bottle is automatically capped and forwarded to shipping packaging.")
    else:
        print("\n💥 CALIBRATION FAULT: Verification mismatch! Bottle routed to the rejection bin.")

if __name__ == "__main__":
    linear_regression_counter_game()
