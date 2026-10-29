import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # the repository root
from data_path import DATA_DIR  # noqa: E402
import numpy as np
import os
import glob

# הנתיב לתיקיית הרעידות
VIB_DIR = os.path.join(DATA_DIR, "insane_vibration_analysis_sensor_data")
DATASETS = ['v10', 'v20', 'v40', 'v60', 'v80', 'v100']


def main():
    print("--- Starting Vibration Noise Profiling ---")
    throttles = []
    accel_vars = []
    mag_vars = []

    for d in DATASETS:
        d_path = os.path.join(VIB_DIR, d)
        if not os.path.exists(d_path):
            print(f"Directory {d_path} not found. Skipping...")
            continue

        try:
            # חיפוש קבצי ה-CSV
            imu_file = glob.glob(os.path.join(d_path, "*imu*.csv"))[0]
            mag_file = glob.glob(os.path.join(d_path, "*mag*.csv"))[0]

            # הוספנו skiprows=1 כדי לדלג על שורת הכותרת (t, a_x, a_y...)
            imu_data = np.loadtxt(imu_file, delimiter=',', comments='#', skiprows=1)
            mag_data = np.loadtxt(mag_file, delimiter=',', comments='#', skiprows=1)

            # הוצאת התאוצה (עמודות 1,2,3) והמגנטומטר (1,2,3)
            accel = imu_data[:, 1:4]
            mag = mag_data[:, 1:4]

            # אחוז המצערת חלקי 100 (למשל v40 = 0.4)
            throttle = float(d.replace('v', '')) / 100.0

            # חישוב השונות (Variance) הכוללת של החיישן
            accel_var = np.mean(np.var(accel, axis=0))
            mag_var = np.mean(np.var(mag, axis=0))

            throttles.append(throttle)
            accel_vars.append(accel_var)
            mag_vars.append(mag_var)

            print(f"[{d}] Throttle: {throttle:.1f} | Accel Var: {accel_var:.5f} | Mag Var: {mag_var:.5f}")
        except Exception as e:
            print(f"Error processing dataset {d}: {e}")

    # אם הצלחנו לקרוא לפחות שני מאגרים, נחשב את הפיט (Fit)
    if len(throttles) > 1:
        # --- חישוב פקטור הניפוח הריבועי (Least Squares Fit: Noise = Base + K * Throttle^2) ---
        throttles_sq = np.array(throttles) ** 2

        # חישוב הפקטורים (השיפוע של הגרף ביחס לבסיס)
        K_accel = np.polyfit(throttles_sq, accel_vars, 1)[0] / accel_vars[0]
        K_mag = np.polyfit(throttles_sq, mag_vars, 1)[0] / mag_vars[0]

        print("\n>>> PROFILING COMPLETE! COPY THESE VALUES TO usque.py <<<")
        print(f"ACCEL_K = {K_accel:.2f}")
        print(f"MAG_K   = {K_mag:.2f}")
    else:
        print("\n[!] Not enough data to perform profiling. Check your directory path.")

if __name__ == "__main__":
    main()