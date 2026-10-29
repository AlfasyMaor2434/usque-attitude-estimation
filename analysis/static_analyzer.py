import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # the repository root
from data_path import DATA_DIR  # noqa: E402
import os
import numpy as np

# שנה את הנתיב הזה לנתיב האמיתי של התיקייה אצלך במחשב
STATIC_DIR = os.path.join(DATA_DIR, "insane_static_sensor_data")


def analyze_sensor(filename, sensor_name):
    filepath = os.path.join(STATIC_DIR, filename)
    if not os.path.exists(filepath):
        print(f"[!] File {filename} not found in {STATIC_DIR}")
        return

    print(f"\nAnalyzing {sensor_name} ({filename})... This might take a few seconds.")

    # טעינת הנתונים (מדלגים על שורת הכותרת)
    try:
        data = np.loadtxt(filepath, delimiter=',', comments='#', skiprows=1)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return

    # עמודות (מבוסס על INSANE פורמט: t, x, y, z)
    if 'imu' in filename:
        accel = data[:, 1:4]
        gyro = data[:, 4:7]

        # חישוב ה-Bias (ממוצע) - סחיפה קבועה
        gyro_bias = np.mean(gyro, axis=0)

        # חישוב הרעש (שונות / Variance) - כמה החיישן "רועד" כשהוא סטטי
        accel_var = np.var(accel, axis=0)
        gyro_var = np.var(gyro, axis=0)

        print(f"--- {sensor_name} Results ---")
        print(f"Gyroscope Bias (Mean) [rad/s]: X={gyro_bias[0]:.6e}, Y={gyro_bias[1]:.6e}, Z={gyro_bias[2]:.6e}")
        print(f"Gyroscope Noise (Variance):    X={gyro_var[0]:.6e}, Y={gyro_var[1]:.6e}, Z={gyro_var[2]:.6e}")
        print(f"Mean Gyro Noise (For Filter):  {np.mean(gyro_var):.6e}")

        print(f"Accel Noise (Variance):        X={accel_var[0]:.6e}, Y={accel_var[1]:.6e}, Z={accel_var[2]:.6e}")
        print(f"Mean Accel Noise (For Filter): {np.mean(accel_var):.6e}")

    elif 'mag' in filename:
        mag = data[:, 1:4]
        mag_var = np.var(mag, axis=0)
        print(f"--- {sensor_name} Results ---")
        print(f"Mag Noise (Variance):          X={mag_var[0]:.6e}, Y={mag_var[1]:.6e}, Z={mag_var[2]:.6e}")
        print(f"Mean Mag Noise (For Filter):   {np.mean(mag_var):.6e}")


if __name__ == "__main__":
    print("Starting Static Sensor Analysis...")
    analyze_sensor("px4_imu.csv", "PX4 IMU")
    analyze_sensor("lsm_imu.csv", "LSM9DS1 IMU")
    analyze_sensor("px4_mag.csv", "PX4 Magnetometer")
    print("\nDone!")