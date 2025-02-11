import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Main HR database
HR_DB_PATH = os.path.join(BASE_DIR, "database", "hr_tool.db")

# Additional Predictions database (for burnout risk/future performance)
ADDITIONAL_DB_PATH = os.path.join(BASE_DIR, "database", "hr_additional.db")
# Additional Predictions database (for burnout risk/future performance)
ADDITIONAL_HR_DB_PATH = os.path.join(BASE_DIR, "database", "additional_hr_data.db")

# HRMS database for employee monitoring and attendance
HRMS_DB_PATH = os.path.join(BASE_DIR, "database", "hrms_data.db")
