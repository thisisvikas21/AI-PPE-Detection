
# 🛡️📹 AI-Powered Industrial Safety Surveillance System
---

**Category:** Industry Defined Problem   |  **Organization:** Jyoti CNC Automation, Rajkot 

> 🍴 **Fork notice:** This is a fork of [darshan1654/AI-PPE-Detection](https://github.com/darshan1654/AI-PPE-Detection) with a refreshed dashboard UI. See [What's New in This Fork](#-whats-new-in-this-fork). The screenshots below show the original UI.

![AI CCTV Surveillance · Streamlit_page-0001](https://github.com/user-attachments/assets/58a5799f-b7a4-47f7-b165-b4078429a1ea) ![AI CCTV Surveillance · Streamlit2_page-0001](https://github.com/user-attachments/assets/452a0632-cbe2-4a3b-b6ee-3c876327804f)

---

## 📄 Project Overview

This project integrates **AI-powered Personal Protective Equipment (PPE) detection** and **CCTV-based Anomaly Surveillance** into a single real-time system. It is designed to improve **Workplace Safety**, **Regulatory Compliance**, and **Operational Efficiency** in industrial environments.

Using advanced **Deep Learning models** and **Computer Vision**, the system ensures that workers adhere to safety protocols and that unusual or unsafe activities are promptly flagged.

---

## ✅ Key Features

### 👷 PPE Detection Module
- Real-time detection of:
  - 🪖 Helmets  
  - 😷 Face Masks  
  - 👷 Safety Vests  
  - 🧤 Gloves  
- Supports live feed from **Webcam/IP Camera**
- Upload and analyze **Video/Image files**
- Detection logs with timestamps and confidence scores

### 📹 CCTV Anomaly Detection Module
- Real-time detection of:
  - 🚫 Entry into restricted zones  
  - ⚠️ Safety violations (e.g., no helmet, improper behavior)  
  - 🚷 Suspicious or unsafe movements  
- Continuous monitoring via CCTV/IP camera
- Alert generation on detection

### 📊 Dashboard (Built with Streamlit)
- Live status feed with detection results
- Real-time preview of camera feed
- Violation and compliance logs
- Summary statistics and compliance reports
- KPI summary cards for total violations, today's count, most common violation and average confidence

---

## ✨ What's New in This Fork

### 🌙 Dark "Control Room" Theme
- A dark navy palette with an amber accent, designed for monitoring screens
- A hero banner with a pulsing **SYSTEM ONLINE** status indicator
- Restyled sidebar, tabs, buttons, metric cards, violation alerts and footer
- Theme defaults are set in `.streamlit/config.toml`, so native Streamlit widgets match the custom styling

### 📈 Dashboard KPI Row
Four summary cards above the tabs, calculated live from `violation_logs.csv`:

| Card | Shows |
|------|-------|
| 🔴 Total Violations | Every violation ever logged |
| 🟠 Today | Violations logged today |
| 🟣 Most Common | The most frequent violation type and how often it occurred |
| 🟢 Avg. Confidence | Average detection confidence across all logs |

---

## 📦 Tech Stack

| Component       | Technology         |
|----------------|--------------------|
| 💡 AI Model     | YOLOv8 (Ultralytics) |
| 🧠 Backend       | Python, OpenCV     |
| 🌐 Frontend/UI  | Streamlit          |
| 🎥 Video Input  | Webcam/IP Camera   |
| 📊 Data Logging | Pandas, CSV Logs   |

---

## 🏭 Industrial Benefits

- ✅ Automated compliance with PPE policies  
- 🔍 Real-time safety monitoring  
- 📉 Reduced accident risk and manual supervision  
- 📊 Actionable insights from safety data  

---

## 📸 Sample Outputs

- 📷 Detected image with PPE boxes  
- 🧾 Logs with timestamp and violation type  
- 📈 Streamlit dashboard with real-time updates  

---

## 🔧 Installation & Setup

```bash
# 1. Clone the repository
git clone https://github.com/thisisvikas21/AI-PPE-Detection.git
cd AI-PPE-Detection

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 3. Install required dependencies
pip install -r requirements.txt

# 4. Run the application
streamlit run app.py
```

---

## 📁 Project Structure

```
├── .streamlit/
│   └── config.toml      # Dark theme settings
├── app.py               # Streamlit app
├── best.pt              # YOLOv8 model files
├── yolov8n.pt           # YOLOv8 model files
├── violation_logs.csv   # Detection logs CSV file
├── requirements.txt     # Dependencies
├── packages.txt         # packages
└── runtime.txt          # For deployment (e.g., Heroku)
```

---

## 👨‍💻 Team Details

**Group ID:** G00171

| Name             | Email                                                                       |
| ---------------- | --------------------------------------------------------------------------- |
| Kushal A. Parekh | [22ce113@svitvasad.ac.in](mailto:22ce113@svitvasad.ac.in)                   |
| Darshan Pardeshi | [darshanpardeshi1654@gmail.com](mailto:darshanpardeshi1654@gmail.com)       |
| Param V. Jani    | [janiparam61@gmail.com](mailto:janiparam61@gmail.com)                       |
| Darshan Panchal  | [mpdarshanpanchal001031@gmail.com](mailto:mpdarshanpanchal001031@gmail.com) |
| Jaymin Raval     | [ravaljaymin2908@gmail.com](mailto:ravaljaymin2908@gmail.com)               |

---

## 🔮 Future Scope

* 🔔 Voice/Email/SMS alert system
* 🔗 Integration with ERP systems
* 📊 Admin dashboard with analytics
* 📤 Auto-upload violation clips
* 📡 Multi-location camera support

---

## 🏆 Achievements

* ✅ Used in **Real Industrial setup** at Jyoti CNC
* 🎓 Presented at **College-level expo**
* 📡 Successfully tested with **Live Camera feeds**

---

## 📜 License

For **Academic and Research use** only.

---

## 🙏 Acknowledgements

* [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
* [OpenCV](https://opencv.org)
* [Streamlit](https://streamlit.io)
* Jyoti CNC Automation Pvt. Ltd.
* SVIT Vasad

---

## 📬 Contact

📧 [janiparam61@gmail.com](mailto:janiparam61@gmail.com)
📧 [darshanpardeshi1654@gmail.com](mailto:darshanpardeshi1654@gmail.com)
📧 [22ce113@svitvasad.ac.in](mailto:22ce113@svitvasad.ac.in)
📧 [mpdarshanpanchal001031@gmail.com](mailto:mpdarshanpanchal001031@gmail.com)
📧 [ravaljaymin2908@gmail.com](mailto:ravaljaymin2908@gmail.com)

<img width="709" height="186" alt="image" src="https://github.com/user-attachments/assets/9afae834-e0bd-4a0f-9b6a-919b491df72d" />

---
