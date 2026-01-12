# Real Estate Price Forecasting System

An end-to-end full-stack application designed to analyze historical real estate data and forecast future market trends using Deep Learning. This project ingests Zillow research data, trains region-specific LSTM (Long Short-Term Memory) models, and serves these insights through a high-performance REST API and an interactive modern web dashboard.

## 🚀 Key Features

*   **Deep Learning Forecasting**: Utilizes TensorFlow/Keras LSTM models to predict future real estate prices based on historical trends.
*   **Interactive Dashboard**: A responsive Vue.js frontend featuring dynamic charts (Chart.js) to visualize historical data and future forecasts.
*   **Granular Analysis**: Drill down data by **State**, **Region** (Metro area), and specific **Market Indicators** (e.g., Median Sale Price, Market Heat Index).
*   **Robust Backend**: Powered by FastAPI for high-performance, asynchronous API execution.
*   **Containerized Architecture**: Fully Dockerized application for easy deployment and orchestration.

---

## 🛠️ Tech Stack

### **Frontend**
*   **Framework**: Vue.js 3 (Composition API)
*   **Visualization**: Chart.js 
*   **Networking**: Axios
*   **Styling**: CSS3

### **Backend**
*   **API Framework**: FastAPI
*   **Server**: Uvicorn
*   **Validation**: Pydantic
*   **Database**: SQLAlchemy / SQLite

### **Machine Learning & Data**
*   **Core Logic**: TensorFlow (Keras)
*   **Data Processing**: Pandas, NumPy, Scikit-Learn
*   **Architecture**: LSTM (Long Short-Term Memory) Networks for Time-Series Forecasting
*   **Data Source**: Zillow Research Data

---

## 📂 Project Structure

```text
Real-Estate-Pricing/
├── data/                       # Raw and processed Zillow data storage
├── ml/                         # Machine Learning Core
│   ├── training.py             # LSTM model training pipeline
│   ├── api_helper.py           # Bridge between API and ML models
│   └── model_store/            # Serialized trained models and scalers
├── real_estate_api/            # Backend (FastAPI)
│   ├── app/main.py             # API Entry point and Routes
│   └── Dockerfile              # Backend container config
├── real_estate_frontend/       # Frontend (Vue.js)
│   ├── src/components/         # Vue components (e.g., Forecast.vue)
│   └── Dockerfile              # Frontend container config
└── docker-compose.yml          # Orchestration for full-stack deployment
```

---

## ⚡ Getting Started

### Prerequisites
*   **Docker & Docker Compose** (Recommended)
*   *Or for local dev:* Python 3.10+, Node.js 16+

### 1. Clone the Repository
```bash
git clone https://github.com/jaideepmurkute/Real-Estate-Pricing.git
cd Real-Estate-Pricing
```

### 2. Data Setup
Ensure you have the Zillow dataset placed in the `data/` directory. The system expects the Zillow CSV files (e.g., `Metro_mean_sale_price...`) to be located at:
`data/zillow/original_data/`

---

## 🖥️ Running the Application

### Option A: Using Docker (Recommended)
The easiest way to run the entire stack is via Docker Compose.

```bash
# Build and Start both Frontend and Backend services
docker-compose up --build
```
Once running, access the application at:
*   **Frontend (Dashboard):** [http://localhost:8080](http://localhost:8080)
*   **Backend (API Docs):** [http://localhost:8000/docs](http://localhost:8000/docs)

To stop the services:
```bash
docker-compose down
```

### Option B: Manual Local Development

#### 1. Backend Setup
```bash
cd real_estate_api

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

#### 2. Frontend Setup
```bash
cd real_estate_frontend

# Install dependencies
npm install

# Run the development server
npm run serve
```
Access the frontend at `http://localhost:8080`.

---

## 🧠 Machine Learning Pipeline

The project uses a specialized pipeline for time-series forecasting:

1.  **Data Ingestion**: Reads CSV metrics from Zillow.
2.  **Preprocessing**: Normalizes data using MinMax scaling to aid LSTM convergence.
3.  **Windowing**: Converts time-series data into supervised learning sequences (Default `look_back = 6 months`).
4.  **Training**: Trains an LSTM network for each region. Models are saved in the `ml/model_store` directory.
5.  **Inference**: The `api_helper.py` loads the region-specific model on demand to generate real-time forecasts.

---

## 📄 License

[MIT License](LICENSE)
