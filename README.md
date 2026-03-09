# ITS TV Storyboard AI
A full-stack, AI-powered application designed to automate and augment the storyboard creation process for ITS TV. 

This project allows users to input text prompts and asynchronously generate visual storyboard frames using local AI models (like Ollama and Stable Diffusion), making it highly optimized for hardware environments with limited VRAM (specifically targeting the RTX 3050 6GB VRAM profile).

## 🚀 Features
* **Asynchronous Generation Engine**: Backend generation processes run asynchronously via FastAPI background tasks, ensuring the UI remains highly responsive during heavy GPU workloads.
* **Agentic UI Design**: State-of-the-art React frontend with Tailwind CSS, featuring ambient glows, glassmorphism, dynamic animations, and a real-time hardware status monitor.
* **Local AI Integration Hooks**: Pre-configured architecture for connecting Diffusers (Stable Diffusion v1.5), Ollama, and ChromaDB locally without third-party API costs.

---

## 💻 Tech Stack
* **Frontend**: React.js (Vite), Tailwind CSS, Lucide React (Icons), Axios
* **Backend**: Python 3.10+, FastAPI, Uvicorn, Pydantic v2
* **Package Management**: Conda (Backend), npm (Frontend)

---

## 🛠️ Prerequisites & Setup

Ensure you have the following installed on your system:
* [Node.js](https://nodejs.org/) (v16 or higher)
* [Miniconda](https://docs.anaconda.com/free/miniconda/index.html) or Anaconda

### 1. Backend Setup
The backend runs on FastAPI and uses a dedicated Conda environment named `agas_skripsi`.

1. Open a terminal and navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Activate your conda environment (create it first if you haven't already):
   ```bash
   conda activate agas_skripsi
   ```
3. Install the required dependencies (if you haven't yet). It is recommended to install FastAPI and Uvicorn:
   ```bash
   pip install fastapi uvicorn pydantic
   ```
4. **Run the backend server**:
   ```bash
   python main.py
   ```
   *The server will start at `http://localhost:8000` or `http://0.0.0.0:8000`.*

### 2. Frontend Setup
The frontend is a React application styled with Tailwind CSS.

1. Open a **new** terminal window and navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install the Node.js dependencies:
   ```bash
   npm install
   ```
3. **Run the frontend development server**:
   ```bash
   npm run dev
   ```
   *Vite will provide a local URL (e.g., `http://localhost:5173`) where you can view the application in your browser.*

---

## 🎮 How to Use the Application
1. **Ensure both servers are running**: Your terminal running the backend should say `Application startup complete`, and your frontend terminal should show the Vite local URL.
2. **Access the application**: Open the frontend URL in your web browser.
3. **Check connection status**: Look at the top right of the application header. It should display a pulsing green indicator that says `"Backend Active"`. 
4. **Generate a Frame**: Type a scene description into the prompt box at the bottom and hit `Enter` (or click the send icon).
5. **Wait for synthesis**: The UI will display a loading scanner animation while the backend processes the request asynchronously. Once finished, the image will appear in the Output Viewer canvas.

## ⚙️ Hardware Limitations Notes
This application implements simulated delays (`asyncio.sleep`) in the current backend logic to mimic the exact time delays you would experience when rendering images on an RTX 3050 6GB. The true integration hooks for `diffusers` and `ollama` are commented out with `TODO`s in `backend/main.py` and are ready for implementation in the next phase!
