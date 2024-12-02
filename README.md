# Chat with PDF locally(Or using streamlit) using Ollama + LangChain

A powerful local RAG (Retrieval Augmented Generation) application that lets you chat with your PDF documents using Ollama and LangChain. This project includes both a Jupyter notebook for experimentation and a Streamlit web interface for easy interaction.

## 🚀 Getting Started

### Prerequisites

1. **Install Ollama**
   - Visit [Ollama's website](https://ollama.ai) to download and install
   - Pull required models:
     ```bash
     ollama pull llama2  # or your preferred model
     ollama pull nomic-embed-text
     ```

2. **Clone Repository**
   ```bash
   git clone https://github.com/TAHAKH7/TAHA_RAG_PDF_OLLAMA.git
   cd TAHA_RAG_PDF_OLLAMA
   ```

3. **Set Up Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

### 🎮 Running the Application

#### Option 1: Streamlit Interface
```bash
streamlit run streamlit_app.py
```
Then open your browser to `http://localhost:8501`

![Streamlit UI](st_app_ui.png)
*Streamlit interface showing PDF viewer and chat functionality*

#### Option 2: Jupyter Notebook
```bash
jupyter notebook
```
Open `updated_rag_notebook.ipynb` to experiment with the code

### Common Errors

#### ONNX DLL Error
If you encounter this error:
```
DLL load failed while importing onnx_copy2py_export: a dynamic link Library (DLL) initialization routine failed.
```

Try these solutions:
1. Install Microsoft Visual C++ Redistributable:
   - Download and install both x64 and x86 versions from [Microsoft's official website](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)
   - Restart your computer after installation

2. If the error persists, try installing ONNX Runtime manually:
   ```bash
   pip uninstall onnxruntime onnxruntime-gpu
   pip install onnxruntime
   ```

#### CPU-Only Systems
If you're running on a CPU-only system:

1. Ensure you have the CPU version of ONNX Runtime:
   ```bash
   pip uninstall onnxruntime-gpu  # Remove GPU version if installed
   pip install onnxruntime  # Install CPU-only version
   ```

2. You may need to modify the chunk size in the code to prevent memory issues:
   - Reduce `chunk_size` to 500-1000 if you experience memory problems
   - Increase `chunk_overlap` for better context preservation

Note: The application will run slower on CPU-only systems, but it will still work effectively.
