# Smruti Setup & Installation Guide

This guide provides instructions for setting up **Smruti** across different operating systems (macOS, Linux, Windows), configuring virtual environments, and resolving system dependencies.

---

## 1. Prerequisites

- **Python**: Version 3.10, 3.11, or 3.12 (Check via `python --version` or `python3 --version`).
- **RAM**: Minimum 500 MB free memory (Smruti runs Quantized `bge-small-en-v1.5` embeddings on CPU via ONNX Runtime).
- **Disk Space**: ~150 MB for the local embedding model cache.
- **Internet Access**: Required once on first run to download the local model weights (~130 MB); completely offline afterward.

---

## 2. OS-Specific Installation

### macOS (Apple Silicon M1/M2/M3 & Intel)

1. **Verify Python & Homebrew**:
   ```bash
   brew install python@3.11
   ```

2. **Create a Dedicated Virtual Environment**:
   ```bash
   python3 -m venv ~/.smruti-env
   source ~/.smruti-env/bin/activate
   ```

3. **Install Smruti**:
   ```bash
   # From source repository:
   git clone https://github.com/wizardwithcodehazard/smriti.git
   cd smriti
   pip install -e .
   ```

4. **Verify CLI**:
   ```bash
   smruti --help
   ```

---

### Linux (Ubuntu, Debian, Fedora, Arch)

1. **Install Python Build Tools**:
   ```bash
   # Ubuntu / Debian
   sudo apt update && sudo apt install -y python3 python3-pip python3-venv git
   
   # Fedora
   sudo dnf install -y python3 python3-pip git
   ```

2. **Setup Virtual Environment**:
   ```bash
   python3 -m venv ~/.smruti-env
   source ~/.smruti-env/bin/activate
   ```

3. **Install Smruti**:
   ```bash
   git clone https://github.com/wizardwithcodehazard/smriti.git
   cd smriti
   pip install -e .
   ```

4. **Verify CLI**:
   ```bash
   smruti status
   ```

---

### Windows (PowerShell / Command Prompt)

1. **Ensure Python is on PATH**:
   Download Python 3.11 or 3.12 from [python.org](https://www.python.org/) and check **"Add Python to PATH"** during installation.

2. **Open PowerShell as User**:
   ```powershell
   # Create a virtual environment
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
   *(If execution policy errors occur, run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`)*

3. **Install Smruti**:
   ```powershell
   git clone https://github.com/wizardwithcodehazard/smriti.git
   cd smriti
   pip install -e .
   ```

4. **Verify Installation**:
   ```powershell
   smruti status
   ```

---

## 3. Storage Backends

By default, Smruti creates a local SQLite database in `.smruti/smruti.db` inside your active project root.

### Pluggable PostgreSQL Backend (Optional)
For team environments or centralized services, configure PostgreSQL by setting the environment variable:

```bash
export SMRUTI_DATABASE_URL="postgresql://user:password@localhost:5432/smruti_db"
```
Or in Windows PowerShell:
```powershell
$env:SMRUTI_DATABASE_URL="postgresql://user:password@localhost:5432/smruti_db"
```

Smruti automatically applies migrations and provisions tables on startup.

---

## 4. Verifying Model Embeddings

On first run, `fastembed` downloads the quantized model `bge-small-en-v1.5`. You can verify this works offline by running:

```bash
smruti remember "Strict test verification fact" --category "test"
smruti recall "test verification"
```

If the heuristic is returned with a strength score > 1.0, your local embeddings and storage pipeline are fully operational.
