# 吞嚥障礙檢測專案 (Dysphagia Detection Project)

本專案旨在利用先進的 RF (射頻) 訊號處理與機器學習技術，分析與吞嚥相關的生理訊號，以實現對吞嚥障礙 (Dysphagia) 的早期檢測與評估。專案核心是 `nearpy` 函式庫，它提供了一套完整的工具鏈，從原始數據讀取、訊號預處理、特徵提取到模型訓練。

## ✨ 專案特色

- **端到端的分析流程**: 提供從原始 `.h5` 數據到最終分類結果的完整 Jupyter Notebook 範例。
- **模組化訊號處理**: `nearpy` 函式庫將濾波、分段、特徵提取等功能模組化，易於擴充與維護。
- **先進的機器學習整合**: 整合了 `scikit-learn` 與 `PyTorch Lightning`，方便進行傳統機器學習與深度學習實驗。
- **可視化工具**: 內建多種繪圖函式，方便使用者直觀地理解訊號與分析結果。
- **跨平台支援**: 提供了在 Windows 與 macOS 上的詳細安裝指引。

---

## 🚀 快速開始

### 必要條件

在開始之前，請確保您的系統已安裝以下軟體：

- **Git**: 用於版本控制與複製專案。
- **Python**: 建議版本 3.11+。
- **Poetry**: 用於管理 Python 依賴套件。

### 💻 安裝指南

本專案使用 Poetry 來管理虛擬環境與依賴套件，確保在不同平台上有一致的開發體驗。

#### macOS

1.  **安裝 Homebrew** (若尚未安裝):
    ```bash
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    ```

2.  **安裝 Python**:
    ```bash
    brew install python
    ```

3.  **安裝 `pipx`** (用於安裝 Python CLI 工具):
    ```bash
    pip install pipx
    ensurepath
    ```

4.  **安裝 Poetry**:
    ```bash
    pipx install poetry
    ```

5.  **複製專案並安裝依賴**:
    ```bash
    # 複製專案
    git clone <YOUR_REPOSITORY_URL>
    cd dysphagia_project

    # Poetry 會自動偵測 pyproject.toml 並安裝所有依賴
    poetry install
    ```

#### Windows

1.  **安裝 Python**:
    - 前往 [Python 官網](https://www.python.org/downloads/) 下載並安裝 Python 3.11+。
    - **重要**: 在安裝過程中，請務必勾選 `Add Python to PATH` 選項。

2.  **安裝 `pipx`** (開啟命令提示字元 `cmd` 或 `PowerShell`):
    ```powershell
    pip install pipx
    pipx ensurepath
    ```
    *您可能需要重啟命令提示字元來讓 `pipx` 生效。*

3.  **安裝 Poetry**:
    ```powershell
    pipx install poetry
    ```

4.  **複製專案並安裝依賴**:
    ```powershell
    # 複製專案
    git clone <YOUR_REPOSITORY_URL>
    cd dysphagia_project

    # Poetry 會自動偵測 pyproject.toml 並安裝所有依賴
    poetry install
    ```

---

## 🛠️ 如何使用

安裝完成後，您可以開始進行數據分析。

1.  **啟動虛擬環境**:
    在專案根目錄下，執行以下命令來啟動由 Poetry 管理的虛擬環境 shell：
    ```bash
    poetry shell
    ```

2.  **啟動 Jupyter Notebook**:
    本專案的主要分析流程位於 `notebooks/` 資料夾中。
    ```bash
    jupyter notebook notebooks/01_Full_Workflow_Analysis.ipynb
    ```

3.  **遵循 Notebook 流程**:
    - **數據讀取**: Notebook 會從 `data/raw/` 目錄讀取 `.h5` 格式的原始數據。
    - **訊號處理**: 應用 `nearpy` 函式庫進行濾波、去趨勢等操作。
    - **特徵提取**: 計算時域與頻域特徵。
    - **模型訓練**: 使用提取的特徵訓練一個簡單的分類器來區分不同的動作 (如：休息、吞嚥、踮腳)。

---

## 📂 專案結構

```
dysphagia_project/
├── data/
│   ├── raw/          # 存放原始 .h5 數據
│   ├── processed/    # 存放處理後的數據 (可選)
│   └── results/      # 存放分析結果 (可選)
│
├── nearpy/           # 核心訊號處理與機器學習函式庫
│   ├── nearpy/
│   │   ├── ai/       # AI/ML 相關模組
│   │   ├── features/ # 特徵提取模組
│   │   ├── io/       # 數據讀寫模組
│   │   ├── plots/    # 繪圖模組
│   │   └── preprocess/ # 訊號預處理模組
│   └── pyproject.toml  # nearpy 函式庫的依賴設定
│
├── notebooks/        # Jupyter Notebooks 分析腳本
│   └── 01_Full_Workflow_Analysis.ipynb
│
├── venv/             # Python 虛擬環境 (由 Poetry 自動管理)
│
└── README.md         # 本說明檔案
```

---

## 📦 主要依賴

本專案的主要依賴由 `nearpy/pyproject.toml` 管理，摘錄如下：

- `torch`, `lightning`: 用於深度學習模型。
- `scikit-learn`: 用於傳統機器學習模型與評估。
- `pandas`, `numpy`: 用於數據處理。
- `matplotlib`, `seaborn`: 用於數據可視化。
- `h5py`: 用於讀取 HDF5 檔案。

所有依賴都會透過 `poetry install` 命令自動安裝。
