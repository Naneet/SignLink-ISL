# SignLink-ISL

This is a work-in-progress repo 🙂  
New features will be added in the future, and things might be subject to change as we make progress on this project!

## **SignLink-ISL: Video Dataset Pipeline**

### **Dataset**

Download the dataset: [Include Dataset Subset](https://www.kaggle.com/datasets/naneet1/include-dataset-subset)

### **Dataset Structure**

The dataset should be organized in the following folder structure for the pipeline to work:

```plaintext
root/
├── dog/
│   ├── xxx.mp4
│   ├── xxy.mp4
│   ...
├── cat/
│   ├── 123.mp4
│   ├── nsdf3.mp4
│   ...

```

### **Features**

- **Efficient Preprocessing:** Automatically extracts and processes video frames into PyTorch tensors.
- **User-Friendly:** No extra steps required. Simply organize your dataset as shown above, and you're good to go!
- **Customizable:** Supports parameters like number of frames (`NUM_FRAMES`), frame rate (`fps`), resolution, and more.

### **Requirements**

To run the notebook, ensure you have the following installed:

- **Python** (3.7+)
- **PyTorch**
- **TorchVision**
- **FFmpeg**
- **Pillow (PIL)**
- **OS utilities:** `os`, `shutil`, `subprocess`
- **Visualization:** `matplotlib`


