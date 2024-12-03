### **README: Trash Classification Using CNN (MobileNetV2)**  

---

## **Project Overview**  
This project aims to classify images of trash into predefined categories using a Convolutional Neural Network (CNN) based on the MobileNetV2 architecture. The classification model is trained to assist in waste segregation and recycling processes by accurately identifying the type of trash in an image.

---

## **Features**  
- Uses the MobileNetV2 architecture for efficient and accurate image classification.  
- Classifies trash into six categories: **trash**, **plastic**, **cardboard**, **metal**, **paper**, and **glass**.  
- Includes an automated pipeline for data preparation, model training, and evaluation.  
- Employs techniques like **data augmentation** and **oversampling** to improve model performance.  

---

## **Dataset**  
The dataset is sourced from [Hugging Face - TrashNet Dataset](https://huggingface.co/datasets/garythung/trashnet), which contains labeled images of trash items categorized into six classes.  

---

## **Steps Involved**  

### **1. Data Loading**  
- The dataset is uploaded to Google Drive and accessed using Python's `os` library.  

### **2. Data Exploration**  
- Analyze the dataset to understand class distributions and image counts for each category.  
- Visualize sample images to gain insights into their patterns and variations.  

### **3. Data Splitting**  
- Split the dataset into **training (70%)**, **validation (15%)**, and **testing (15%)** subsets to ensure robust training and evaluation.

### **4. Data Preprocessing**  
- Perform resizing, normalization, and balancing of classes using oversampling to handle class imbalances.  
- Apply data augmentation techniques, including rotation, zooming, and rescaling, to enhance dataset diversity.  

### **5. Model Building**  
- Utilize MobileNetV2, a lightweight and efficient CNN architecture pre-trained on ImageNet, for transfer learning.  
- Fine-tune the model using the trash dataset to adapt it to the classification task.  

### **6. Model Training**  
- Train the model using the prepared dataset, optimizing the loss function with an appropriate optimizer (e.g., Adam).  
- Use validation data to monitor performance and avoid overfitting.  

### **7. Model Evaluation**  
- Test the model on unseen data and evaluate its performance using metrics such as accuracy, precision, recall, and F1-score.  

---

## **How to Run the Project**  

### **Requirements**  
- Python 3.8 or later  
- TensorFlow/Keras  
- NumPy, Pandas, Matplotlib  
- Google Colab or any Python-compatible IDE  

## **Results**  
- The final model achieved an accuracy of **90%** on the test set and **95%** on the train set. 
 
---

## **Future Work**  
- Improve model accuracy with additional data and advanced augmentation techniques.  
- Deploy the model as a web application for real-time trash classification.  
- Incorporate additional classes for more granular waste categorization.  

---
