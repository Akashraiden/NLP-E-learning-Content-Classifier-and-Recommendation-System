## "NLP-E-learning-Content-Classifier-and-Recommendation-System"

### Project Overview:
#### This project aims to classify and recommend e-learning content based on user preferences, engagement, and learning patterns. Using NLP and ML techniques, it automates content categorization and suggests relevant courses to enhance personalized learning. The system analyzes course descriptions, user behavior, and feedback to provide accurate recommendations. It improves user experience by ensuring access to the most relevant educational resources.


### The system has two major components:
1. **Content Classification**: Classifies courses into sub-categories based on the course title, skills, course type, and duration.
2. **Course Recommendation**: Recommends similar courses based on a given course using cosine similarity on feature vectors.

## Key Features

### Content Classification:
- **Text Preprocessing**: Uses NLP techniques such as **lemmatization** and **stopword removal** to clean and preprocess the course titles and skills.
- **Feature Extraction**: Uses **TF-IDF vectorization** to convert the course titles and skills into numerical vectors.
- **Machine Learning Model**: A **Random Forest Classifier** is trained to predict the **sub-category** of the course.
- **Model Saving**: Saves the trained model and vectorizers as **pickle files** to allow future use.
- **Prediction**: Uses the trained model to predict the **sub-category** of unseen courses.

### Course Recommendation:
- **Cosine Similarity**: Recommends similar courses based on cosine similarity of feature vectors derived from course titles, skills, course type, and duration.
- **Top-N Recommendations**: Returns the top N most similar courses to a given course based on cosine similarity.

### Data Preprocessing Enhancements:
- **KMeans Clustering for Null Value Imputation**: KMeans clustering is applied to fill **null values** in the dataset based on similarities derived from the **Title** column. The clustering algorithm helps to identify similar course titles and predict the missing values based on the cluster characteristics.
- **Feature Engineering**: Additional features such as **duration in months** are extracted from course duration descriptions. This allows for better prediction and recommendation accuracy.
- **Data Visualization**: Various visualization techniques are applied to better understand the dataset and relationships between features. Visualizations like bar charts, histograms, and scatter plots are used to explore course distribution, sub-category trends, and feature correlations.

### Exploratory Data Analysis (EDA):
- **Data Cleaning**: In the EDA phase, the dataset is thoroughly cleaned to remove unnecessary or duplicate information.
- **Handling Missing Values**: Columns with **90% or more missing data** are dropped to improve the dataset's quality and efficiency in modeling.
- **Visual Exploration**: Key statistics and visualizations are generated, including the distribution of sub-categories, correlation of features, and missing value patterns, to guide the feature engineering process.
- **Further Feature Engineering and Data Visualization**: Additional feature engineering techniques and more advanced visual exploration were performed on the dataset to improve the classification and recommendation models' accuracy. For more details, refer to the `EDA_and_Feature_Engineering.ipynb` file.

## Technologies Used

- **Python**: Primary programming language for implementing the system.
- **Pandas**: Used for data manipulation and cleaning.
- **Scikit-learn**: For feature extraction (TF-IDF) and machine learning models (Random Forest, KMeans).
- **Natural Language Processing (NLP)**: Text cleaning, tokenization, lemmatization, and stopword removal.
- **NLTK**: Used for lemmatization and stopword removal.
- **Pickle**: For saving and loading the trained models and vectorizers.
- **Cosine Similarity**: Used to measure the similarity between courses.
- **Matplotlib / Seaborn**: Used for data visualization and exploration.

- ### Model Evaluation and Results

The **Random Forest Classifier** was trained to predict the **sub-category** of a course based on features such as course title, skills, course type, and duration. The model's performance was evaluated using the following metrics:

- **Accuracy**: 0.83
- **Precision**: 0.82
- **Recall**: 0.83
- **F1-Score**: 0.81

