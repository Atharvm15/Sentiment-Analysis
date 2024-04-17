## Documentation

## Introduction:

Sentiment analysis, also known as opinion mining, is a vital area of natural language processing (NLP) that focuses on understanding and classifying the sentiment expressed in textual data. In today's digital age, where vast amounts of textual data are generated through social media, customer reviews, and online discussions, sentiment analysis plays a crucial role in extracting valuable insights from this unstructured data.

The objective of sentiment analysis is to determine the sentiment polarity of text, which can be positive, negative, or neutral. By analyzing sentiment, businesses can gain valuable insights into customer opinions, preferences, and satisfaction levels, enabling them to make informed decisions, improve products and services, and enhance customer experiences.

In this project, we aim to perform sentiment analysis on a dataset containing textual reviews. We will preprocess the textual data to standardize it and prepare it for analysis. Using machine learning algorithms, including Random Forest, XGBoost, and Decision Tree classifiers, we will train models to classify the sentiment expressed in the reviews accurately.

Through this project, we seek to demonstrate the effectiveness of sentiment analysis in extracting valuable insights from textual data and its potential applications in various domains, including marketing, customer service, and product development. Additionally, we aim to evaluate the performance of different machine learning models in sentiment classification and identify the most suitable approach for our dataset.

## Project Objective:

Primary Objective:

The primary objective of this project is to perform sentiment analysis on textual data to classify the sentiment polarity of reviews accurately. Specifically, we aim to:

1. Preprocess the textual data: We will preprocess the textual data by removing noise, such as non-alphabetical characters and stopwords, converting text to lowercase, and stemming words to their root forms. This step is essential for standardizing the textual data and preparing it for analysis.

2. Train machine learning models: We will train multiple machine learning models, including Random Forest, XGBoost, and Decision Tree classifiers, to classify the sentiment expressed in the reviews. These models will learn patterns and relationships in the data to accurately predict the sentiment polarity of each review.

3. Evaluate model performance: We will evaluate the performance of each machine learning model using metrics such as accuracy, precision, recall, and F1-score. By comparing the performance of different models, we aim to identify the most effective approach for sentiment analysis on our dataset.

4. Generate insights: Once the models are trained and evaluated, we will use them to classify the sentiment polarity of reviews in the dataset. By analyzing the results, we aim to extract valuable insights into customer opinions, preferences, and satisfaction levels, which can inform decision-making processes and drive business strategies.

Overall, the primary objective of this project is to demonstrate the effectiveness of sentiment analysis in extracting meaningful insights from textual data and to identify the most suitable machine learning approach for sentiment classification in our specific context.

## Cell 1: Importing Libraries and Modules

This cell serves the purpose of importing essential Python libraries and configuring the environment for data analysis and visualization tasks.

- **numpy**: It is used for numerical computing, providing support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays.

- **pandas**: It is used for data manipulation and analysis, offering data structures and operations for manipulating numerical tables and time series data.

- **matplotlib.pyplot**: It is used for data visualization, providing a MATLAB-like interface for creating plots and graphs to visualize data.

- **seaborn**: It is used for enhanced data visualization, building on top of matplotlib to provide a high-level interface for drawing attractive and informative statistical graphics.

- **nltk**: It is used for natural language processing tasks, such as tokenization, stemming, and stopwords removal, providing a suite of libraries and programs for symbolic and statistical natural language processing.

- **scikit-learn**: It is used for machine learning algorithms and tools, offering simple and efficient tools for data mining and data analysis, including classification, regression, clustering, and dimensionality reduction.

- **wordcloud**: It is used for generating word clouds, which are visual representations of text data, where the size of each word indicates its frequency or importance within the text.


## Cell 2: Reading and Previewing the Dataset

In this cell, the dataset `amazon_alexa.tsv` is read into a Pandas DataFrame named `data` using the `read_csv()` function from the Pandas library. This dataset likely contains hourly bike rental data. After reading the dataset, the `head()` function is called on the DataFrame `data` to display the first few rows of the dataset. This provides a preview of the dataset's structure and allows users to inspect the column names, data types, and some sample values. By examining the first few rows, users can gain initial insights into the dataset's contents and determine how to proceed with data analysis and preprocessing tasks.


## Cell 3: Data Exploration

- #### Viewing Data (data.head())
This step is essential to get a quick overview of the dataset by displaying the first few rows. It helps in understanding the structure of the dataset and the type of information it contains.

- #### Checking Column Names
Knowing the column names is crucial for identifying the features available in the dataset. This information is necessary for data manipulation, analysis, and modeling tasks.

- #### Checking for Null Values
Detecting null values is a critical data preprocessing step to ensure data quality and integrity. Null values can affect the performance of machine learning models and must be handled appropriately.

- #### Dropping Null Values
Removing records with null values from the dataset is a common approach to dealing with missing data. This ensures that subsequent analyses and modeling are performed on clean and complete data.

- #### Dataset Shape After Dropping Null Values
Displaying the shape of the dataset after removing null values provides insights into the impact of data cleaning. It helps in assessing the extent of data loss and ensures that the dataset is still suitable for analysis.

- #### Creating a New Column 'length'
Adding a new column to the dataset can provide additional insights or features for analysis. In this case, the 'length' column is created to store the length of each string in the 'verified_reviews' column. This information can be useful for understanding the distribution of review lengths and its potential impact on analysis or modeling tasks.

## Cell 4: Data Visualization

#### Bar Plot for Rating Distribution
Generates a bar plot to visualize the total counts of each rating. The plot is colored red and includes a title ('Rating distribution count'), xlabel ('Ratings'), and ylabel ('Count'). This visualization offers insights into the distribution of ratings within the dataset, aiding in understanding the frequency of each rating category.

#### Percentage Distribution of Ratings
Calculates the percentage distribution of each rating category. This information is printed to show the proportion of records for each rating relative to the total number of records. It provides a quantitative understanding of the distribution of ratings in the dataset.

#### Pie Chart for Feedback Distribution
Produces a pie chart to visualize the total counts of each feedback category. The plot is colored blue and includes a title ('Feedback distribution count'), xlabel ('Feedback'), and ylabel ('Count'). This visualization helps in understanding the distribution of feedback categories within the dataset.

#### Percentage Distribution of Feedback
Computes the percentage distribution of each feedback category. The results are printed to demonstrate the proportion of records for each feedback category relative to the total number of records. This quantitative insight aids in understanding the distribution of feedback categories in the dataset.

## Cell 5: Feedback Analysis

#### Feedback = 0
This section retrieves the counts of ratings for feedback where the value is 0. It helps in understanding the distribution of ratings for negative feedback. Negative feedback typically indicates dissatisfaction or issues with the product, making it essential to analyze associated ratings.

#### Feedback = 1
Here, the counts of ratings for feedback where the value is 1 are retrieved. This aids in understanding the distribution of ratings for positive feedback. Positive feedback signifies satisfaction or positive experiences with the product, making it crucial to examine associated ratings.

### Variation Analysis

#### Distinct Values of 'Variation'
The analysis presents the count of distinct values in the 'variation' column. Understanding the different variations of products available in the dataset is important for assessing product diversity and potential impacts on customer preferences.

#### Bar Graph for Variation Distribution
This visualization provides a graphical representation of the total counts of each variation using a bar plot. It offers a clear overview of the distribution of product variations within the dataset, enabling quick identification of popular or common variations.

## Cell 6: Text Preprocessing and Model Training

This section of the code preprocesses the text data and trains multiple machine learning models for sentiment analysis. The process involves the following steps:

Text Preprocessing:
The code initializes an empty list called `corpus` to store preprocessed text data. It utilizes a Porter stemmer from NLTK to stem words. For each review in the dataset, the following preprocessing steps are performed: Removal of non-alphabetical characters using regular expressions, conversion of the review to lowercase and splitting it into individual words, stemming of each word using the Porter stemmer, and joining of the stemmed words back into a single string, which is then appended to the `corpus`.

Vectorization:
The `CountVectorizer` from scikit-learn is used to convert the text data into numerical vectors. The maximum number of features is set to 2500 to limit the dimensionality of the vectorized data.

Model Training:
The vectorized data (`X`) is split into independent and dependent variables. The independent variables are then scaled using MinMaxScaler. Several machine learning models, including Random Forest Classifier, XGBoost Classifier, and Decision Tree Classifier, are trained on the scaled training data (`X_train_scl`) and corresponding labels (`y_train`). The accuracy of each model is evaluated on both the training and testing datasets. Additionally, cross-validation is performed to assess the model's performance across multiple folds. Hyperparameter tuning using grid search is employed to find the best combination of parameters for the Random Forest Classifier model. Finally, confusion matrices are generated to evaluate the classification performance of each model on the testing data.

## Conclusion:

In conclusion, the sentiment analysis project accomplished the task of processing textual data and training machine learning models for sentiment classification. Through comprehensive text preprocessing steps, including the removal of non-alphabetical characters, lowercase conversion, word tokenization, and stemming, the textual data was standardized and prepared for analysis. This preprocessing facilitated the reduction of data dimensionality, ensuring efficient model training.

Utilizing the CountVectorizer, the textual data was transformed into numerical vectors, enabling the machine learning models to effectively process and analyze sentiment. The trained models, such as the Random Forest Classifier, XGBoost Classifier, and Decision Tree Classifier, exhibited varying levels of accuracy in sentiment classification, demonstrating the importance of model selection and tuning in achieving optimal performance.

Furthermore, the project employed cross-validation techniques to assess model performance across multiple folds and utilized grid search for hyperparameter tuning, enhancing the models' predictive capabilities. Confusion matrices were generated to evaluate the classification performance of each model on the testing data, providing insights into their strengths and weaknesses.

Overall, the sentiment analysis project showcased the significance of text preprocessing, vectorization, and model training in accurately classifying sentiment from textual data. The project outcomes offer valuable insights for sentiment analysis tasks across various domains, aiding in decision-making processes and enhancing user experiences.