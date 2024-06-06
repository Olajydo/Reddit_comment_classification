# Reddit comment classification
Here is the link to the streamlink classifier " https://redditcommentclassifier.streamlit.app/ "
This work compares the performance of a support vector machine (SVM) and a convolutional neural network (CNN) model to classify Reddit posts: "medical doctor," "veterinarian," or "other."

# Approach
1. Connection to PostgreSQL Database: Retrieve Reddit user names and information from Database using psycopg2 and put them in pandas DataFrame.
2. Data preprocessing
Label encoding: Manually encode text based on content.
Save to CSV: Save the labeled DataFrame to a CSV file for further processing.
3. Structural Engineering
SVM: Convert expressions to TF-IDF objects using TfidfVectorizer.
CNN: Tokenized and padded sequences using the Keras tokenizer.
4. Model training
SVM: Train the SVM model with linear ears.
CNN: Traditional CNN models using embedding, convolution, pooling, dense, dropout layers.
5. Sample analysis
Evaluate the models: Calculate the accuracy of both models and determine which performs better.
6. Save the predicted results
Predict labels: Use the best model to predict the labels in the test set.
Save the result: Save the predicted characters with user name and comments to a CSV file.
The best choice

After evaluating both models, SVM achieved slightly more on accuracy than CNN achieved did. SVM was chosen as the preferred model for this classification task due to its relatively high accuracy. Although the difference in accuracy is small, the excellent performance of SVM makes it a good choice for ensuring accuracy of prediction.
