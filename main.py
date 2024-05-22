import streamlit as st
import pickle

model_path = "C:/Users/M.I.S SITHLTD/Desktop/olajide/reddit_comment_classification/svm_classifier_model.pkl"
vect_path = "tfidf_vectorizer.pkl"

try:
    # Open the model file in binary read mode and load the model
    with open(model_path, "rb") as model_file:
        svm_model = pickle.load(model_file)

    # Open the vectorizer file in binary read mode and load the vectorizer
    with open(vect_path, "rb") as vect_file:
        vectorizer = pickle.load(vect_file)

except Exception as e:
    print(f"Error loading model or vectorizer: {e}")
    st.error("An error occurred loading the model or vectorizer. Please check the console for details.")
    exit(1)

def main():
    st.title("Reddit Comment Classifier")

    # Input variable
    comment = st.text_area("Enter a Reddit comment:")

    # Prediction code
    if st.button("Predict"):
        # Transform reddit comment with vectorizer
        vect_comment = vectorizer.transform([comment])  # Use transform instead of fit_transform

        # Predict using the SVM model
        prediction = svm_model.predict(vect_comment)

        # Display the prediction
        st.success(f"This individual is a: {prediction[0]}")

# Required to let Streamlit instantiate our web app.
if __name__ == '__main__':
    main()
