import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Title of the app
st.title("Happy Birthday 🎉")

# Display balloons to celebrate the birthday
st.balloons()

# Sidebar content
st.sidebar.header("About This Project")
st.sidebar.text("This is our demo project for Streamlit. Enjoy exploring it!")

# Select box for algorithm choice
algo = st.sidebar.selectbox("Choose an algorithm", ["Linear Regression", "SVM", "Random Forest"])

# Display chosen algorithm
st.sidebar.write(f"You selected {algo} algorithm.")

# Input form for user details
st.header("User Information Form")
name = st.text_input("Enter your name:")
age = st.number_input("Enter your age:", min_value=0, max_value=100, value=25)
email = st.text_input("Enter your email:")

# Submit button for form
if st.button("Submit"):
    st.write(f"Thank you for submitting your details, {name}!")
    st.write(f"Age: {age}, Email: {email}")

# Displaying a dataframe with some data
st.header("Sample DataFrame")

# Creating a sample dataframe
data = {
    "Name": ["Alice", "Bob", "Charlie", "David"],
    "Age": [24, 30, 35, 28],
    "Occupation": ["Engineer", "Doctor", "Artist", "Scientist"]
}
df = pd.DataFrame(data)

# Display dataframe
st.write(df)

# Line chart example
st.header("Line Chart of Sample Data")
x = np.arange(0, 10, 0.1)
y = np.sin(x)
plt.plot(x, y)
st.pyplot()

# Bar chart example
st.header("Bar Chart of Sample Occupation Data")
occupation_counts = df["Occupation"].value_counts()
st.bar_chart(occupation_counts)

# Checkboxes for user preferences
st.header("User Preferences")
if st.checkbox("Show additional info"):
    st.write("You selected to show additional info!")
    st.text("Here is some extra information about Streamlit and how to use it.")

# Slider to select level of experience
st.header("Rate your experience with Streamlit")
experience = st.slider("Select your experience level:", 1, 5)
st.write(f"Your experience level is: {experience}")

# Image upload functionality
st.header("Upload an Image")
uploaded_file = st.file_uploader("Choose a file", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

# Display JSON data
st.header("JSON Data Example")
st.json({
    "name": "John Doe",
    "age": 30,
    "skills": ["Python", "Machine Learning", "Data Science"]
})

# Displaying raw code
st.header("Code Example")
st.code("""
import streamlit as st
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)
st.line_chart(y)
""")
