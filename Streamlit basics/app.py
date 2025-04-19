import streamlit as st
from PIL import Image

# Title of the app
st.title("Streamlit Tutorial")

# Header and Subheader
st.header("This is a Header")
st.subheader("This is a Subheader")

# Plain Text
st.text("Hello, this is a plain text message.")

# Markdown Text
st.markdown(" # First Markdown - This is a Markdown heading")

# Display messages with different types of texts
st.success("Operation was successful!")
st.info("This is an informational message.")
st.warning("This is a warning message!")
st.error("An error occurred.")

# Displaying an exception
st.exception("NameError('name not defined')")

# Using the help function to show details of the 'range' function
st.help(range)

# Using st.write to display both text and data
st.write("Text with write function:")
st.write(range(10))

# Displaying an image
img = Image.open("abc.jpg")
st.image(img, caption="Sample Image")
