import streamlit as st
from PIL import Image
import datetime
import time
import pandas as pd

# Title of the app
st.title("Machine Learning Project")

# Displaying an image
img = Image.open("abc.jpg")
st.image(img, width=300, caption="Sample Image")

# Video (commented out as video is optional)
# vid_file = open("sample.mov", "rb")
# vid_bytes = vid_file.read()
# st.video(vid_bytes)

# Audio (commented out as audio is optional)
# aud_file = open("pop.mp3", "rb").read()
# st.audio(aud_file)

# Checkbox (show/hide widget)
if st.checkbox("Show/Hide"):
    st.text("Showing or hiding widget")

# Radio button for status
status = st.radio("What is your status?", ("Active", "Inactive"))

# Display status message based on radio selection
if status == "Active":
    st.success("You are active")
else:
    st.warning("You are inactive")

# Selectbox for occupation
occupation = st.selectbox("Your occupation", ["Doctor", "Engineer", "Teacher"])
st.write("You selected this option:", occupation)

# Multiselect for location preferences
location = st.multiselect("Where do you prefer to work?", ["Mumbai", "Pune", "Nashik", "Delhi"])

# Display selected locations and count
st.write("You selected", len(location), "locations")

# Slider for skill level
level = st.slider("What is your skill level?", 1, 5)

# Buttons with different actions
if st.button("Simple Button"):
    st.text("Button clicked")

if st.button("About"):
    st.text("Streamlit is a great framework for building web apps")

if st.button("Submit"):
    st.text("Submit success")

# Text input for first name
first_name = st.text_input("Enter your first name", "Type here")
if st.button("Submit", key="1"):
    result = first_name.title()
    st.success(f"Hello, {result}!")

# Text area for message input
message = st.text_area("Enter your message", "Type here")
if st.button("Submit", key="2"):
    result = message.title()
    st.success(f"Your message: {result}")

# Date input
today = st.date_input("Today is", datetime.datetime.now())

# Time input
the_time = st.time_input("What time is it?", datetime.time())

# Display JSON data
st.text("Displaying JSON data:")
st.json({"name": "Rahul", "gender": "Male"})

# Display code in a text block
st.text("Displaying raw code:")
st.code("import numpy as np")

# Display raw code block with DataFrame creation
with st.echo():
    df = pd.DataFrame()

# Progress bar
my_bar = st.progress(0)
for p in range(10):
    my_bar.progress(p + 10)

# Spinner to show loading state
with st.spinner("Waiting..."):
    time.sleep(5)
st.success("Finished!")
