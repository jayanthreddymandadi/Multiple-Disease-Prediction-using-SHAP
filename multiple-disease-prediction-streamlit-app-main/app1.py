import streamlit as st
import matplotlib.pyplot as plt

# Create a sample figure
fig, ax = plt.subplots()
ax.plot([1, 2, 3, 4], [10, 20, 25, 30])  # Simple line plot
ax.set_title("Test Plot")

# Display the plot in Streamlit
st.pyplot(fig)