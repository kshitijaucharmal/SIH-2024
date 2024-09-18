import streamlit as st
from PIL import Image
from io import BytesIO

st.title(''' :orange[Try SAR enhancing here] ''')

MAX_FILE_SIZE = 200 * 1024 * 1024   # 200Mb

# Convert and save the image in memory for download
def get_image_download_bytes(img: Image.Image, file_format: str = 'JPEG'):
    buffer = BytesIO()
    img.save(buffer, format=file_format)
    buffer.seek(0)
    return buffer


uploaded_file = st.file_uploader("Upload a SAR image", type=["png", "jpg", "jpeg"])

# col1, col2 = st.columns(2)

# with col1:
#     st.write("Uploaded SAR Image")
#     st.image("SIH-2024/assests/simple.jpg", caption="SAR Image", use_column_width=True)

# with col2:
#     st.write("Colorized Image")
#     st.image("SIH-2024/assests/edited.jpg", caption="Colorized Image", use_column_width=True)
# creating columns and add images to it
col1,col2=st.columns(2)
col1.markdown("**This is a unprocessed SAR image**")
col1.image("assests/simple.jpg",caption="SAR image",use_column_width=True)
col2.markdown("**This is a processed SAR image**")
col2.image("assests/edited.jpg",caption="Enhanced SAR image",use_column_width=True)

colorized_image=Image.open("assests/edited.jpg")
# Convert the colorized image to bytes for download
img_bytes = get_image_download_bytes(colorized_image, 'JPEG')

# Create a download button for the colorized image
st.download_button(
label="Download Colorized Image",
data=img_bytes,
file_name="colorized_image.jpg",
mime="image/jpeg"
)        