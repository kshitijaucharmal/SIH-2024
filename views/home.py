import streamlit as st
from PIL import Image
import io

# Dummy function to "colorize" the SAR image (replace with your deep learning model)
def colorize_image(image: Image.Image) -> Image.Image:
    # Convert the SAR image to RGB for demonstration
    colorized_image = image.convert('RGB')  # Simulated colorization (replace this with model output)
    return colorized_image

# Function to convert image to byte array for download
def get_image_download_bytes(img: Image.Image, file_format: str = 'JPEG'):
    buffer = io.BytesIO()
    img.save(buffer, format=file_format)
    buffer.seek(0)
    return buffer


st.title("SAR Image Colorization")

# Image upload
uploaded_file = st.file_uploader("Upload a SAR image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Split the layout into two columns
    col1, col2 = st.columns(2)

    # Display the uploaded SAR image in the first column
    with col1:
        st.write("Uploaded SAR Image")
        image = Image.open(uploaded_file)
        st.image(image, caption="SAR Image", use_column_width=True)

    # Call the colorize function (replace this with your model inference)
    colorized_image = colorize_image(image)

    # Display the colorized image in the second column
    with col2:
        st.write("Colorized Image")
        st.image(colorized_image, caption="Colorized Image", use_column_width=True)

    # Convert the colorized image to bytes for download
    img_bytes = get_image_download_bytes(colorized_image, 'JPEG')

    # Create a download button for the colorized image
    st.download_button(
        label="Download Colorized Image",
        data=img_bytes,
        file_name="colorized_image.jpg",
        mime="image/jpeg"
    )
else:
    st.write("Please upload a SAR image.")


# import streamlit as st
# from PIL import Image
# from io import BytesIO
# from rembg import remove
# import base64

# st.title(''' :orange[Try SAR enhancing here] :smile:''')

# """MAX_FILE_SIZE=200*1024*1024   #200Mb

# #Download the enhanced image
# def convert_img(img):
#     buf=BytesIO()
#     img.save(buf,format="PNG")
#     byte_im=buf.getvalue()
#     return byte_im

# col1,col2=st.columns(2) 

# def my_model(upload):
#     # Img=st.image("assests/simple.jpg", caption="Sample SAR Image", use_column_width=True)
#     img=Image.open(upload)
#     col1.write(" This is original SAR image")
#     col1.image(img)

#     # Img2=st.image("assests/edited.jpg", caption="Enhanced SAR Image", use_column_width=True)
#     #processing the image
#     output_img=

#     # get image2 from the model
#     col2.write(" This is enhanced SAR image")
#     col2.image(output_img)

# # this is to input file
# uploaded_photo=st.file_uploader("Upload the SAR image here 📷",type=["png","jpg","jpeg"])

# #this is to download the output file
# st.markdown("\n") #adds a line
# st.download_button("Download Colorized SAR image",convert_img(output_img),"coloredSAR.png","image.png")

# if uploaded_photo is not None:
#     if uploaded_photo.size > 2e+8:
#         st.error("The uploaded file is too large. Upload file smaller than 200MB ")
#     else:
#         my_model(upload=uploaded_photo)
# else:
#     my_model("assests/simple.jpg")
# """

#     # def fix_img(upload):
# #     image=Image.open(upload)
# #     col1.write("Original Image :camera:")
# #     col1.image(image)

# #     fixed=remove(image)

# #     col2.write("Bgfree Image :camera:")
# #     col2.image(fixed)
#     # st.markdown("\n") #adds a line
# #     st.download_button("Download bgfree image",convert_img(fixed),"fixed.png","image.png")

# # my_upload=st.file_uploader("Upload an image", type=["png","jpg","jpeg"])
# #error handling
# # error handling with uploaded file