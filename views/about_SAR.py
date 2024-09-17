import streamlit as st

st.title(" :orange[About SAR]")

# what is SAR image
st.write("\n")
st.subheader(" What exactly are SAR images ?",anchor=False)
st.write(
    """ Synthetic Aperture Radar (SAR) images are a type of radar imaging that uses the motion of a 
    radar sensor (often mounted on satellites or aircraft) to create high-resolution, two-dimensional 
    images of the Earth's surface. Unlike optical imaging, SAR works in all weather conditions, day or 
    night, since it uses microwave signals that penetrate clouds and operate independently of sunlight.
    SAR sensors emit microwave pulses and measure the reflected signals to form images. 
    The "synthetic aperture" refers to the technique of combining multiple radar echoes as the sensor 
    moves, simulating a larger antenna for higher resolution. SAR images are widely used in remote 
    sensing for applications like land mapping, monitoring natural disasters, agriculture, and 
    military surveillance due to their ability to detect surface characteristics, even under 
    challenging conditions."""
)

# why need of SAR image enhancement
st.write("\n")
st.subheader("Why need of SAR image enhancement ?",anchor=False)
st.write(
    """Color enhancement in SAR images is needed to improve visualization and interpretation of the
    radar data. SAR images are often grayscale, which can make it difficult to distinguish between
    objects with similar reflectivity. Applying color helps highlight subtle differences and 
    enhances contrast between features like water, vegetation, and urban areas.
    SAR data also involves multiple polarizations (e.g., HH, HV, VV), and color mapping allows these 
    channels to be represented in a single image, making it easier to interpret different scattering 
    mechanisms. This makes complex features, like urban areas or rough terrain, more identifiable 
    and aids in better detection, analysis, and classification of objects."""
)
st.markdown("---")

# creating columns and add images to it
col1,col2=st.columns(2)
col1.markdown("**This is a unprocessed SAR image**")
col1.image("assests/simple.jpg",caption="SAR image",use_column_width=True)
col2.markdown("**This is a processed SAR image**")
col2.image("assests/edited.jpg",caption="Enhanced SAR image",use_column_width=True)

st.markdown("---")
text1=''' Here are the key points where **colored SAR images** are more helpful than grayscale SARs:
1. :orange[**Enhanced Feature Differentiation**]: Color helps distinguish between land types (e.g., water, vegetation, urban areas) more clearly.
2. :orange[**Improved Contrast**]: Different colors enhance contrast between objects with similar reflectivity, making subtle features stand out.
3. :orange[**Multi-Polarization Visualization**]: Color can represent multiple polarizations (e.g., HH, HV, VV) in a single image, improving interpretation of complex data.
4. :orange[**Better Object Detection**]: Colors help in identifying and classifying objects, improving detection accuracy.
5. :orange[**Easier Pattern Recognition**]: Patterns in terrain, vegetation, or urban layouts are easier to spot with color-enhanced images. '''

st.markdown(text1)
st.markdown("---")
