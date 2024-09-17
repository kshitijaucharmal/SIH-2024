import streamlit as st

st.set_page_config(layout="wide",page_title="Milkdromeda")  #new
#-- page setup
# take a note that streamlit used '\' as escape sequence so even if you are in windows use '/' in paths 
home_page=st.Page(
    page="views/home.py",
    title="SAR image enhancer ",
    icon="😊",
    default=True,
)

aboutSAR_page=st.Page(
    page="views/about_SAR.py",
    title="What is SAR ?",
    icon="🤓",
)

aboutUS_page=st.Page(
    page="views/about_US.py",           
    title="Our team",
    icon="😎",
)

#---creating a navigation menu

# Navigation setup( Without sections)
# pg=st.navigation(pages=[home_page,aboutSAR_page,aboutUS_page])

# Navigation setup(With sections)
pg=st.navigation(
    {
        "Home":[home_page],
        "Info":[aboutSAR_page,aboutUS_page],
    }
)

# to be shared on all the pages 
# st.logo("assests/New Project.png")    #the logo is too small
st.sidebar.text("Made by team Milkdromeda")

# Run Navigation
pg.run()