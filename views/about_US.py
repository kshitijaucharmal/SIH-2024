import streamlit as st

st.title("About US and Our Model")
col1,col2=st.columns(2)

col1.subheader(''' :orange[**About Us**] ''')
col1.markdown('''We are :orange[pre-final year Engineering students at PICT], interested in computer science.''')

col2.subheader(''' :orange[**About Our Model**]''')
col2.markdown('''In this model we have used :orange[GANs] and .....''')

st.markdown("\n")
st.markdown("Thank You :rose:")
