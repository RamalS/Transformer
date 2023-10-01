import streamlit as st
from transformer import load_transformer, translate

saved_transformer = load_transformer("2e98b75a")

# sada ima toliko novaca da može kupiti sve, sve što god zaželi!


st.title('Prevoditelj sa hrvatskog na engleski')
st.caption("Unesi rečenicu na hrvatskom jeziku i pritisni enter. Rečenica će biti prevedena na engleski jezik.")

sentence = st.text_input('Input sentence', '')

if sentence:
    translated = translate(sentence, saved_transformer)
    st.write("**Prijevod rečenice:** ", translated)