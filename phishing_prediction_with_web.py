import pickle
import streamlit as st
from PIL import Image
#from fastapi import FastAPI
#import joblib,os

#app = FastAPI()

#pkl
#phish_model = open('phishing.pkl','rb')
#phish_model_ls = joblib.load(phish_model)
#loading the model

loaded_model = pickle.load(open('phishing.pkl','rb'))
# ML Aspect
#@app.get('/predict/{feature}')
def predict(features):
	X_predict = []
	X_predict.append(str(features))
	y_Predict = loaded_model.predict(X_predict)
	if y_Predict == 'bad':
		result = "This is a Phishing Site"
	else:
		result = "This is not a Phishing Site"
	return (features, result)
def main():
    #giving title
    st.title('Phishing Site Predicting App')
    #getting input from user
    image = Image.open('phishing image for streamlit.jpg')
    st.image(image, caption='Phishing Detection')
    #st.sidebar.title('Some tips')
    sidebar_image = Image.open('Data_security.jpg')
    st.sidebar.image(sidebar_image ,use_column_width=True)
    st.sidebar.header('How to protect your computer')
    st.sidebar.write('Below are some key steps to protecting your computer from intrusion:')
    st.sidebar.write('** 1. Keep Your Firewall Turned On **: A firewall helps protect your computer from hackers who might try to gain access \
 to crash it, delete information, or even steal passwords or other sensitive information. Software firewalls are \
 widely recommended for single computers. The software is prepackaged on some operating systems or can be purchased\
  for individual computers. For multiple networked computers, hardware routers typically provide firewall protection.')
    st.sidebar.write(
        ' ** 2. Install or Update Your Antivirus Software **: Antivirus software is designed to prevent malicious software programs from embedding on your computer.\
 If it detects malicious code, like a virus or a worm, it works to disarm or remove it. Viruses can infect computers without users’ knowledge. \
 Most types of antivirus software can be set up to update automatically.')
    st.sidebar.write(' ** 3. Install or Update Your Antispyware Technology**: Spyware is just what it sounds like—software that is\
  surreptitiously installed on your computer to let others peer into your activities on the computer.\
  Some spyware collects information about you without your consent or produces unwanted pop-up ads \
  on your web browser. Some operating systems offer free spyware protection, and inexpensive software is \
  readily available for download on the Internet or at your local computer store. Be wary of ads on the\
  Internet offering downloadable antispyware—in some cases these products may be fake and may actually contain\
 spyware or other malicious code. It’s like buying groceries—shop where you trust.')
    st.sidebar.write(' **4. Keep Your Operating System Up to Date **:  Computer operating systems are periodically updated to stay in tune with\
  technology requirements and to fix security holes. Be sure to install the updates to ensure your computer has\
  the latest protection.')
    st.sidebar.write(' **5. Turn Off Your Computer** : With the growth of high-speed Internet connections, many opt to leave \
 their computers on and ready for action. The downside is that being “always on” renders computers more\
  susceptible. Beyond firewall protection, which is designed to fend off unwanted attacks, \
  turning the computer off effectively severs an attacker’s connection—be it spyware or a botnet that\
  employs your computer’s resources to reach out to other unwitting users.')

    col1 = st.columns(1)
    col1_expander = st.expander("More Info")
    with col1_expander :
        st.subheader('About')
        st.write('Phishing is a form of cybercrime in which a target is contacted\
             via email, telephone, or text message by an attacker disguising as\
             a reputable entity or person. The attacker then lures individuals to\
             conterfeit websites to trick recipients into providing sensitive data.\
             The purpose of this application is to help individuals identify \
                 these phishing URLs in order to provide safer practices online.')
    url = st.text_input('Website url')
    predicted =''
    #creating button for prediction
    if st.button('Predict website result'):
        predicted = predict(url)
        if predicted[1] == 'This is a Phishing Site' :
            st.error(predicted)
        else:
            st.success(predicted)
if __name__ == '__main__':
	main()
#uvicorn prediction_app:app --reload
    
    
