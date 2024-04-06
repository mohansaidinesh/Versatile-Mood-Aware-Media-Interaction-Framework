import streamlit as st
from streamlit_option_menu import option_menu
import requests
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner
import pyowm
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import pandas as pd
import streamlit as st
import pandas as pd
import streamlit as st
from deepface import DeepFace
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import keras
import tensorflow
import random
from datetime import datetime, timedelta
from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer
import streamlit as st
import cv2
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import time
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import urllib
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
import speech_recognition as sr
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import warnings
from bs4 import BeautifulSoup  
warnings.filterwarnings('ignore')
owm = pyowm.OWM('11081b639d8ada3e97fc695bcf6ddb20')
from PIL import Image
import time
lang = "English"
def search_and_display_images(query, num_images=50):
    try:
        k=[]  
        idx=0  
        url = f"https://www.google.com/search?q={query}&tbm=isch"  
        response = requests.get(url) 
        soup = BeautifulSoup(response.text, "html.parser")  
        images = []  
        for img in soup.find_all("img"):  
            if len(images) == num_images: 
                break
            src = img.get("src")  
            if src.startswith("http") and not src.endswith("gif"):  
                images.append(src)  
        for image in images:  
            k.append(image)  
        idx = 0  
        while idx < len(k):
            for _ in range(len(k)): 
                cols = st.columns(4)  
                cols[0].image(k[idx], width=150)  
                idx += 1 
                cols[1].image(k[idx], width=150)
                idx += 1  
                cols[2].image(k[idx], width=150)  
                idx += 1  
                cols[3].image(k[idx], width= 150)  
                idx = idx + 1  
    except:
         pass
def get_movie_details(title):
    api_key = "bca49c66" 
    url = f"http://www.omdbapi.com/?apikey={api_key}&t={title}"
    response = requests.get(url)
    if response.status_code == 200:
        movie_details = response.json()
        return movie_details
    else:
        return None
def analyze_emotion(image):
    # Convert the PIL image to OpenCV format
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    # Analyze emotion using DeepFace
    res = DeepFace.analyze(img_cv, actions=['emotion'])
    # Extract the emotion
    data_dict = res[0]['emotion']
    # Find the maximum key-value pair
    max_pair = max(data_dict.items(), key=lambda x: x[1])
    return max_pair[0]
data = pd.read_csv("Text Emotion Detection/EX 2/train.txt", sep=';')
data.columns = ["Text", "Emotions"]

texts = data["Text"].tolist()
labels = data["Emotions"].tolist()

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
max_length = max([len(seq) for seq in sequences])

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
json_file = open("Text Emotion Detection/EX 2/model_architecture.json", "r")
loaded_model_json = json_file.read()
json_file.close()

# Load the saved model weights
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("Text Emotion Detection/EX 2/model_weights.h5")
st.set_page_config(page_title = 'Emotion Detection', 
        layout='wide',page_icon=":smiley:", initial_sidebar_state="expanded")
with st.sidebar:
    selected = option_menu("DashBoard", ["Home",'Text','Image','Speech'], 
        icons=['house','card-text','card-image','headset'], menu_icon="cast", default_index=0,
        styles={
        "nav-link-selected": {"background-color": "red"},
    })
def load_lottieurl(url: str):
            r = requests.get(url)
            if r.status_code != 200:
                return None
            return r.json()
if selected=='Home':
        st.markdown(f"<h1 style='text-align: center;font-size:60px;color:red;'>Versatile Mood Aware Media Interaction Framework</h1>", unsafe_allow_html=True)
        lottie_url = "https://lottie.host/f58252b9-2c23-439c-a817-e5434768f2f7/Kzr033jFd8.json"
        lottie_json = load_lottieurl(lottie_url)
        st_lottie(lottie_json,width=900,height=400)
if selected=='Text':
        st.markdown(f"<h1 style='text-align: center;font-size:60px;color:red;'>Text Emotion Detection</h1>", unsafe_allow_html=True)
        text = st.text_input('Enter Text')
        if text:
            input_sequence = tokenizer.texts_to_sequences([text])
            padded_input_sequence = pad_sequences(input_sequence, maxlen=max_length)
            prediction = loaded_model.predict(padded_input_sequence)
            predicted_label = label_encoder.inverse_transform([np.argmax(prediction[0])])[0]
            select = option_menu("",["Movies","Music",'Quotes','Images'], 
                icons=['clipboard', 'diagram-3-fill','file-earmark-image'],default_index=0, orientation="horizontal",
                styles={
                "container": {"padding": "0!important", "background-color": "white"},
                "icon": {"color": "DarkMagenta", "font-size": "15px"}, 
                "nav-link": {"font-size": "15px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                "nav-link-selected": {"background-color": "green"},
            })
            if select=='Movies':
                try:
                    genre = st.radio(
                        "Select Language",
                        [":rainbow[Telugu]", ":rainbow[Hindi]", ":rainbow[English]"],
                        index=None,
                    )
                    if genre==":rainbow[Telugu]":
                        data=pd.read_csv("Data/Movies/Telugu.csv",encoding='unicode_escape')
                    if genre==":rainbow[Hindi]":
                        data=pd.read_csv("Data/Movies/Hindi.csv",encoding='unicode_escape')
                    if genre==":rainbow[English]": 
                        data = pd.read_csv("Data/Movies/English.csv",encoding='unicode_escape')
                    data['Emotion']=data['Emotion'].apply(lambda x: x.lower())
                    res = data[data["Emotion"].isin([predicted_label])]
                    col = res['Movie Name'].tolist()
                    for i in col:
                        m = get_movie_details(i)
                        try:
                            col1, col2,col4= st.columns([2,9,4])
                            col4.image(m['Poster'], width=250)
                            if m['Released']!='N/A':
                                with col1:
                                    st.markdown(f"<p style='color:red;'>{'Released:  '}</p>", unsafe_allow_html=True)
                                col2.write(m['Released'])
                            if m['Genre']!='N/A':
                                with col1:
                                    st.markdown(f"<p style='color:red;'>{'Genre:  '}</p>", unsafe_allow_html=True)
                                col2.write(m['Genre'])
                            if m['Director']!='N/A':
                                with col1:
                                    st.markdown(f"<p style='color:red;'>{'Director:  '}</p>", unsafe_allow_html=True)
                                col2.write(m['Director'])
                            if m['Writer']!='N/A':
                                with col1:
                                    st.markdown(f"<p style='color:red;'>{'Writer:  '}</p>", unsafe_allow_html=True)
                                col2.write(m['Writer'])
                            if m['Actors']!='N/A':
                                with col1:
                                    st.markdown(f"<p style='color:red;'>{'Actors:  '}</p>", unsafe_allow_html=True)
                                col2.write(m['Actors'])
                            if m['Awards']!='N/A':
                                with col1:
                                    st.markdown(f"<p style='color:red;'>{'Awards:  '}</p>", unsafe_allow_html=True)
                                col2.write(m['Awards'])
                            if m['imdbRating']!='N/A':
                                with col1:
                                    st.markdown(f"<p style='color:red;'>{'IMDbRating:  '}</p>", unsafe_allow_html=True)
                                col2.write(m['imdbRating'])
                            if m['BoxOffice']!='N/A':
                                with col1:
                                    st.markdown(f"<p style='color:red;'>{'Collection:  '}</p>", unsafe_allow_html=True)
                                col2.write(m['BoxOffice'])
                            if m['Production']!='N/A':
                                with col1:
                                    st.markdown(f"<p style='color:red;'>{'Production:  '}</p>", unsafe_allow_html=True)
                                col2.write(m['Production'])
                            if m['Plot']!='N/A':
                                with col1:
                                    st.markdown(f"<p style='color:red;'>{'About:  '}</p>", unsafe_allow_html=True)
                                col2.write(m['Plot'])
                            st.divider()
                        except:
                            pass
                except:
                    pass
            if select=='Music':
                try:
                    genre = st.radio(
                        "Select Language",
                        [":rainbow[Telugu]", ":rainbow[Hindi]", ":rainbow[English]"],
                        index=None,
                    )
                    if genre==":rainbow[Telugu]":
                        data=pd.read_csv("Data/Music/Telugu.csv",encoding='unicode_escape')
                    if genre==":rainbow[Hindi]":
                        data=pd.read_csv("Data/Music/Hindi.csv",encoding='unicode_escape')
                    if genre==":rainbow[English]": 
                        data = pd.read_csv("Data/Music/English.csv",encoding='unicode_escape')
                    data['Emotion']=data['Emotion'].apply(lambda x: x.lower())
                    res = data[data["Emotion"].isin([predicted_label])]
                    col = res['Song Name'].tolist()
                    for i in col:
                        client_id = 'dc0f03d0101a4e20a12944f16ca15be8'
                        client_secret = '2887fb77391440c3bf45d6118bb0ad68'
                        sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=client_id, client_secret=client_secret))
                        result = sp.search(q=i, limit=1)
                        if len(result['tracks']['items'])!=0:
                            track = result['tracks']['items'][0]
                            preview_url = track['preview_url']
                            name = track['name']
                            artist = track['artists'][0]['name']
                            album_image_url = track['album']['images'][0]['url']
                            response = requests.get(album_image_url)
                            image = Image.open(BytesIO(response.content))
                            colors = ['Red', 'Blue', 'Green', 'Yellow', 'Orange', 'Purple', 'Pink', 'Turquoise', 'Magenta', 'Cyan', 'Brown', 'Gray', 'Black', 'Lavender', 'Beige', 'Olive', 'Maroon', 'Navy', 'Teal']
                            color = random.choice(colors)
                            st.write(f"<h1 style='color: {color};'>{name}</h1>", unsafe_allow_html=True)
                            col1,col2,col3,col4= st.columns([2.5,10,5,0.5])
                            col3.image(image,width=250)
                            with col1:
                                st.markdown(f"<h5 style='color:red';>{'Movie :'}</h5>", unsafe_allow_html=True)
                                st.markdown(f"<h5 style='color:red';>{'Artist  :'}</h5>", unsafe_allow_html=True)
                            with col2:
                                st.markdown(f"<p style='color:black';>{track['album']['name']}</p>", unsafe_allow_html=True)
                                st.markdown(f"<p style='color:black';>{artist}</p>", unsafe_allow_html=True)
                                st.audio(preview_url)                            
                except:
                    pass
            if select=='Quotes':
                pass
            if select=='Images':
                predicted_label = text+ predicted_label + "memes"
                search_and_display_images(predicted_label)
        else:
            st.write("Please Enter Text")
                
if selected=='Image':
        st.markdown(f"<h1 style='text-align: center;font-size:60px;color:red;'>Facial Emotion Detection</h1>", unsafe_allow_html=True)
        # Create a VideoCapture object
        try:
            cap = cv2.VideoCapture(0)
            
            # Check if the webcam is opened correctly
            if not cap.isOpened():
                st.error("Unable to access camera")
                
            st.text("Press 'Capture' to take a snapshot")
            
            
            # Capture a frame
            ret, frame = cap.read()
            if ret:
                # Convert the frame from BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Convert the captured frame to PIL Image
                captured_image = Image.fromarray(frame)
                # Analyze emotion in the captured image
                emotion = analyze_emotion(captured_image)
                # Display the captured image
                # Display the detected emotion
                # Save the captured image
                captured_image.save("captured_image.jpg")
            else:
                st.error("Failed to capture image")
            
            # Release the VideoCapture object
            cap.release()
            if emotion=='neutral':
                emotion='happy'
            if emotion=="sad":
                emotion='sadness'
            if emotion =='happy':
                emotion = 'joy'
            data = pd.read_csv("Data/Movies/English.csv",encoding='unicode_escape')
            data['Emotion']=data['Emotion'].apply(lambda x: x.lower())
            res = data[data["Emotion"].str.contains(emotion, case=False, na=False)]
            col = res['Movie Name'].tolist()
            for i in col:
                m = get_movie_details(i)
                try:
                    col1, col2,col4= st.columns([2,9,4])
                    col4.image(m['Poster'], width=250)
                    if m['Released']!='N/A':
                        with col1:
                            st.markdown(f"<p style='color:red;'>{'Released:  '}</p>", unsafe_allow_html=True)
                        col2.write(m['Released'])
                    if m['Genre']!='N/A':
                        with col1:
                            st.markdown(f"<p style='color:red;'>{'Genre:  '}</p>", unsafe_allow_html=True)
                        col2.write(m['Genre'])
                    if m['Director']!='N/A':
                        with col1:
                            st.markdown(f"<p style='color:red;'>{'Director:  '}</p>", unsafe_allow_html=True)
                        col2.write(m['Director'])
                    if m['Writer']!='N/A':
                        with col1:
                            st.markdown(f"<p style='color:red;'>{'Writer:  '}</p>", unsafe_allow_html=True)
                        col2.write(m['Writer'])
                    if m['Actors']!='N/A':
                        with col1:
                            st.markdown(f"<p style='color:red;'>{'Actors:  '}</p>", unsafe_allow_html=True)
                        col2.write(m['Actors'])
                    if m['Awards']!='N/A':
                        with col1:
                            st.markdown(f"<p style='color:red;'>{'Awards:  '}</p>", unsafe_allow_html=True)
                        col2.write(m['Awards'])
                    if m['imdbRating']!='N/A':
                        with col1:
                            st.markdown(f"<p style='color:red;'>{'IMDbRating:  '}</p>", unsafe_allow_html=True)
                        col2.write(m['imdbRating'])
                    if m['BoxOffice']!='N/A':
                        with col1:
                            st.markdown(f"<p style='color:red;'>{'Collection:  '}</p>", unsafe_allow_html=True)
                        col2.write(m['BoxOffice'])
                    if m['Production']!='N/A':
                        with col1:
                            st.markdown(f"<p style='color:red;'>{'Production:  '}</p>", unsafe_allow_html=True)
                        col2.write(m['Production'])
                    if m['Plot']!='N/A':
                        with col1:
                            st.markdown(f"<p style='color:red;'>{'About:  '}</p>", unsafe_allow_html=True)
                        col2.write(m['Plot'])
                    st.divider()
                except:
                    pass
            data = pd.read_csv("Data/Music/English.csv",encoding='unicode_escape')
            data['Emotion']=data['Emotion'].apply(lambda x: x.lower())
            res = data[data["Emotion"].isin([emotion])]
            col = res['Song Name'].tolist()
            for i in col:
                client_id = 'dc0f03d0101a4e20a12944f16ca15be8'
                client_secret = '2887fb77391440c3bf45d6118bb0ad68'
                sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=client_id, client_secret=client_secret))
                result = sp.search(q=i, limit=1)
                if len(result['tracks']['items'])!=0:
                    track = result['tracks']['items'][0]
                    preview_url = track['preview_url']
                    name = track['name']
                    artist = track['artists'][0]['name']
                    album_image_url = track['album']['images'][0]['url']
                    response = requests.get(album_image_url)
                    image = Image.open(BytesIO(response.content))
                    colors = ['Red', 'Blue', 'Green', 'Yellow', 'Orange', 'Purple', 'Pink', 'Turquoise', 'Magenta', 'Cyan', 'Brown', 'Gray', 'Black', 'Lavender', 'Beige', 'Olive', 'Maroon', 'Navy', 'Teal']
                    color = random.choice(colors)
                    st.write(f"<h1 style='color: {color};'>{name}</h1>", unsafe_allow_html=True)
                    col1,col2,col3,col4= st.columns([2.5,10,5,0.5])
                    col3.image(image,width=250)
                    with col1:
                        st.markdown(f"<h5 style='color:red';>{'Movie :'}</h5>", unsafe_allow_html=True)
                        st.markdown(f"<h5 style='color:red';>{'Artist  :'}</h5>", unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"<p style='color:black';>{track['album']['name']}</p>", unsafe_allow_html=True)
                        st.markdown(f"<p style='color:black';>{artist}</p>", unsafe_allow_html=True)
                        st.audio(preview_url)  
            st.divider()                          
            predicted_label = emotion + "memes"
            search_and_display_images(predicted_label)
            
            # Save the captured image
            captured_image.save("captured_image.jpg")    
        # Release the VideoCapture object
        except:
            pass
    


if selected=='Speech':
        st.markdown(f"<h1 style='text-align: center;font-size:60px;color:red;'>Speech Emotion Detection</h1>", unsafe_allow_html=True)
        # Function to transcribe speech to text
        def transcribe_audio():
            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                st.write("Listening...")
                audio_data = recognizer.listen(source)
                try:
                    text = recognizer.recognize_google(audio_data)
                    return text
                except sr.UnknownValueError:
                    return "Could not understand audio"
                except sr.RequestError as e:
                    return "Could not request results; {0}".format(e)
        
        # Button to start capturing voice input
        text = transcribe_audio()
        if text =="Could not understand audio":
            st.write("Please speak clearly and try again")
        else:
            if text:
                input_sequence = tokenizer.texts_to_sequences([text])
                padded_input_sequence = pad_sequences(input_sequence, maxlen=max_length)
                prediction = loaded_model.predict(padded_input_sequence)
                predicted_label = label_encoder.inverse_transform([np.argmax(prediction[0])])[0]
                select = option_menu("",["Movies","Music",'Quotes','Images'], 
                    icons=['clipboard', 'diagram-3-fill','file-earmark-image'],default_index=0, orientation="horizontal",
                    styles={
                    "container": {"padding": "0!important", "background-color": "white"},
                    "icon": {"color": "DarkMagenta", "font-size": "15px"}, 
                    "nav-link": {"font-size": "15px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                    "nav-link-selected": {"background-color": "green"},
                })
                if select=='Movies':
                    try:
                        
                        data = pd.read_csv("Data/Movies/English.csv",encoding='unicode_escape')
                        data['Emotion']=data['Emotion'].apply(lambda x: x.lower())
                        res = data[data["Emotion"].isin([predicted_label])]
                        col = res['Movie Name'].tolist()
                        for i in col:
                            m = get_movie_details(i)
                            try:
                                col1, col2,col4= st.columns([2,9,4])
                                col4.image(m['Poster'], width=250)
                                if m['Released']!='N/A':
                                    with col1:
                                        st.markdown(f"<p style='color:red;'>{'Released:  '}</p>", unsafe_allow_html=True)
                                    col2.write(m['Released'])
                                if m['Genre']!='N/A':
                                    with col1:
                                        st.markdown(f"<p style='color:red;'>{'Genre:  '}</p>", unsafe_allow_html=True)
                                    col2.write(m['Genre'])
                                if m['Director']!='N/A':
                                    with col1:
                                        st.markdown(f"<p style='color:red;'>{'Director:  '}</p>", unsafe_allow_html=True)
                                    col2.write(m['Director'])
                                if m['Writer']!='N/A':
                                    with col1:
                                        st.markdown(f"<p style='color:red;'>{'Writer:  '}</p>", unsafe_allow_html=True)
                                    col2.write(m['Writer'])
                                if m['Actors']!='N/A':
                                    with col1:
                                        st.markdown(f"<p style='color:red;'>{'Actors:  '}</p>", unsafe_allow_html=True)
                                    col2.write(m['Actors'])
                                if m['Awards']!='N/A':
                                    with col1:
                                        st.markdown(f"<p style='color:red;'>{'Awards:  '}</p>", unsafe_allow_html=True)
                                    col2.write(m['Awards'])
                                if m['imdbRating']!='N/A':
                                    with col1:
                                        st.markdown(f"<p style='color:red;'>{'IMDbRating:  '}</p>", unsafe_allow_html=True)
                                    col2.write(m['imdbRating'])
                                if m['BoxOffice']!='N/A':
                                    with col1:
                                        st.markdown(f"<p style='color:red;'>{'Collection:  '}</p>", unsafe_allow_html=True)
                                    col2.write(m['BoxOffice'])
                                if m['Production']!='N/A':
                                    with col1:
                                        st.markdown(f"<p style='color:red;'>{'Production:  '}</p>", unsafe_allow_html=True)
                                    col2.write(m['Production'])
                                if m['Plot']!='N/A':
                                    with col1:
                                        st.markdown(f"<p style='color:red;'>{'About:  '}</p>", unsafe_allow_html=True)
                                    col2.write(m['Plot'])
                                st.divider()
                            except:
                                pass
                    except:
                        pass
                if select=='Music':
                    try:
                    
                        data = pd.read_csv("Data/Music/English.csv",encoding='unicode_escape')
                        data['Emotion']=data['Emotion'].apply(lambda x: x.lower())
                        res = data[data["Emotion"].isin([predicted_label])]
                        col = res['Song Name'].tolist()
                        for i in col:
                            client_id = 'dc0f03d0101a4e20a12944f16ca15be8'
                            client_secret = '2887fb77391440c3bf45d6118bb0ad68'
                            sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=client_id, client_secret=client_secret))
                            result = sp.search(q=i, limit=1)
                            if len(result['tracks']['items'])!=0:
                                track = result['tracks']['items'][0]
                                preview_url = track['preview_url']
                                name = track['name']
                                artist = track['artists'][0]['name']
                                album_image_url = track['album']['images'][0]['url']
                                response = requests.get(album_image_url)
                                image = Image.open(BytesIO(response.content))
                                colors = ['Red', 'Blue', 'Green', 'Yellow', 'Orange', 'Purple', 'Pink', 'Turquoise', 'Magenta', 'Cyan', 'Brown', 'Gray', 'Black', 'Lavender', 'Beige', 'Olive', 'Maroon', 'Navy', 'Teal']
                                color = random.choice(colors)
                                st.write(f"<h1 style='color: {color};'>{name}</h1>", unsafe_allow_html=True)
                                col1,col2,col3,col4= st.columns([2.5,10,5,0.5])
                                col3.image(image,width=250)
                                with col1:
                                    st.markdown(f"<h5 style='color:red';>{'Movie :'}</h5>", unsafe_allow_html=True)
                                    st.markdown(f"<h5 style='color:red';>{'Artist  :'}</h5>", unsafe_allow_html=True)
                                with col2:
                                    st.markdown(f"<p style='color:black';>{track['album']['name']}</p>", unsafe_allow_html=True)
                                    st.markdown(f"<p style='color:black';>{artist}</p>", unsafe_allow_html=True)
                                    st.audio(preview_url)                            
                    except:
                        pass
                if select=='Quotes':
                    pass
                if select=='Images':
                    predicted_label = text+ predicted_label + "memes"
                    search_and_display_images(predicted_label)
            else:
                st.write("Please Enter Text")


