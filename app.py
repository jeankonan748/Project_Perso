import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pickle
from PIL import Image
import cv2
import time
import io

st.title('Détection de Maladies des Plantes')
st.write("Cette application identifie si une plante est malade ou saine à partir d'une image ou via votre caméra")

@st.cache_resource
def load_model_and_config():
    with open('model_config.pkl', 'rb') as f:
        config = pickle.load(f)
    
    with open(config['class_mapping_file'], 'rb') as f:
        class_mapping = pickle.load(f)
    
    model = load_model(config['model_file'])
    
    return model, config, class_mapping

#Déterminer la classe si une plante saine ou malade
def is_healthy_plant(class_name):
    healthy_keywords = ['healthy', 'sain', 'saine', 'healthy_leaf', 'feuille_saine']
    return any(keyword in class_name.lower() for keyword in healthy_keywords)

#Prétraitement d'image
def preprocess_image_for_prediction(image, config):
    if isinstance(image, Image.Image):
        image = np.array(image)
    
#Image en niveaux de gris
    if len(image.shape) == 2:
        image = np.stack((image,) * 3, axis=-1)
#Canal alpha (RGBA)
    elif image.shape[2] == 4:
#RVB
        image = image[:, :, :3]
    
#Redimensionner à la même taille
    image = tf.image.resize(image, (config['img_height'], config['img_width']))
    
    image = tf.cast(image, tf.float32) / 255.0
    
#Dimension pour le batch
    return tf.expand_dims(image, 0)


#Resultats
def display_prediction_results(prediction, class_mapping):
    class_index = np.argmax(prediction[0])
    confidence = float(prediction[0][class_index]) * 100
    predicted_class = class_mapping[class_index]
    
#Détermine plante saine ou malade
    is_healthy = is_healthy_plant(predicted_class)
    
#Le statut de santé
    if is_healthy:
        st.success("✅ Plante SAINE!")
    else:
        st.error("⚠️ Plante MALADE!")
    
    st.write(f"**Diagnostic détaillé:** {predicted_class}")
    st.write(f"**Niveau de confiance:** {confidence:.2f}%")
    
    st.subheader("Recommandations:")
    if is_healthy:
        st.write("""
        * Continuez vos bonnes pratiques d'entretien
        * Maintenez l'arrosage et l'exposition au soleil actuels
        * Surveillez régulièrement pour détecter tout changement
        """)
    else:
        st.write("""
        * Isolez la plante des autres pour éviter la propagation
        * Envisagez un traitement adapté à la maladie détectée
        * Ajustez l'arrosage et l'exposition au soleil si nécessaire
        """)

try:
    model, config, class_mapping = load_model_and_config()
    
#Onglets
    tab1, tab2 = st.tabs(["Télécharger une image", "Utiliser la caméra"])
    
    with tab1:
        uploaded_file = st.file_uploader("Choisissez une image de plante...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Image téléchargée", use_column_width=True)
            
            if st.button("Analyser l'image", key="analyze_upload"):
                with st.spinner('Analyse en cours...'):
                    processed_image = preprocess_image_for_prediction(image, config)
                    
                    prediction = model.predict(processed_image)
                    
#Afficher les résultats
                    display_prediction_results(prediction, class_mapping)
    
    with tab2:
        st.write("Utilisez votre caméra pour analyser une plante en temps réel")
        
        camera_image = st.camera_input("Prenez une photo de votre plante", key="camera", disabled=False)
        
        realt_results = st.empty()
        
        continu = st.checkbox("Activer l'analyse en continu")
        
        if continu and camera_image is not None:            
#Conteneur pour l'affichage en continu
            continuous_container = st.empty()
            
            while continu:
                if camera_image is not None:
                    image = Image.open(camera_image)
                    
                    processed_image = preprocess_image_for_prediction(image, config)
                    
                    prediction = model.predict(processed_image)
                    
                    class_index = np.argmax(prediction[0])
                    confidence = float(prediction[0][class_index]) * 100
                    predicted_class = class_mapping[class_index]
                    is_healthy = is_healthy_plant(predicted_class)
                    
                    with continuous_container.container():
                        if is_healthy:
                            st.success(f"✅ SAINE: {predicted_class} ({confidence:.1f}%)")
                        else:
                            st.error(f"⚠️ MALADE: {predicted_class} ({confidence:.1f}%)")   
                time.sleep(2)
                
                if not continu:
                    break
        
        elif camera_image is not None:
            with st.spinner('Analyse en cours...'):
                image = Image.open(camera_image)
                
                processed_image = preprocess_image_for_prediction(image, config)
                
                prediction = model.predict(processed_image)
                
#Afficher les résultats
                with realt_results.container():
                    display_prediction_results(prediction, class_mapping)

#Guide
    with st.expander("Comment utiliser cette application?"):
        st.write("""
        **Option 1: Téléchargement d'image**
        1. Sélectionnez l'onglet "Télécharger une image"
        2. Téléchargez une image claire d'une feuille de plante
        3. Cliquez sur "Analyser l'image" pour obtenir le diagnostic
        
        **Option 2: Caméra en temps réel**
        1. Sélectionnez l'onglet "Utiliser la caméra"
        2. Choisissez la résolution souhaitée
        3. Prenez une photo de la plante à analyser
        
        Pour de meilleurs résultats, assurez-vous que:
        * L'image montre clairement les feuilles ou les zones symptomatiques
        * L'éclairage est suffisant
        * L'arrière-plan n'est pas trop encombré
        """)

except Exception as e:
    st.error(f"Une erreur s'est produite: {e}")