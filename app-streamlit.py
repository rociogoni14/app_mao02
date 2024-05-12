import streamlit as st
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import io
import numpy as np

st.set_page_config(layout="wide")

# Inicializo las variables de sesión que voy a utilizar a lo largo del script

if "canvas_result" not in st.session_state:
    st.session_state["canvas_result"]= None

if "respuesta" not in st.session_state:
    st.session_state["respuesta"]= None

if "respuesta2" not in st.session_state:
    st.session_state["respuesta2"]= None



#Defino la función execute que lo que hace es recibir el resultado del dibujo del canvas. Este dibujo se convierte a imagen en formato PNG
#La imagen se configura para que esté en escala de grises de 28×28 píxeles que es el formato que tiene que tener para que pueda operar con ella
#el modelo preentrenado. Se llama al modelo, a este se le pasa la imagen configurada y predice las probabilidades que representan a cada dígito.
#Se coge aquel dígito correspondiente a la mayor probabilidad que en este caso se guarda en la variable de sesión respuesta



def execute():
    
    image = Image.fromarray(st.session_state.canvas_result.image_data.astype('uint8'))
    with io.BytesIO() as output:
        image.save(output, format="PNG")
        png_data = output.getvalue()
    model = load_model('modelo_digitos_mao02.keras')
    
    # Carga la imagen desde los bytes
    image = Image.open(io.BytesIO(png_data))

    # Convierte la imagen a escala de grises
    image_gray = image.convert('L')

    # Redimensiona la imagen a las dimensiones requeridas por el modelo
    # (en est caso, 28x28 para MNIST)
    image_resized = image_gray.resize((28, 28))

    # Convierte la imagen a un array numpy y normaliza los valores de píxeles
    image_array = np.array(image_resized) / 255.0

    # Agrega una dimensión adicional para representar el batch (el modelo espera un batch de imágenes)
    image_array = np.expand_dims(image_array, axis=0)
    prediccion = model.predict(image_array)
    st.session_state.respuesta = np.argmax(prediccion, axis=1)
    return st.session_state.respuesta[0]


#La función execute2 hace lo mismo que execute pero lo que le entra por parámetro es la imagen 
#en formato png, jpg o jpeg subida por el usuario 

def execute2(imagen):

    model = load_model('modelo_digitos_mao02.keras')
    # Cargar la imagen en formato PIL
    imagen_ = Image.open(imagen)

    # Transformar la imagen a un array numpy y realizar las transformaciones necesarias
    imagen_array = np.array(imagen_) / 255.0  # Normalizar los valores de píxeles

    # Agregar una dimensión adicional para representar el batch (el modelo espera un batch de imágenes)
    imagen_array = np.expand_dims(imagen_array, axis=0)

    # Realizar la predicción con el modelo
    prediccion = model.predict(imagen_array)

    # Obtener el índice de la clase con la mayor probabilidad
    st.session_state.respuesta2 = np.argmax(prediccion, axis=1)
    return st.session_state.respuesta2[0]




# Título con Markdown
st.markdown("<h1 style='text-align: center;'>🖋️ Aplicación para Reconocimiento de Dígitos Manuscritos 🖋️</h1>", unsafe_allow_html=True)


# Añado un espacio en blanco entre el título y el resto de cosas para que no quede todo junto
st.write("")

# Creo dos columnas para que haya dos formas distintas de introducir los dígitos manuscritos
pag1, pag2  = st.tabs(["Dibujar","Subir foto"])

with pag1:

    st.markdown("""<div style='display: flex; justify-content: center;'>
                    <div>
                        <h2 style='color: black;'>🎨 Dibuja aquí el dígito que quieras reconocer 🎨</h2></div></div>""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col2:
        st.write("")
        # Configuración del lienzo de dibujo
        st.session_state.canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Color de relleno
            stroke_width=10,  # Ancho del trazo
            stroke_color="#ffffff",  # Color del trazo
            background_color="#000000",  # Color de fondo
            update_streamlit=True,  # Actualizar Streamlit cuando se dibuje
            height=200,  # Altura del lienzo
            width=200,  # Anchura del lienzo
            drawing_mode="freedraw",  # Modo de dibujo ("freedraw" o "transform")
            key="canvas"
        )



        
        #Creo un botón para que se confirme que se han terminado los cambios en el dibujo y cuando se hace click sobre él, se ejecuta la función execute
        boton= st.button("Confirmar dibujo", on_click=execute)

with pag2:
    
    st.markdown("<h2 style='text-align:center; color: black;'>📷 ¡Sube la foto que quieras reconocer! 📸</h2>", unsafe_allow_html=True)


    #Creo un widget que permite subir ficheros del formato establecido, en este caso, png, jpg y jpeg
    imagen_png = st.file_uploader("Selecciona una imagen en formato PNG o JPG", type=["png",'jpg','jpeg'])
    col21,col22,col23 = st.columns(3)
    # Muestro la imagen cargada
    if imagen_png is not None:
        with col22:
            st.image(imagen_png, caption='Imagen cargada', width = 200)
            execute2(imagen_png)

if st.session_state.respuesta is not None:
    with pag1:
        with col2:

            if st.session_state.respuesta == 0:
            # Cargo la imagen
                imagen = Image.open("Foto 0.PNG")

                # Muestro la imagen en Streamlit
                st.image(imagen, caption='El número reconocido es un 0',  width=200)

            elif st.session_state.respuesta==1:
                # Cargo la imagen
                imagen = Image.open("Foto 1.PNG")

                # Muestro la imagen en Streamli
                st.image(imagen, caption='El número reconocido es un 1', width=200)
    
            elif st.session_state.respuesta==2:
                # Cargar la imagen desde el disco
                imagen = Image.open("Foto 2.PNG")

                # Muestro la imagen en Streamli
                st.image(imagen, caption='El número reconocido es un 2', width=200)

            elif st.session_state.respuesta==3:
                # Cargar la imagen desde el disco
                imagen = Image.open("Foto 3.PNG")

                # Muestro la imagen en Streamli
                st.image(imagen, caption='El número reconocido es un 3',width=200)

            elif st.session_state.respuesta==4:
                # Cargar la imagen desde el disco
                imagen = Image.open("Foto 4.PNG")

                # Muestro la imagen en Streamli
                st.image(imagen, caption='El número reconocido es un 4', width=200)

            elif st.session_state.respuesta==5:
                # Cargar la imagen desde el disco
                imagen = Image.open("Foto 5.PNG")

                # Muestro la imagen en Streamli
                st.image(imagen, caption='El número reconocido es un 5', width=200)

            elif st.session_state.respuesta==6:
                # Cargar la imagen desde el disco
                imagen = Image.open("Foto 6.PNG")

                # Muestro la imagen en Streamli
                st.image(imagen, caption='El número reconocido es un 6', width=200)

            elif st.session_state.respuesta==7:
                # Cargar la imagen desde el disco
                imagen = Image.open("Foto 7.PNG")

                # Muestro la imagen en Streamli
                st.image(imagen, caption='El número reconocido es un 7', width=200)

            elif st.session_state.respuesta==8:
                # Cargar la imagen desde el disco
                imagen = Image.open("Foto 8.PNG")

                # Muestro la imagen en Streamli
                st.image(imagen, caption='El número reconocido es un 8', width=200)

            else:
                # Cargar la imagen desde el disco
                imagen = Image.open("Foto 9.PNG")

                # Muestro la imagen en Streamli
                st.image(imagen, caption='El número reconocido es un 9',width=200)


if st.session_state.respuesta2 is not None:
    with pag2:
        with col22:
            if st.session_state.respuesta2 == 0:
                # Cargar la imagen desde el disco
                imagen = Image.open("Foto 0.PNG")

                # Muestro la imagen en Streamli
                st.image(imagen, caption='El número reconocido es un 0',  width=200)

            elif st.session_state.respuesta2==1:
                # Cargar la imagen desde el disco
                imagen = Image.open("Foto 1.PNG")

                # Muestro la imagen en Streamli
                st.image(imagen, caption='El número reconocido es un 1', width=200)
    
            elif st.session_state.respuesta2==2:
                # Cargar la imagen desde el disco
                imagen = Image.open("Foto 2.PNG")

                # Muestro la imagen en Streamli
                st.image(imagen, caption='El número reconocido es un 2', width=200)

            elif st.session_state.respuesta2==3:
                # Cargar la imagen desde el disco
                imagen = Image.open("Foto 3.PNG")

                # Muestro la imagen en Streamli
                st.image(imagen, caption='El número reconocido es un 3',width=200)

            elif st.session_state.respuesta2==4:
                # Cargar la imagen desde el disco
                imagen = Image.open("Foto 4.PNG")

                # Muestro la imagen en Streamli
                st.image(imagen, caption='El número reconocido es un 4', width=200)

            elif st.session_state.respuesta2==5:
                # Cargar la imagen desde el disco
                imagen = Image.open("Foto 5.PNG")

                # Muestro la imagen en Streamli
                st.image(imagen, caption='El número reconocido es un 5', width=200)

            elif st.session_state.respuesta2==6:
                # Cargar la imagen desde el disco
                imagen = Image.open("Foto 6.PNG")

                # Muestro la imagen en Streamli
                st.image(imagen, caption='El número reconocido es un 6', width=200)

            elif st.session_state.respuesta2==7:
                # Cargar la imagen desde el disco
                imagen = Image.open("Foto 7.PNG")

                # Muestro la imagen en Streamli
                st.image(imagen, caption='El número reconocido es un 7', width=200)

            elif st.session_state.respuesta2==8:
                # Cargar la imagen desde el disco
                imagen = Image.open("Foto 8.PNG")

                # Muestro la imagen en Streamli
                st.image(imagen, caption='El número reconocido es un 8', width=200)

            else:
                # Cargar la imagen desde el disco
                imagen = Image.open("Foto 9.PNG")

                # Muestro la imagen en Streamli
                st.image(imagen, caption='El número reconocido es un 9',width=200)

