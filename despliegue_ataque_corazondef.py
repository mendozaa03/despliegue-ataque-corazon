# # Despliegue: Consultorio del Dr. KNN (predicción de ataque al corazón)

import numpy as np
import pandas as pd

#cargamos el modelo
import pickle
filename = "modelo-class.pkl"
modelo, labelencoder, variables, min_max_scaler = pickle.load(open(filename, 'rb'))
#modelo

#cargar datos futuros
#data= pd.read_csv("ataque_corazon-datosFuturos.csv")
#data.head()

#Estilos de la página: consultorio médico (historia clínica, fórmula médica, corazón que late)
import streamlit as st

st.set_page_config(page_title="Consultorio del Dr. KNN", page_icon="🩺", layout="wide")

estilos = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,400;12..96,600;12..96,800&family=Courier+Prime:wght@400;700&family=Caveat:wght@500;700&display=swap');
:root { --tinta:#17212B; --fondo:#EEF2F3; --rojo:#D7263D; --vino:#8E1B2B; --verde:#0F766E;
--manila:#E9D6A4; --manila-oscuro:#CDB473; --suave:#566370; --linea:#C9D7E8; }
.stApp { background: var(--fondo); }
.stApp, .stApp p, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp input, .stApp button,
.stApp .stMarkdown div { font-family:'Bricolage Grotesque', system-ui, sans-serif; color: var(--tinta); }
.block-container { padding-top:2rem; max-width:1180px; }
.titulo { font-size:clamp(2.6rem,6vw,4.4rem); font-weight:800; letter-spacing:-0.04em; line-height:.95; margin:0 0 .6rem; }
.bajada { font-size:1.1rem; color:var(--suave); max-width:58ch; line-height:1.5; margin:0 0 1.8rem; }
.paso { font-size:1.55rem; font-weight:800; letter-spacing:-0.02em; margin:2.8rem 0 .3rem; }
.nota { color:var(--suave); max-width:64ch; line-height:1.5; margin:0 0 1.1rem; }

/* Monitor de electrocardiograma en movimiento */
.monitor { background:#0B1A1E; border-radius:18px; padding:14px 18px 12px; position:relative;
box-shadow: inset 0 0 0 2px #1E3238, 0 18px 40px -24px #17212B; }
.monitor.sano { --traza:#3BE38B; }
.monitor.riesgo { --traza:#FF5A6A; }
.stApp .stMarkdown div.monitor-arriba { display:flex; justify-content:space-between; align-items:center;
color:var(--traza); font-family:'Courier Prime', monospace !important; font-size:.95rem; margin-bottom:6px; }
.pulso { display:inline-block; animation: pulso var(--ritmo) ease-out infinite; }
@keyframes pulso { 0% { opacity:1; transform:scale(1.25); } 40% { opacity:.25; transform:scale(1); } 100% { opacity:.25; } }
.pantalla { position:relative; border-radius:8px; overflow:hidden;
background-image: linear-gradient(rgba(120,200,170,.10) 1px, transparent 1px), linear-gradient(90deg, rgba(120,200,170,.10) 1px, transparent 1px);
background-size: 24px 24px; }
.pantalla::after { content:""; position:absolute; inset:0; pointer-events:none;
background: linear-gradient(90deg, #0B1A1E 0%, rgba(11,26,30,0) 30%, rgba(11,26,30,0) 88%, #0B1A1E 100%); }
.pantalla svg { width:100%; height:230px; display:block; }
.traza { fill:none; stroke:var(--traza); stroke-width:2.6; stroke-linejoin:round; stroke-linecap:round;
vector-effect:non-scaling-stroke; filter: drop-shadow(0 0 4px var(--traza)); }
.desplaza { animation: desplaza var(--vuelta) linear infinite; }
@keyframes desplaza { from { transform:translateX(0); } to { transform:translateX(-600px); } }
@media (prefers-reduced-motion: reduce) { .desplaza, .pulso { animation:none; } }

/* Historia clínica (carpetas manila) */
.carpeta { background:var(--manila); border-radius:0 10px 10px 10px; padding:18px 20px 16px; position:relative;
box-shadow: 0 1px 0 var(--manila-oscuro), 0 10px 24px -18px #5b4a1a; margin-top:26px; }
.carpeta::before { content: attr(data-pestana); position:absolute; top:-24px; left:0; background:var(--manila);
padding:4px 14px 2px; border-radius:8px 8px 0 0; font-family:'Courier Prime', monospace; font-size:.8rem; color:#5b4a1a; }
.carpeta .renglon { font-family:'Courier Prime', monospace !important; font-size:1rem; line-height:1.75; color:#2b2410; }
.carpeta.grande .renglon { font-size:1.1rem; }
.sello { display:inline-block; margin-top:12px; transform:rotate(-6deg); border:3px solid; border-radius:6px;
padding:4px 10px; font-family:'Courier Prime', monospace !important; font-weight:700; font-size:1.05rem; letter-spacing:.06em; opacity:.85; }
.sello.ataque { color:var(--rojo); border-color:var(--rojo); }
.sello.sin { color:var(--verde); border-color:var(--verde); }
.carpeta.chica .sello { position:static; display:inline-block; margin-top:8px; font-size:.8rem; transform:rotate(-4deg); }
.carpetas { display:grid; grid-template-columns:repeat(auto-fit, minmax(160px, 1fr)); gap:14px; }
.carpeta.chica { padding:14px 14px 12px; }
.carpeta.chica .renglon { font-size:.88rem; line-height:1.55; }

/* Línea de vida */
.vida { display:flex; gap:2px; align-items:flex-end; height:70px; }
.ano { flex:1; height:46px; border-radius:2px; }
.ano.r { background:var(--rojo); }
.ano.s { background:#BFD8D4; }
.ano.hoy { height:70px; outline:3px solid var(--tinta); outline-offset:1px; }
.edades { display:flex; gap:2px; margin-top:6px; }
.edades span { flex:1; font-size:.75rem; color:var(--suave); white-space:nowrap; overflow:visible; }
.resumen-vida { font-size:1.15rem; margin:.9rem 0 0; max-width:62ch; }

/* Fórmula médica */
.formula { background:#FFFFFF; max-width:680px; border-radius:6px; padding:26px 30px 22px; position:relative;
box-shadow: 0 18px 40px -28px #17212B; }
.formula .cuerpo { background-image: repeating-linear-gradient(transparent 0 37px, var(--linea) 37px 38px); }
.formula .membrete { display:flex; justify-content:space-between; align-items:baseline; border-bottom:2px solid var(--tinta);
padding-bottom:10px; margin-bottom:12px; background:#FFFFFF; }
.formula .membrete b { font-size:1.25rem; letter-spacing:-0.01em; }
.formula .membrete span { color:var(--suave); font-size:.9rem; }
.rx { font-family:'Caveat', cursive !important; font-size:1.9rem; color:#1D3A8A !important; line-height:38px; height:38px; }
.mano { font-family:'Caveat', cursive !important; font-size:1.45rem; color:#1D3A8A !important; line-height:38px; margin:0; }
.firma { margin-top:20px; text-align:right; font-family:'Caveat', cursive !important; font-size:2rem; color:#1D3A8A !important; }
.firma small { display:block; font-family:'Bricolage Grotesque', sans-serif !important; font-size:.8rem; color:var(--suave); border-top:1px solid var(--suave);
width:180px; margin-left:auto; text-align:center; padding-top:2px; }
</style>
"""
st.markdown(estilos, unsafe_allow_html=True)

#interfaz grafica
#Se crea interfaz gráfica con streamlit para captura de los datos (triaje)
st.markdown('<h1 class="titulo">Consultorio del Dr. KNN</h1>', unsafe_allow_html=True)
st.markdown('<div class="bajada">El Dr. KNN no adivina: busca, entre los pacientes que ya conoce, '
            'al que más se parece a usted y le cuenta qué le pasó. Llene el triaje y empiece la consulta.</div>',
            unsafe_allow_html=True)

#Textos en español para mostrar (los valores internos quedan iguales a los de los datos)
si_no = {'No': 'No', 'Yes': 'Sí'}
nombres_fuma = {"'never smoked'": 'Nunca ha fumado', "'formerly smoked'": 'Fumaba antes',
                'smokes': 'Fuma', 'Unknown': 'No se sabe'}
#Las opciones de fumar salen de las columnas que guardó el modelo
opciones_fuma = [c.replace('smoking_status_', '') for c in variables if c.startswith('smoking_status_')]

#Límites de edad y glucosa según los datos con que se entrenó (min y max del normalizador)
edad_min, glucosa_min = min_max_scaler.data_min_
edad_max, glucosa_max = min_max_scaler.data_max_

col_triaje, col_consulta = st.columns([1, 1.6], gap="large")

with col_triaje:
    with st.container(border=True):
        st.markdown("**Triaje**")
        age = st.slider('Edad (años)', min_value=int(edad_min), max_value=int(edad_max), value=55, step=1)
        avg_glucose_level = st.slider('Glucosa promedio en sangre (mg/dL)', min_value=float(round(glucosa_min)),
                                      max_value=float(round(glucosa_max)), value=110.0, step=1.0, format="%.0f")
        hypertension = st.radio('¿Tiene hipertensión?', ['No', 'Yes'], format_func=lambda v: si_no[v], horizontal=True)
        heart_disease = st.radio('¿Tiene alguna enfermedad del corazón?', ['No', 'Yes'], format_func=lambda v: si_no[v], horizontal=True)
        ever_married = st.radio('¿Se ha casado alguna vez?', ['No', 'Yes'], index=1, format_func=lambda v: si_no[v], horizontal=True)
        smoking_status = st.selectbox('¿Fuma?', opciones_fuma, format_func=lambda v: nombres_fuma.get(v, v))

#Dataframe
datos = [[age, hypertension, heart_disease, ever_married, avg_glucose_level, smoking_status]]
columnas = ['age', 'hypertension', 'heart_disease', 'ever_married', 'avg_glucose_level', 'smoking_status']
data = pd.DataFrame(datos, columns=columnas)  #Dataframe con los mismos nombres de variables

#Se realiza la preparación de datos
data_preparada = data.copy()

#En despliegue drop_first= False
data_preparada = pd.get_dummies(data_preparada, columns=['hypertension', 'heart_disease', 'ever_married', 'smoking_status'],
                                drop_first=False, dtype=int)

#Se adicionan las columnas faltantes
data_preparada = data_preparada.reindex(columns=variables, fill_value=0).astype(float)

#Se normalizan las variables numéricas para predecir con Knn
#En los despliegues no se llama fit
variables_numericas = ['age', 'avg_glucose_level']
data_preparada[variables_numericas] = min_max_scaler.transform(data_preparada[variables_numericas])

# ## ***PREDICCIONES***

#Hacemos la predicción con el Knn
Y_pred = modelo.predict(data_preparada)
print(Y_pred)

#hacemos la prediccion (se devuelve de 0/1 a No/Yes con el labelencoder)
data["prediccion"] = labelencoder.inverse_transform(Y_pred)
riesgo = data["prediccion"].iloc[0] == "Yes"

#Buscamos los 5 pacientes más parecidos en los datos de entrenamiento (así decide el Knn)
distancias, indices = modelo.kneighbors(data_preparada, n_neighbors=5)

vecinos = pd.DataFrame(np.asarray(modelo._fit_X)[indices[0]], columns=variables)
vecinos[variables_numericas] = min_max_scaler.inverse_transform(vecinos[variables_numericas])  #volver a años y mg/dL
vecinos["resultado"] = labelencoder.inverse_transform(modelo._y[indices[0]])
vecinos["numero"] = indices[0]

#Función para predecir varios pacientes a la vez (misma preparación de arriba)
#Se usa para la línea de vida y para la fórmula médica
def predecir_varios(tabla):
    preparada = pd.get_dummies(tabla, columns=['hypertension', 'heart_disease', 'ever_married', 'smoking_status'],
                               drop_first=False, dtype=int)
    preparada = preparada.reindex(columns=variables, fill_value=0).astype(float)
    preparada[variables_numericas] = min_max_scaler.transform(preparada[variables_numericas])
    return labelencoder.inverse_transform(modelo.predict(preparada))

#Sonido del corazón (ilustrativo): latido regular si no hay riesgo, rápido e irregular si hay riesgo
def sonido_corazon(riesgo, frecuencia=8000, segundos=8):
    senal = np.zeros(frecuencia * segundos)
    azar = np.random.default_rng(7)

    def golpe(inicio, tono, duracion, volumen):
        t = np.arange(int(frecuencia * duracion)) / frecuencia
        forma = np.exp(-t * 30) * (1 - np.exp(-t * 300))
        sonido = volumen * forma * (np.sin(2 * np.pi * tono * t) + 0.5 * np.sin(2 * np.pi * 2 * tono * t))
        i = int(inicio * frecuencia)
        fin = min(i + len(sonido), len(senal))
        senal[i:fin] += sonido[:fin - i]

    tiempo = 0.2
    while tiempo < segundos - 0.6:
        golpe(tiempo, 110, 0.14, 1.0)            #primer ruido: "lub"
        if riesgo:
            golpe(tiempo + 0.22, 160, 0.10, 0.7)  #segundo ruido: "dub"
            golpe(tiempo + 0.36, 85, 0.10, 0.45)  #tercer ruido (galope)
            tiempo += azar.uniform(0.42, 0.78)    #latidos irregulares
        else:
            golpe(tiempo + 0.30, 160, 0.10, 0.7)
            tiempo += 0.85                        #latidos regulares
    return senal / np.abs(senal).max() * 0.9

#Función que dibuja el trazo del electrocardiograma (dos vueltas iguales para que el movimiento no se corte)
def trazo_monitor(riesgo, ancho=600, base=125):
    azar = np.random.default_rng(3)
    latidos = []
    x = 20
    while x < ancho - 110:
        latidos.append(x)
        x += azar.uniform(70, 140) if riesgo else 150
    puntos = []
    for vuelta in (0, ancho):
        puntos.append((vuelta, base))
        for i, x0 in enumerate(latidos):
            x = x0 + vuelta
            r_alto = 92 + (azar.uniform(-14, 14) if riesgo else 0)
            st_sube, t_alto = (14, 26) if riesgo else (0, 15)
            puntos.append((x, base))
            for k in range(1, 9):  #onda P
                puntos.append((x + k * 2.5, base - 7 * np.sin(np.pi * k / 8)))
            x += 28
            puntos += [(x, base), (x + 4, base + 8), (x + 9, base - r_alto), (x + 14, base + 26), (x + 19, base - st_sube)]  #QRS
            x += 31
            for k in range(1, 13):  #onda T
                puntos.append((x + k * 3, base - st_sube * (1 - k / 12) - t_alto * np.sin(np.pi * k / 12)))
            puntos.append((x + 40, base))
        puntos.append((vuelta + ancho, base))
    return "M " + " L ".join(f"{px:.1f} {py:.1f}" for px, py in puntos)


#Monitor y diagnóstico del Dr. KNN
with col_consulta:
    clase = "riesgo" if riesgo else "sano"
    vuelta = "2.6s" if riesgo else "3.6s"    #velocidad del trazo
    ritmo = "0.55s" if riesgo else "0.9s"    #parpadeo del corazón
    estado = "Ritmo irregular" if riesgo else "Ritmo regular"
    monitor = (
        f'<div class="monitor {clase}" style="--vuelta:{vuelta}; --ritmo:{ritmo}">'
        f'<div class="monitor-arriba"><span>Derivación II</span><span><span class="pulso">♥</span> {estado}</span></div>'
        '<div class="pantalla"><svg viewBox="0 0 600 200" preserveAspectRatio="none" role="img" '
        f'aria-label="Electrocardiograma en movimiento, {estado.lower()}">'
        f'<g class="desplaza"><path class="traza" d="{trazo_monitor(riesgo)}"/></g>'
        '</svg></div></div>'
    )
    st.markdown(monitor, unsafe_allow_html=True)

    st.caption("Monitor y sonido ilustrativos: cambian según la lectura del modelo. El latido suena solo al mover cualquier dato del triaje.")
    st.audio(sonido_corazon(riesgo), sample_rate=8000, autoplay=True, loop=True)  #suena solo y se repite

    con_ataque = int((vecinos["resultado"] == "Yes").sum())
    gemelo = vecinos.iloc[0]
    with st.chat_message("assistant", avatar="🩺"):
        if riesgo:
            st.markdown(f"**Mi lectura: riesgo de ataque al corazón.** Revisé mis historias clínicas y el paciente "
                        f"que más se parece a usted, de {round(gemelo['age'])} años, **sí tuvo** un ataque. "
                        f"De sus 5 pacientes más parecidos, {con_ataque} lo tuvieron.")
        else:
            st.markdown(f"**Mi lectura: sin señales de riesgo.** Revisé mis historias clínicas y el paciente "
                        f"que más se parece a usted, de {round(gemelo['age'])} años, **no tuvo** un ataque. "
                        f"De sus 5 pacientes más parecidos, {con_ataque} lo tuvieron.")

#Historia clínica de los pacientes más parecidos
columnas_fuma = [c for c in variables if c.startswith('smoking_status_')]

def historia(v, tamano):
    fuma = v[columnas_fuma].astype(float).idxmax().replace('smoking_status_', '')
    sello = ('<div class="sello ataque">TUVO ATAQUE</div>' if v["resultado"] == "Yes"
             else '<div class="sello sin">SIN ATAQUE</div>')
    renglones = (f'<div class="renglon">Edad: {round(v["age"])} años</div>'
                 f'<div class="renglon">Glucosa: {round(v["avg_glucose_level"])} mg/dL</div>'
                 f'<div class="renglon">Hipertensión: {"Sí" if v["hypertension_Yes"] > 0.5 else "No"}</div>'
                 f'<div class="renglon">Enf. corazón: {"Sí" if v["heart_disease_Yes"] > 0.5 else "No"}</div>'
                 f'<div class="renglon">Casado/a: {"Sí" if v["ever_married_Yes"] > 0.5 else "No"}</div>'
                 f'<div class="renglon">Fuma: {nombres_fuma.get(fuma, fuma)}</div>')
    pestana = f'Historia clínica N.º {int(v["numero"])}' if tamano == "grande" else f'N.º {int(v["numero"])}'
    return (f'<div class="carpeta {tamano}" data-pestana="{pestana}">'
            + renglones + sello + '</div>')

st.markdown('<h2 class="paso">La historia clínica más parecida</h2>', unsafe_allow_html=True)
st.markdown('<div class="nota">Este es el paciente que el Dr. KNN encontró más parecido a usted. '
            'Su resultado es exactamente la lectura del modelo. Algunas historias pueden ser casos creados '
            'durante el balanceo de los datos.</div>', unsafe_allow_html=True)

col_gemelo, col_otros = st.columns([1, 1.5], gap="large")
with col_gemelo:
    st.markdown(historia(vecinos.iloc[0], "grande"), unsafe_allow_html=True)
with col_otros:
    st.markdown('<div class="nota" style="margin-top:6px">Los siguientes cuatro más parecidos:</div>', unsafe_allow_html=True)
    st.markdown('<div class="carpetas">' + "".join(historia(vecinos.iloc[i], "chica") for i in range(1, 5)) + '</div>',
                unsafe_allow_html=True)

#Línea de vida: la lectura del modelo para el mismo paciente en todas las edades
edades = list(range(int(edad_min), int(edad_max) + 1))
tabla_vida = pd.DataFrame([[e, hypertension, heart_disease, ever_married, avg_glucose_level, smoking_status] for e in edades],
                          columns=columnas)
vida = predecir_varios(tabla_vida)

barras = ""
etiquetas = ""
for e, r in zip(edades, vida):
    clase = "r" if r == "Yes" else "s"
    hoy = " hoy" if e == age else ""
    barras += f'<span class="ano {clase}{hoy}" title="{e} años"></span>'
    etiquetas += f'<span>{e if e % 10 == 0 else ""}</span>'

edades_riesgo = [e for e, r in zip(edades, vida) if r == "Yes"]
if not edades_riesgo:
    resumen = "Con este mismo triaje, el modelo no ve riesgo en ninguna edad."
elif len(edades_riesgo) == len(edades):
    resumen = "Con este mismo triaje, el modelo ve riesgo en todas las edades."
else:
    resumen = (f"Con este mismo triaje, el modelo ve riesgo por primera vez a los {edades_riesgo[0]} años, "
               f"y en {len(edades_riesgo)} de las {len(edades)} edades posibles.")

st.markdown('<h2 class="paso">Su línea de vida</h2>', unsafe_allow_html=True)
st.markdown('<div class="nota">Cada barra es una edad, con el resto del triaje igual. En rojo, las edades en que el '
            'modelo leería riesgo. La barra alta y marcada es su edad actual.</div>', unsafe_allow_html=True)
st.markdown(f'<div class="vida">{barras}</div><div class="edades">{etiquetas}</div>'
            f'<div class="resumen-vida">{resumen}</div>', unsafe_allow_html=True)

#Fórmula médica: qué cambio, por sí solo, haría cambiar la lectura del modelo
paciente = dict(zip(columnas, datos[0]))
cambios = []
if riesgo:
    if smoking_status != "'never smoked'" and "'never smoked'" in opciones_fuma:
        cambios.append(("Nunca haber fumado", {**paciente, "smoking_status": "'never smoked'"}))
    if avg_glucose_level > 140:
        cambios.append(("Bajar la glucosa a 100 mg/dL", {**paciente, "avg_glucose_level": 100.0}))
    if hypertension == "Yes":
        cambios.append(("Controlar la hipertensión", {**paciente, "hypertension": "No"}))
    if heart_disease == "Yes":
        cambios.append(("Tratar la enfermedad del corazón", {**paciente, "heart_disease": "No"}))
else:
    if smoking_status != "smokes" and "smokes" in opciones_fuma:
        cambios.append(("Empezar a fumar", {**paciente, "smoking_status": "smokes"}))
    if avg_glucose_level < 180:
        cambios.append(("Que la glucosa suba a 200 mg/dL", {**paciente, "avg_glucose_level": 200.0}))
    if hypertension == "No":
        cambios.append(("Tener hipertensión", {**paciente, "hypertension": "Yes"}))
    if age + 10 <= edad_max:
        cambios.append(("Tener 10 años más", {**paciente, "age": age + 10}))

if cambios:
    resultados = predecir_varios(pd.DataFrame([p for _, p in cambios], columns=columnas))
    efectivos = [texto for (texto, _), r in zip(cambios, resultados) if (r == "Yes") != riesgo]
else:
    efectivos = []

if riesgo:
    encabezado = "Cambios que, por sí solos, pasarían la lectura a sin riesgo:"
    vacio = "Ningún cambio por sí solo baja la lectura. Se recomienda valoración médica."
else:
    encabezado = "Señales de alerta: cambios que, por sí solos, pasarían la lectura a riesgo:"
    vacio = "Ningún cambio por sí solo sube la lectura. Mantener los hábitos actuales."

items = "".join(f'<div class="mano">— {t}</div>' for t in efectivos) if efectivos else f'<div class="mano">{vacio}</div>'
st.markdown('<h2 class="paso">Fórmula médica</h2>', unsafe_allow_html=True)
st.markdown('<div class="formula">'
            '<div class="membrete"><b>Consultorio del Dr. KNN</b><span>Fórmula ilustrativa del modelo</span></div>'
            f'<div class="cuerpo"><div class="rx">Rx.</div><div class="mano">{encabezado}</div>{items}</div>'
            '<div class="firma">Dr. KNN<small>Firma del modelo</small></div>'
            '</div>', unsafe_allow_html=True)

#predicciones finales
print(data)

# Recordar medida de desempeño del modelo
st.warning("El modelo tiene una exactitud del 88% y detecta 72 de cada 100 casos reales de ataque al corazón (recall). "
           "Es un ejercicio académico, no un diagnóstico médico.")
