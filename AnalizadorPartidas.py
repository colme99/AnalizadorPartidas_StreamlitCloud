


import streamlit as st
import pandas as pd
from graphviz import render

from PreprocesamientoLimpiezaDatos import PreprocesamientoLimpiezaDatos
from WebScrappingContinentes import WebScrappingContinentes
from AnalisisExploratorio import AnalisisExploratorio
from Clustering import Clustering
from Regresion import Regresion
from ArbolDecision import ArbolDecision
from RedNeuronal import RedNeuronal
from ComparadorModelos import ComparadorModelos



explicacion_columnas = [
                        'nombre de la liga a la que pertenece el equipo',
                        'duración de la partida (segundos)',
                        'nombre del equipo',
                        'resultado de la partida (1: ganar, 0: perder)',
                        'nº de muertes del equipo',
                        'nº de asistencias del equipo',
                        'nº de asesinatos a campeones enemigos del equipo',
                        '1 si el equipo ha conseguido el primer asesinato de la partida, 0 en caso contrario',
                        'nº de dragones asesinados por el equipo',
                        'nº de dragones infernales asesinados por el equipo',
                        'nº de dragones de montaña asesinados por el equipo',
                        'nº de dragones de nube asesinados por el equipo',
                        'nº de dragones de océano asesinados por el equipo',
                        'nº de dragones ancianos asesinados por el equipo',
                        '1 si se el equipo ha conseguido el primer dragón de la partida, 0 en caso contrario',
                        'nº de heraldos asesinados por el equipo',
                        '1 si se el equipo ha conseguido el primer heraldo de la partida, 0 en caso contrario',
                        'nº de barones Nashor asesinados por el equipo',
                        '1 si se el equipo ha conseguido el primer barón Nashor de la partida, 0 en caso contrario',
                        'nº de torretas destruidas por el equipo',
                        '1 si el equipo ha sido el primero en destruir una torreta, 0 en caso contrario',
                        '1 si el equipo ha sido el primero en destruir una torreta del carril central, 0 en caso contrario',
                        'nº de inhibidores destruidos por el equipo',
                        'cantidad total de daño hecho a campeones enemigos',
                        'nº de guardianes de visión colocados por el equipo',
                        'nº de guardianes de visión destruidos al equipo contrario',
                        'nº de súbditos asesinados por el equipo',
                        'nº de monstruos de la jungla asesinados por el equipo',
                        'nº de monstruos de la jungla del lado del mapa del equipo contrario asesinados por el equipo',
                        'cantidad de asesinatos a campeones enemigos en los 10 primeros minutos de partida',
                        'cantidad de oro en los 10 primeros minutos de partida'
                        ]
continentes_nombres = ['América del Norte', 'América del Sur', 'Europa', 'Asia', 'Oceania']
nombre_atributo_continente = 'continent'
nombre_atributo_lado_mapa = 'mapside'
enlace_web_lol_fandom_wiki = 'https://lol.fandom.com/wiki/'
enlace_web_wikipedia_torneos_lol = 'https://en.wikipedia.org/wiki/List_of_League_of_Legends_leagues_and_tournaments'
texto_seleccionar_dataset = 'Suba el conjunto de datos'
texto_indicar_caracteriscas_dataset = 'Indique el nombre de los atributos el conjunto de datos'
texto_seleccionar_seciones = 'Seleccione las secciones que quiera visualizar'
extension_fichero_conjunto_datos = {'csv'}
extension_fichero_dot = 'dot'
extension_fichero_png = 'png'
nombre_archivo_grafo_estrategia = 'grafo_estrategia_info'
etiquetas_equipo = ['t1_', 't2_']
nombre_fichero_estructura_grafo_estrategia = 'estructura_grafo_estrategia.txt'
patron_mejor_predictor_resultado = '***'
patron_mejor_predictor_predictor_resultado = '___'

# Nombres de las secciones
nombre_seccion_atributos_utilizados = 'Atributos utilizados'
nombre_seccion_procesado_dataset = 'Procesado del conjunto de datos'
nombre_seccion_analisis_exploratorio_datos = 'Análisis exploratorio de datos'
nombre_seccion_analisis_continentes = 'Análisis exploratorio por continentes'
nombre_seccion_analisis_lado_mapa = 'Análisis exploratorio según el lado del mapa'
nombre_seccion_clustering = 'Clustering'
nombre_seccion_predecir_resultado_partida = 'Predecir el resultado de la partida'
nombre_seccion_predecir_mejor_predictor_resultado_partida = 'Predecir el atributo más importante para determinar el resultado'



def mostrarTituloJustificado(importancia_titulo, texto):
    st.markdown('<div align="justify"> <h' + str(importancia_titulo) + '>' + texto + '</h + ' + str(importancia_titulo) + '> </div>', unsafe_allow_html = True)


def mostrarTextoJustificado(texto):
    st.markdown('<div align="justify"> <p>' + texto + '</p> </div>', unsafe_allow_html = True)


def mostrarTitulo(importancia_titulo, texto):
    st.markdown('<center> <h' + str(importancia_titulo) + '>' + texto + '</h + ' + str(importancia_titulo) + '> <center>', unsafe_allow_html = True)


@st.cache(show_spinner = False)
def columnasConEtiquetaDeEquipo(columnas, etiquetas_equipo):
    columnas_equipo = []

    # Columnas generales
    columnas_equipo.append(columnas[0])
    columnas_equipo.append(columnas[1])

    # Columnas dependientes del equipo
    for i in range(len(etiquetas_equipo)):
        for j in range(2, len(columnas)):
            columnas_equipo.append(etiquetas_equipo[i] + columnas[j])
    return columnas_equipo


def mostrarListaTexto(lista):
    lista_elementos = '<ul>'
    for elemento in lista:
        lista_elementos += '<li>' + elemento + '</li>'
    lista_elementos += '</ul>'
    
    st.markdown(lista_elementos, unsafe_allow_html = True)


def aumentarAnchuraBarraLateral():
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 600px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 600px;
            margin-left: -600px;
        }
        </style>
        """,
        unsafe_allow_html = True,
    )


@st.cache(show_spinner = False)
def explicacionesAtributos(columnas):
    explicacion_columnas_mostrar = []
    for columna, explicacion in zip(columnas, explicacion_columnas):
        explicacion_columnas_mostrar.append('<b>' + columna + '</b>: ' + explicacion)
    return explicacion_columnas_mostrar


@st.cache(show_spinner = False)
def explicacionesAtributosPrimeraLetraMayuscula():
    explicacion_columnas_mayus = []
    for columna in explicacion_columnas:
        explicacion_columnas_mayus.append(columna.capitalize())
    return explicacion_columnas_mayus


@st.cache(show_spinner = False)
def leerDatos(fichero_subido):
    return pd.read_csv(fichero_subido, sep = None)


@st.cache(allow_output_mutation = True, show_spinner = False)
def hacerPreprocesemientoAnalisisExploratorio(datos, columnas_equipo, columnas, atributos_categoricos):

    preprocesamiento_limpieza_datos = PreprocesamientoLimpiezaDatos(atributos_categoricos)

    # Acotar el dataset a las columnas seleccionadas
    datos_columnas_seleccionadas = datos[columnas_equipo]

    # Separar la partida en dos con la información de la partida de cada equipo
    datos_por_equipo = preprocesamiento_limpieza_datos.separarPartidaPorEquipos(datos_columnas_seleccionadas, columnas, nombre_atributo_lado_mapa)
    num_total_observaciones = datos_por_equipo.shape[0]

    analisis_exploratorio = AnalisisExploratorio(nombre_atributo_oro_10min = columnas[30], nombre_atributo_asesinatos_10min = columnas[29], nombre_atributo_dragones = columnas[8], nombre_atributo_barones = columnas[17], nombre_atributo_heraldos = columnas[15], nombre_atributo_lado_mapa = nombre_atributo_lado_mapa, nombre_atributo_resultado_partida = columnas[3], nombre_atributo_duracion_partida = columnas[1], nombre_atributo_continente = nombre_atributo_continente, nombre_atributo_equipo = columnas[2])

    # Distribución de los valores originales de los atributos
    figuras_distribucion_atributos_originales = analisis_exploratorio.figuraDistribucionAtributos(datos_por_equipo)

    # Crear un nuevo atributo mediante web scrapping: el continente
    datos_con_continentes = hacerWebScrapping(preprocesamiento_limpieza_datos, datos_por_equipo, nombre_atributo_equipo = columnas[2], nombre_atributo_liga = columnas[0])

    # La información geográfica y la duración de la partida no se incluye en la predicción de la victoria (no es una estrategia a seguir o es muy poco precisa)
    # [nombre del equipo, nombre de la liga, duración de la partida]
    columnas_eliminar_predecir_victoria = [columnas[2], columnas[0], columnas[1]]
    datos_por_equipo.drop(columns = columnas_eliminar_predecir_victoria, axis = 1, inplace = True)
    predictores_entrenar_predecir_victoria, predictores_validacion_predecir_victoria, predictores_prueba_predecir_victoria, y_entrenar_predecir_victoria, y_validacion_predecir_victoria, y_prueba_predecir_victoria, predictores_entrenamiento_sin_escalar_predecir_victoria, y_entrenar_sin_escalar_predecir_victoria,  predictores_entrenamiento_validacion_predecir_victoria, y_entrenamiento_validacion_predecir_victoria, indices_entrenamiento_validacion_predecir_victoria, num_observaciones_eliminadas_predecir_victoria = preprocesamiento_limpieza_datos.dividirDatosEnEntrenamientoTest(datos_por_equipo, 'result')

    figura_comparacion_observaciones_eliminadas = preprocesamiento_limpieza_datos.figuraComparacionObservaciones(num_total_observaciones, num_observaciones_eliminadas_predecir_victoria, datos_con_continentes.shape[0])

    # Análisis del porcentaje de victorias en función del lado del mapa
    figuras_analisis_lado_mapa = analisis_exploratorio.figurasAnalisisLadoMapa(datos_con_continentes)

    # Análisis exploratorio por continentes
    figuras_analisis_exploratorio_continentes = analisis_exploratorio.figurasAnalisisExploratorioContinentes(datos_con_continentes)

    # Codificar el atributo referente al continente (label encoding)
    datos_con_continentes_numerico = mapearContinentesNumerico(datos_con_continentes)

    # Escalar los datos con el atributo continente entre 0 y 1
    datos_con_continentes_numerico_escalado = preprocesamiento_limpieza_datos.escalarDatos(datos_con_continentes_numerico)

    return figuras_distribucion_atributos_originales, figuras_analisis_lado_mapa, figuras_analisis_exploratorio_continentes, figura_comparacion_observaciones_eliminadas, datos_con_continentes_numerico_escalado, datos_por_equipo, predictores_entrenar_predecir_victoria, predictores_validacion_predecir_victoria, predictores_prueba_predecir_victoria, y_entrenar_predecir_victoria, y_validacion_predecir_victoria, y_prueba_predecir_victoria, predictores_entrenamiento_validacion_predecir_victoria, y_entrenamiento_validacion_predecir_victoria, indices_entrenamiento_validacion_predecir_victoria, predictores_entrenamiento_sin_escalar_predecir_victoria, y_entrenar_sin_escalar_predecir_victoria, preprocesamiento_limpieza_datos


def hacerWebScrapping(preprocesamiento_limpieza_datos, datos_por_equipo, nombre_atributo_equipo, nombre_atributo_liga):
    datos_por_equipo = preprocesamiento_limpieza_datos.eliminarValoresPerdidos(datos_por_equipo)
    web_scrapping_continentes = WebScrappingContinentes(datos_por_equipo)
    datos_con_continentes, num_observaciones_no_encontradas_web_scrapping = web_scrapping_continentes.crearAtributoContinentes()
    # Se elimina la información geográfica porque se ha unificado en el atributo continente
    return datos_con_continentes.drop(columns = [nombre_atributo_equipo, nombre_atributo_liga], axis = 1).reset_index(drop = True)


def mapearContinentesNumerico(datos):
    datos[nombre_atributo_continente] = pd.to_numeric(datos[nombre_atributo_continente].map({continentes_nombres[0]: 0, continentes_nombres[1]: 1, continentes_nombres[2]: 2,
                                                                    continentes_nombres[3]: 3, continentes_nombres[4]: 4}))
    return datos


def mostrarVariasFiguras(figuras):
    for figura in figuras:
        st.plotly_chart(figura)


def mostrarFigurasClustering(valor_k, figuras_clustering, figura_metodo_codo, precision_clasificacion_clusters, nombre_metrica):
    
    mostrarTituloJustificado(3, 'Determinar automáticamente el número de clusters utilizando como métrica ' + nombre_metrica + ' y como distancia la distancia de Gower')
    mostrarTitulo(4, 'Número de clusters elegido de forma automática: ' + str(valor_k))
    st.pyplot(figura_metodo_codo)

    mostrarTitulo(3, 'KMedoids con ' + str(valor_k) + ' clusters')

    mostrarTitulo(4, 'Cantidad de partidas por cluster')
    st.plotly_chart(figuras_clustering[0])   

    mostrarTitulo(4, 'Porcentaje de victorias por cluster')
    st.plotly_chart(figuras_clustering[1])   

    mostrarTituloJustificado(4, 'Para estimar los atributos que más difieren entre clusters se utiliza un modelo de bosque aleatorio para predecir el cluster al que pertenece cada partida. Para determinar la importancia de los atributos se utiliza la importancia por permutación. Precisión del modelo: ' + str(round(precision_clasificacion_clusters, 3)))
    mostrarTitulo(4, 'Atributos en los que más difieren los clusters')
    st.plotly_chart(figuras_clustering[2]) 

    mostrarTitulo(4, 'Comparación visual de los cinco atributos en los que más difieren los clusters')        
    mostrarVariasFiguras(figuras_clustering[3]) 

    mostrarTitulo(4, 'Distribución de valores de los cinco atributos en los que más difieren los clusters')
    mostrarVariasFiguras(figuras_clustering[4]) 

    mostrarTitulo(4, 'Gráfica que combina los dos atributos en los que más difieren los clusters')
    st.plotly_chart(figuras_clustering[5])


def mostrarFigurasAnalisisExploratorio(figuras_analisis_exploratorio, titulos_figuras):
    for i in range(len(figuras_analisis_exploratorio)):
        mostrarTitulo(3, titulos_figuras[i])
        st.plotly_chart(figuras_analisis_exploratorio[i])
    

def tipoRelacion(datos, atributo_1, atributo_2):
    correlacion = datos[atributo_1].corr(datos[atributo_2])
    correlacion_etiqueta = '-'
    if correlacion > 0:
        correlacion_etiqueta = '+'
    return correlacion_etiqueta


@st.cache(allow_output_mutation = True, show_spinner = False)
def hacerClustering(datos, nombre_atributo_resultado, atributos_categoricos):
    clustering = Clustering(datos, nombre_atributo_resultado, atributos_categoricos)
    k_seleccionado_slhouette, k_seleccionado_distorsion, figura_metodo_codo_silhouette, figura_metodo_codo_distorsion = clustering.valoresSeleccionadosK()
    figuras_clustering_silhouette, precision_clasificacion_clusters_silhouette = clustering.getFigurasClustering(k_seleccionado_slhouette)
    figuras_clustering_distorsion, precision_clasificacion_clusters_distorsion = clustering.getFigurasClustering(k_seleccionado_distorsion)
    return k_seleccionado_slhouette, k_seleccionado_distorsion, figura_metodo_codo_silhouette, figura_metodo_codo_distorsion, figuras_clustering_silhouette, figuras_clustering_distorsion, precision_clasificacion_clusters_silhouette, precision_clasificacion_clusters_distorsion


@st.cache(allow_output_mutation = True, show_spinner = False)
def regresionLogistica(predictores_entrenar, predictores_validacion, predictores_prueba, y_entrenar, y_validacion, y_prueba, predictores_entrenamiento_validacion, y_entrenamiento_validacion, indices_entrenamiento_validacion, nombre_y):
    regresionLogistica = Regresion(predictores_entrenar, predictores_validacion, predictores_prueba, y_entrenar, y_validacion, y_prueba, predictores_entrenamiento_validacion, y_entrenamiento_validacion, indices_entrenamiento_validacion, nombre_y, True)
    puntuacion_mejor_predictor_individual, mejores_predictores_individuales = regresionLogistica.puntuacionesIndividualesRegresion()
    figura_grafica_dos_mejores_predictores_individuales = regresionLogistica.figuraGraficaDosMejoresPredictoresIndividuales(mejores_predictores_individuales)
    figura_importancia_atributos_seleccionados, precision_atributos_seleccionados, importancia_atributos_seleccionados_regresion_logistica = regresionLogistica.pesosAtributosSeleccionados()
    figura_mejores_combinaciones_atributos_seleccionados, precision_mejor_combinacion_atributos_seleccionados, num_atributos_combinacion, importancia_combinacion_atributos_regresion_logistica = regresionLogistica.figuraMejoresCombinacionesAtributosSeleccionados()
    figura_mejor_umbral, precision_mejor_umbral, nombre_modelo_mejor_umbral, mejor_umbral_regresion_logistica = regresionLogistica.seleccionarUmbral()
    figura_comparacion_modelos, mejor_modelo = regresionLogistica.figuraComparacionModelos()
    return puntuacion_mejor_predictor_individual, importancia_atributos_seleccionados_regresion_logistica, importancia_combinacion_atributos_regresion_logistica, figura_importancia_atributos_seleccionados, figura_mejores_combinaciones_atributos_seleccionados, figura_comparacion_modelos, precision_atributos_seleccionados, precision_mejor_combinacion_atributos_seleccionados, num_atributos_combinacion, figura_mejor_umbral, precision_mejor_umbral, nombre_modelo_mejor_umbral, figura_grafica_dos_mejores_predictores_individuales, mejor_umbral_regresion_logistica, mejor_modelo


@st.cache(allow_output_mutation = True, show_spinner = False)
def clasificadorArbolDecision(predictores_entrenar, y_entrenar, predictores_prueba, y_prueba, predictores_entrenamiento_validacion, y_entrenamiento_validacion, indices_entrenamiento_validacion, predictores_entrenar_sin_escalar, y_entrenar_sin_escalar, columna_y):
    arbolDecisionClasificador = ArbolDecision(predictores_entrenar, y_entrenar, predictores_prueba, y_prueba, predictores_entrenamiento_validacion, y_entrenamiento_validacion, indices_entrenamiento_validacion, predictores_entrenar_sin_escalar, y_entrenar_sin_escalar, columna_y, True)
    alfa_arbol, precision_arbol_mejor_alfa, figura_mejor_alfa, modelo_arbol_decision = arbolDecisionClasificador.arbolPodado()
    importancia_atributos_arbol_decision_clasificacion, figura_importancia_atributos = arbolDecisionClasificador.importanciaAtributos()
    figura_grafo_arbol = arbolDecisionClasificador.figuraGrafoArbol()
    return importancia_atributos_arbol_decision_clasificacion, precision_arbol_mejor_alfa, alfa_arbol, figura_importancia_atributos, figura_grafo_arbol, figura_mejor_alfa, modelo_arbol_decision


@st.cache(allow_output_mutation = True, show_spinner = False)
def clasificadorRedNeuronal(predictores_entrenar, y_entrenar, predictores_prueba, y_prueba, predictores_entrenamiento_validacion, y_entrenamiento_validacion, indices_entrenamiento_validacion):
    clasificador_red_neuronal = RedNeuronal(predictores_entrenar, y_entrenar, predictores_prueba, y_prueba, predictores_entrenamiento_validacion, y_entrenamiento_validacion, indices_entrenamiento_validacion, 'result', True)
    puntuacion, modelo = clasificador_red_neuronal.seleccionarHiperparametrosModelo()
    importancia_atributos, figura_importancia_atributos = clasificador_red_neuronal.importanciaAtributos()
    nombre_archivo_figura = clasificador_red_neuronal.figuraGrafoRedNeuronal()
    return importancia_atributos, figura_importancia_atributos, puntuacion, modelo[0], modelo, nombre_archivo_figura


@st.cache(allow_output_mutation = True, show_spinner = False)
def comparacionModelosClasificacion(nombre_y, puntuacion_regresion_mejor_predictor_individual, precision_atributos_seleccionados_regresion_logistica, precision_atributos_seleccionados_arbol_decision_clasificacion, precision_atributos_seleccionados_red_neuronal_clasificacion, precision_mejor_combinacion_regresion_logistica, num_atributos_combinacion_regresion_logistica, precision_umbral_seleccionado_regresion_logistica, umbral_seleccionado_regresion_logistica, nombre_modelo_red_neuronal, alfa_arbol_decision_clasificacion, modelo_regresion_logistica, modelo_arbol_decision_clasificacion, modelo_red_neuronal_clasificacion, importancia_atributos_seleccionados_regresion_logistica, importancia_combinacion_atributos_regresion_logistica, importancia_atributos_arbol_decision, importancia_atributos_red_neuronal, predictores_entrenar, y_entrenar, predictores_prueba, y_prueba, predictores_validacion, y_validacion):
    comparacion_modelos = ComparadorModelos(nombre_y, puntuacion_regresion_mejor_predictor_individual, precision_atributos_seleccionados_regresion_logistica, precision_atributos_seleccionados_arbol_decision_clasificacion, precision_atributos_seleccionados_red_neuronal_clasificacion, precision_mejor_combinacion_regresion_logistica, num_atributos_combinacion_regresion_logistica, nombre_modelo_red_neuronal, alfa_arbol_decision_clasificacion, importancia_atributos_seleccionados_regresion_logistica, importancia_combinacion_atributos_regresion_logistica, importancia_atributos_arbol_decision, importancia_atributos_red_neuronal, True, precision_umbral_seleccionado_regresion_logistica, umbral_seleccionado_regresion_logistica)
    comparacion_modelos.votacionModelos(modelo_regresion_logistica, modelo_arbol_decision_clasificacion, modelo_red_neuronal_clasificacion, predictores_entrenar, y_entrenar, predictores_prueba, y_prueba, predictores_validacion, y_validacion)
    figura_comparacion_modelos, puntuaciones_modelos = comparacion_modelos.figuraComparacionModelos()
    figura_atributo_mayor_importancia, atributo_mayor_importancia, nombre_modelo_mayor_puntuacion = comparacion_modelos.atributoMayorImportancia(predictores_entrenar, y_entrenar, puntuaciones_modelos)
    return figura_comparacion_modelos, figura_atributo_mayor_importancia, atributo_mayor_importancia, nombre_modelo_mayor_puntuacion


@st.cache(allow_output_mutation = True, show_spinner = False)
def regresionLineal(predictores_entrenar, predictores_validacion, predictores_prueba, y_entrenar, y_validacion, y_prueba, predictores_entrenamiento_validacion, y_entrenamiento_validacion, indices_entrenamiento_validacion, nombre_y):
    regresionLineal = Regresion(predictores_entrenar, predictores_validacion, predictores_prueba, y_entrenar, y_validacion, y_prueba, predictores_entrenamiento_validacion, y_entrenamiento_validacion, indices_entrenamiento_validacion, nombre_y, False)
    puntuacion_mejor_predictor_individual, mejores_predictores_individuales = regresionLineal.puntuacionesIndividualesRegresion()
    figura_importancia_atributos_seleccionados, precision_atributos_seleccionados, importancia_atributos_seleccionados = regresionLineal.pesosAtributosSeleccionados()
    figura_mejores_combinaciones_atributos_seleccionados, precision_mejor_combinacion_atributos_seleccionados, num_atributos_combinacion, importancia_combinacion_atributos = regresionLineal.figuraMejoresCombinacionesAtributosSeleccionados()
    figura_comparacion_modelos, mejor_modelo = regresionLineal.figuraComparacionModelos()
    return puntuacion_mejor_predictor_individual, importancia_atributos_seleccionados, importancia_combinacion_atributos, figura_importancia_atributos_seleccionados, figura_mejores_combinaciones_atributos_seleccionados, figura_comparacion_modelos, precision_atributos_seleccionados, precision_mejor_combinacion_atributos_seleccionados, num_atributos_combinacion, mejor_modelo


@st.cache(allow_output_mutation = True, show_spinner = False)
def regresionArbolDecision(predictores_entrenar, y_entrenar, predictores_prueba, y_prueba, predictores_entrenamiento_validacion, y_entrenamiento_validacion, indices_entrenamiento_validacion, predictores_entrenar_sin_escalar, y_entrenar_sin_escalar, columna_y):
    arbolDecisionRegresor = ArbolDecision(predictores_entrenar, y_entrenar, predictores_prueba, y_prueba, predictores_entrenamiento_validacion, y_entrenamiento_validacion, indices_entrenamiento_validacion, predictores_entrenar_sin_escalar, y_entrenar_sin_escalar, columna_y, False)
    alfa_arbol, puntuacion_arbol_mejor_alfa, figura_mejor_alfa, modelo_arbol_decision = arbolDecisionRegresor.arbolPodado()
    importancia_atributos_arbol_decision_clasificacion, figura_importancia_atributos = arbolDecisionRegresor.importanciaAtributos()
    figura_grafo_arbol = arbolDecisionRegresor.figuraGrafoArbol()
    return importancia_atributos_arbol_decision_clasificacion, puntuacion_arbol_mejor_alfa, alfa_arbol, figura_importancia_atributos, figura_grafo_arbol, figura_mejor_alfa, modelo_arbol_decision


@st.cache(allow_output_mutation = True, show_spinner = False)
def regresionRedNeuronal(predictores_entrenar, y_entrenar, predictores_prueba, y_prueba,  predictores_entrenamiento_validacion, y_entrenamiento_validacion, indices_entrenamiento_validacion, nombre_y):
    regresor_red_neuronal = RedNeuronal(predictores_entrenar, y_entrenar, predictores_prueba, y_prueba, predictores_entrenamiento_validacion, y_entrenamiento_validacion, indices_entrenamiento_validacion, nombre_y, False)
    puntuacion, modelo = regresor_red_neuronal.seleccionarHiperparametrosModelo()
    importancia_atributos, figura_importancia_atributos = regresor_red_neuronal.importanciaAtributos()
    nombre_archivo_figura = regresor_red_neuronal.figuraGrafoRedNeuronal()
    return importancia_atributos, figura_importancia_atributos, puntuacion, modelo[0], modelo, nombre_archivo_figura


@st.cache(allow_output_mutation = True, show_spinner = False)
def comparacionModelosRegresion(nombre_y, mejores_predictores_individuales_regresion_lineal, precision_atributos_seleccionados_regresion_lineal, precision_atributos_seleccionados_arbol_decision_regresion, precision_atributos_seleccionados_red_neuronal_regresion, precision_mejor_combinacion_atributos_seleccionados_regresion, num_atributos_regresion, nombre_modelo_red_neuronal, alfa_arbol_decision_regresion, modelo_regresion_lineal, modelo_arbol_decision_regresion, modelo_red_neuronal_regresion, importancia_atributos_seleccionados_regresion_lineal, importancia_combinacion_atributos_regresion_lineal, importancia_atributos_arbol_decision, importancia_atributos_red_neuronal, predictores_entrenar, y_entrenar, predictores_prueba, y_prueba, predictores_validacion, y_validacion):
    comparacion_modelos = ComparadorModelos(nombre_y, mejores_predictores_individuales_regresion_lineal, precision_atributos_seleccionados_regresion_lineal, precision_atributos_seleccionados_arbol_decision_regresion, precision_atributos_seleccionados_red_neuronal_regresion, precision_mejor_combinacion_atributos_seleccionados_regresion, num_atributos_regresion, nombre_modelo_red_neuronal, alfa_arbol_decision_regresion, importancia_atributos_seleccionados_regresion_lineal, importancia_combinacion_atributos_regresion_lineal, importancia_atributos_arbol_decision, importancia_atributos_red_neuronal, False)
    comparacion_modelos.votacionModelos(modelo_regresion_lineal, modelo_arbol_decision_regresion, modelo_red_neuronal_regresion, predictores_entrenar, y_entrenar, predictores_prueba, y_prueba, predictores_validacion, y_validacion)
    figura_comparacion_modelos, puntuaciones_modelos =  comparacion_modelos.figuraComparacionModelos()
    figura_atributo_mayor_importancia, atributo_mayor_importancia, nombre_modelo_mayor_importancia = comparacion_modelos.atributoMayorImportancia(predictores_entrenar, y_entrenar, puntuaciones_modelos)
    return figura_comparacion_modelos, figura_atributo_mayor_importancia, atributo_mayor_importancia, nombre_modelo_mayor_importancia


def atributosCategoricas(columnas):
    return [columnas[3], columnas[7], columnas[14], columnas[16], columnas[18], columnas[20], columnas[21], nombre_atributo_continente, nombre_atributo_lado_mapa]


def mostrarRegresion(figura_importancia_atributos_seleccionados, figura_mejores_combinaciones_atributos_seleccionados, figura_comparacion_modelos, es_clasificacion, figura_mejor_umbral = None, figura_grafica_dos_mejores_predictores_individuales = None, mejor_umbral_regresion_logistica = None):

    if es_clasificacion:
        nombre_tipo_regresion = 'logística'
        nombre_puntuacion = 'la precisión'
    else:
        nombre_tipo_regresion = 'lineal'
        nombre_puntuacion = 'R^2'

    mostrarTitulo(3, 'Regresión ' + nombre_tipo_regresion)

    if figura_grafica_dos_mejores_predictores_individuales is not None:
        mostrarTitulo(4, 'Gráficas de los dos mejores atributos individuales')
        st.pyplot(figura_grafica_dos_mejores_predictores_individuales)

    num_comparacion_figuras = 1
    if figura_importancia_atributos_seleccionados is not None:
        mostrarTitulo(4, 'Pesos del modelo con los 6 atributos seleccionados')
        st.plotly_chart(figura_importancia_atributos_seleccionados)
        num_comparacion_figuras += 1

    if figura_mejores_combinaciones_atributos_seleccionados is not None:
        mostrarTitulo(4, 'Mejor combinación con 3 o menos atributos (entre los 6 atributos seleccionados)')
        st.plotly_chart(figura_mejores_combinaciones_atributos_seleccionados)
        num_comparacion_figuras += 1

    if es_clasificacion and figura_mejor_umbral is not None:
        mostrarTitulo(4, 'Umbral seleccionado' + str(mejor_umbral_regresion_logistica))
        mostrarTitulo(4, 'Curva ROC umbral seleccionado')
        st.plotly_chart(figura_mejor_umbral)

    if num_comparacion_figuras > 1:
        mostrarTitulo(4, 'Comparación de ' + nombre_puntuacion + ' entre los modelos')
        st.plotly_chart(figura_comparacion_modelos)


def mostrarArbolDecision(figura_importancia_atributos, figura_grafo_arbol, alfa_arbol, figura_mejor_alfa_arbol_decision, precision_atributos_seleccionados_arbol_decision, es_clasificacion):
    if es_clasificacion:
        nombre_tipo_arbol = 'Clasificador'
        nombre_puntuacion = 'Precisión'
    else:
        nombre_tipo_arbol = 'Regresión'
        nombre_puntuacion = 'R^2'
    mostrarTitulo(3, nombre_tipo_arbol + ' con árbol de decisión')
    mostrarTitulo(4, 'Seleccionar el valor de alfa para podar el árbol utilizando validación cruzada')
    st.plotly_chart(figura_mejor_alfa_arbol_decision)
    alfa_arbol_mostrar_titulo = '(alfa ' + str(alfa_arbol) + ')'
    mostrarTitulo(4, nombre_puntuacion + ' del árbol seleccionado ' + alfa_arbol_mostrar_titulo + ': ' + str(round(precision_atributos_seleccionados_arbol_decision, 4)))
    mostrarTitulo(4, 'Importancia de los atributos en el árbol de decisión')
    st.plotly_chart(figura_importancia_atributos)
    mostrarTitulo(4, 'Principales nodos del árbol  ' + alfa_arbol_mostrar_titulo)
    st.pyplot(figura_grafo_arbol)


def mostrarRedNeuronal(figura_importancia_atributos, nombre_mejor_modelo_red_neuronal, nombre_archivo_figura, es_clasificacion):
    if es_clasificacion:
        nombre_tipo_red_neuronal = 'clasificador'
        puntuacion = 'precisión'
        separador = ''
    else:
        nombre_tipo_red_neuronal = 'regresión'
        puntuacion = 'R^2'
        separador = '<br>'
    mostrarTitulo(3, nombre_tipo_red_neuronal.capitalize() + ' con red neuronal')
    mostrarTitulo(4, 'Estructura de la red neuronal con mayor ' + puntuacion)
    st.image(nombre_archivo_figura)
    mostrarTitulo(4, 'Importancia de los atributos de la red neuronal con mayor ' + puntuacion + ': ' + separador + nombre_mejor_modelo_red_neuronal.replace('<br>', ' ')[12:])
    st.plotly_chart(figura_importancia_atributos)


def mostrarComparacionModelos(figura_comparacion_modelos, figura_atributo_mayor_importancia, columna_y, atributo_mejor_predictor, nombre_modelo_mayor_puntuacion, es_clasificacion):
    if es_clasificacion:
        nombre_tipo_modelos = 'clasificación'
        mostrar_puntuacion = 'la precisión'
    else:
        nombre_tipo_modelos = 'regresión'
        mostrar_puntuacion = 'R^2'

    titulo_final = ' en los modelos de ' + nombre_tipo_modelos + ' para predecir ' + columna_y
    mostrarTitulo(3, 'Comparación de ' + mostrar_puntuacion +  titulo_final)
    mostrarTituloJustificado(4, 'Se incluye un modelo que combina el mejor modelo de cada tipo (regresión, árbol de decisión y red neuronal) mediante el empleo de votación')
    
    st.plotly_chart(figura_comparacion_modelos)
    nombre_modelo_mayor_puntuacion_formateado = nombre_modelo_mayor_puntuacion[0].lower() + nombre_modelo_mayor_puntuacion[1:].replace('<br>', ' ')
    
    mostrarTitulo(3, 'Determinar el atributo con mayor importancia en el modelo de mayor puntuación<br>(' + nombre_modelo_mayor_puntuacion_formateado + '):<br>' + atributo_mejor_predictor)
    st.plotly_chart(figura_atributo_mayor_importancia)


def mostrarFigurasRegresion(columna_y, figura_importancia_atributos_seleccionados_regresion_lineal, figura_mejores_combinaciones_atributos_seleccionados_regresion_lineal, figura_comparacion_modelos_regresion_lineal, figura_importancia_atributos_arbol_decision_regresion, figura_grafo_arbol_decision_regresion, alfa_arbol_decision_regresion, figura_mejor_alfa_arbol_decision_regresion, figura_importancia_atributos_red_neuronal_regresion, figura_comparacion_modelos_regresion, figura_atributo_mayor_importancia_mejor_predictor, atributo_mejor_predictor, nombre_mejor_modelo_red_neuronal_regresion, precision_atributos_seleccionados_arbol_decision_regresion, nombre_archivo_figura_regresion_red_neuronal, nombre_modelo_mayor_puntuacion):
    mostrarRegresion(figura_importancia_atributos_seleccionados_regresion_lineal, figura_mejores_combinaciones_atributos_seleccionados_regresion_lineal, figura_comparacion_modelos_regresion_lineal, False)
    mostrarArbolDecision(figura_importancia_atributos_arbol_decision_regresion, figura_grafo_arbol_decision_regresion, alfa_arbol_decision_regresion, figura_mejor_alfa_arbol_decision_regresion, precision_atributos_seleccionados_arbol_decision_regresion, False)
    mostrarRedNeuronal(figura_importancia_atributos_red_neuronal_regresion, nombre_mejor_modelo_red_neuronal_regresion, nombre_archivo_figura_regresion_red_neuronal, False)
    mostrarComparacionModelos(figura_comparacion_modelos_regresion, figura_atributo_mayor_importancia_mejor_predictor, columna_y, atributo_mejor_predictor, nombre_modelo_mayor_puntuacion, False)


def mostrarFigurasClasificacion(columna_y, figura_importancia_atributos_seleccionados_regresion_logistica, figura_mejores_combinaciones_atributos_seleccionados_regresion_logistica, figura_comparacion_modelos_regresion_logistica, figura_mejor_umbral_regresion_logistica, figura_grafica_dos_mejores_predictores_individuales_regresion_logistica, figura_importancia_atributos_arbol_decision_clasificacion, figura_grafo_arbol_decision_clasificacion, alfa_arbol_decision_clasificacion, figura_mejor_alfa_arbol_decision_clasificacion, figura_importancia_atributos_red_neuronal_clasificacion, figura_comparacion_modelos_clasificacion, figura_atributo_mayor_importancia_clasificacion, mejor_umbral_regresion_logistica, atributo_mejor_predictor, nombre_mejor_modelo_red_neuronal_clasificacion, precision_atributos_seleccionados_arbol_decision_clasificacion, nombre_archivo_figura_clasificacion_red_neuronal, nombre_modelo_mayor_puntuacion):
    mostrarRegresion(figura_importancia_atributos_seleccionados_regresion_logistica, figura_mejores_combinaciones_atributos_seleccionados_regresion_logistica, figura_comparacion_modelos_regresion_logistica, True, figura_mejor_umbral_regresion_logistica, figura_grafica_dos_mejores_predictores_individuales_regresion_logistica, mejor_umbral_regresion_logistica)
    mostrarArbolDecision(figura_importancia_atributos_arbol_decision_clasificacion, figura_grafo_arbol_decision_clasificacion, alfa_arbol_decision_clasificacion, figura_mejor_alfa_arbol_decision_clasificacion, precision_atributos_seleccionados_arbol_decision_clasificacion, True)
    mostrarRedNeuronal(figura_importancia_atributos_red_neuronal_clasificacion, nombre_mejor_modelo_red_neuronal_clasificacion, nombre_archivo_figura_clasificacion_red_neuronal, True)
    mostrarComparacionModelos(figura_comparacion_modelos_clasificacion, figura_atributo_mayor_importancia_clasificacion, columna_y, atributo_mejor_predictor, nombre_modelo_mayor_puntuacion, True)


def modelosRegresion(predictores_entrenar, predictores_validacion, predictores_prueba, y_entrenar, y_validacion, y_prueba, predictores_entrenamiento_validacion, y_entrenamiento_validacion, indices_entrenamiento_validacion, predictores_entrenar_sin_escalar, y_entrenar_sin_escalar, nombre_y): 
    
    # Regresión lineal
    puntuacion_mejores_predictor_individual_regresion_lineal, importancia_atributos_seleccionados_regresion_lineal, importancia_combinacion_atributos_regresion_lineal, figura_importancia_atributos_seleccionados_regresion_lineal, figura_mejores_combinaciones_atributos_seleccionados_regresion_lineal, figura_comparacion_modelos_regresion_lineal, precision_atributos_seleccionados_regresion_lineal, precision_mejor_combinacion_atributos_seleccionados_regresion_lineal, num_atributos_combinacion_regresion_lineal, modelo_regresion_lineal = regresionLineal(predictores_entrenar, predictores_validacion, predictores_prueba, y_entrenar, y_validacion, y_prueba, predictores_entrenamiento_validacion, y_entrenamiento_validacion, indices_entrenamiento_validacion, nombre_y)

    # Regresión con árbol de decisión
    importancia_atributos_arbol_decision_regresion, precision_atributos_seleccionados_arbol_decision_regresion, alfa_arbol_decision_regresion, figura_importancia_atributos_arbol_decision_regresion, figura_grafo_arbol_decision_regresion, figura_mejor_alfa_arbol_decision_regresion, modelo_arbol_decision_regresion = regresionArbolDecision(predictores_entrenar, y_entrenar, predictores_prueba, y_prueba, predictores_entrenamiento_validacion, y_entrenamiento_validacion, indices_entrenamiento_validacion, predictores_entrenar_sin_escalar, y_entrenar_sin_escalar, nombre_y)

    # Regresión con red neuronal
    importancia_atributos_red_neuronal_regresion, figura_importancia_atributos_red_neuronal_regresion, precision_atributos_seleccionados_red_neuronal_regresion, nombre_modelo_red_neuronal_regresion, modelo_red_neuronal_regresion, nombre_archivo_figura_red_neuronal_regresion = regresionRedNeuronal(predictores_entrenar, y_entrenar, predictores_prueba, y_prueba, predictores_entrenamiento_validacion, y_entrenamiento_validacion, indices_entrenamiento_validacion, nombre_y)

    # Comparación de los modelos de regresión
    figura_comparacion_modelos_regresion, figura_atributo_mayor_importancia_regresion, atributo_mayor_importancia_regresion, nombre_modelo_mayor_importancia_regresion = comparacionModelosRegresion(nombre_y, puntuacion_mejores_predictor_individual_regresion_lineal, precision_atributos_seleccionados_regresion_lineal, precision_atributos_seleccionados_arbol_decision_regresion, precision_atributos_seleccionados_red_neuronal_regresion, precision_mejor_combinacion_atributos_seleccionados_regresion_lineal, num_atributos_combinacion_regresion_lineal, nombre_modelo_red_neuronal_regresion, alfa_arbol_decision_regresion, modelo_regresion_lineal, modelo_arbol_decision_regresion, modelo_red_neuronal_regresion, importancia_atributos_seleccionados_regresion_lineal, importancia_combinacion_atributos_regresion_lineal, importancia_atributos_arbol_decision_regresion, importancia_atributos_red_neuronal_regresion, predictores_entrenar, y_entrenar, predictores_prueba, y_prueba, predictores_validacion, y_validacion)

    return figura_importancia_atributos_seleccionados_regresion_lineal, figura_mejores_combinaciones_atributos_seleccionados_regresion_lineal, figura_comparacion_modelos_regresion_lineal, alfa_arbol_decision_regresion, figura_importancia_atributos_arbol_decision_regresion, figura_grafo_arbol_decision_regresion, figura_mejor_alfa_arbol_decision_regresion, figura_importancia_atributos_red_neuronal_regresion, figura_comparacion_modelos_regresion, figura_atributo_mayor_importancia_regresion, atributo_mayor_importancia_regresion, nombre_modelo_red_neuronal_regresion, precision_atributos_seleccionados_arbol_decision_regresion, nombre_archivo_figura_red_neuronal_regresion, nombre_modelo_mayor_importancia_regresion


def modelosClasificacion(predictores_entrenar, predictores_validacion, predictores_prueba, y_entrenar, y_validacion, y_prueba, predictores_entrenamiento_validacion, y_entrenamiento_validacion, indices_entrenamiento_validacion, predictores_entrenar_sin_escalar, y_entrenar_sin_escalar, nombre_y):
    
    # Regresión logística
    puntuacion_mejor_predictor_individual, importancia_atributos_seleccionados_regresion_logistica, importancia_combinacion_atributos_regresion_logistica, figura_importancia_atributos_seleccionados_regresion_logistica, figura_mejores_combinaciones_atributos_seleccionados_regresion_logistica, figura_comparacion_modelos_regresion_logistica, precision_atributos_seleccionados_regresion_logistica, precision_mejor_combinacion_atributos_seleccionados_regresion_logistica, num_atributos_combinacion_regresion_logistica, figura_mejor_umbral_regresion_logistica, precision_mejor_umbral_regresion_logistica, nombre_modelo_mejor_umbral_regresion_logistica, figura_grafica_dos_mejores_predictores_individuales_regresion_logistica, mejor_umbral_regresion_logistica, modelo_regresion_logistica = regresionLogistica(predictores_entrenar, predictores_validacion, predictores_prueba, y_entrenar, y_validacion, y_prueba, predictores_entrenamiento_validacion, y_entrenamiento_validacion, indices_entrenamiento_validacion, nombre_y)

    # Clasificador con árbol de decisión
    importancia_atributos_arbol_decision_clasificacion, precision_atributos_seleccionados_arbol_decision_clasificacion, alfa_arbol_decision_clasificacion, figura_importancia_atributos_arbol_decision_clasificacion, figura_grafo_arbol_decision_clasificacion, figura_mejor_alfa_arbol_decision_clasificacion, modelo_arbol_decision_clasificacion = clasificadorArbolDecision(predictores_entrenar, y_entrenar, predictores_prueba, y_prueba, predictores_entrenamiento_validacion, y_entrenamiento_validacion, indices_entrenamiento_validacion, predictores_entrenar_sin_escalar, y_entrenar_sin_escalar, nombre_y)

    # Clasificador con red neuronal
    importancia_atributos_red_neuronal_clasificacion, figura_importancia_atributos_red_neuronal_clasificacion, precision_atributos_seleccionados_red_neuronal_clasificacion, nombre_modelo_red_neuronal_clasificacion, modelo_red_neuronal_clasificacion, nombre_archivo_figura_red_neuronal_clasificacion = clasificadorRedNeuronal(predictores_entrenar, y_entrenar, predictores_prueba, y_prueba, predictores_entrenamiento_validacion, y_entrenamiento_validacion, indices_entrenamiento_validacion)

    # Comparación y puntuación general de los atributos en los modelos de clasificación
    figura_comparacion_modelos_clasificacion, figura_atributo_mayor_importancia_clasificacion, atributo_mayor_importancia_clasificacion, nombre_modelo_mayor_puntuacion_clasificacion = comparacionModelosClasificacion(nombre_y, puntuacion_mejor_predictor_individual, precision_atributos_seleccionados_regresion_logistica, precision_atributos_seleccionados_arbol_decision_clasificacion, precision_atributos_seleccionados_red_neuronal_clasificacion, precision_mejor_combinacion_atributos_seleccionados_regresion_logistica, num_atributos_combinacion_regresion_logistica, precision_mejor_umbral_regresion_logistica, mejor_umbral_regresion_logistica, nombre_modelo_red_neuronal_clasificacion, alfa_arbol_decision_clasificacion, modelo_regresion_logistica, modelo_arbol_decision_clasificacion, modelo_red_neuronal_clasificacion, importancia_atributos_seleccionados_regresion_logistica, importancia_combinacion_atributos_regresion_logistica, importancia_atributos_arbol_decision_clasificacion, importancia_atributos_red_neuronal_clasificacion, predictores_entrenar, y_entrenar, predictores_prueba, y_prueba, predictores_validacion, y_validacion)

    return figura_importancia_atributos_seleccionados_regresion_logistica, figura_mejores_combinaciones_atributos_seleccionados_regresion_logistica, figura_comparacion_modelos_regresion_logistica, figura_mejor_umbral_regresion_logistica, figura_grafica_dos_mejores_predictores_individuales_regresion_logistica, alfa_arbol_decision_clasificacion, figura_importancia_atributos_arbol_decision_clasificacion, figura_grafo_arbol_decision_clasificacion, figura_mejor_alfa_arbol_decision_clasificacion, figura_importancia_atributos_red_neuronal_clasificacion, figura_comparacion_modelos_clasificacion, figura_atributo_mayor_importancia_clasificacion, atributo_mayor_importancia_clasificacion, mejor_umbral_regresion_logistica, nombre_modelo_red_neuronal_clasificacion, precision_atributos_seleccionados_arbol_decision_clasificacion, nombre_archivo_figura_red_neuronal_clasificacion, nombre_modelo_mayor_puntuacion_clasificacion


def datosPredecirMejorPredictorPartida(datos_predecir_resultado_partida, nombre_atributo_mejor_predictor_resultado, nombre_atributo_inhibidores, nombre_atributo_resultado, nombre_atributo_barones, nombre_atributo_dragones_ancianos, preprocesamiento_limpieza_datos):

    if nombre_atributo_mejor_predictor_resultado != nombre_atributo_inhibidores:
        # No se incluyen los atributos que pueden ocurrir después en el tiempo que el resto de atributos
        atributos_eliminar = [nombre_atributo_resultado, nombre_atributo_inhibidores, nombre_atributo_barones, nombre_atributo_dragones_ancianos]
    else:
        # Si el atributo a predecir son los inhibidores, no se elimina ese atributo
        atributos_eliminar = [nombre_atributo_resultado, nombre_atributo_barones, nombre_atributo_dragones_ancianos]

    datos_predecir_mejor_predictor_resultado = datos_predecir_resultado_partida.drop(columns = atributos_eliminar, axis = 1)
                
    # Modelos para predecir el atributo más importante para determinar el resultado de la partida
    predictores_entrenar_predecir_mejor_predictor_resultado, predictores_validacion_predecir_mejor_predictor_resultado, predictores_prueba_predecir_mejor_predictor_resultado, y_entrenar_predecir_mejor_predictor_resultado, y_validacion_predecir_mejor_predictor_resultado, y_prueba_predecir_mejor_predictor_resultado, predictores_entrenar_sin_escalar_predecir_mejor_predictor_resultado, y_entrenar_sin_escalar_predecir_mejor_predictor_resultado, predictores_entrenamiento_validacion_predecir_mejor_predictor_resultado, y_entrenamiento_validacion_predecir_mejor_predictor_resultado, indices_entrenamiento_validacion_predecir_mejor_predictor_resultado, num_datos_eliminados_predecir_mejor_predictor_resultado = preprocesamiento_limpieza_datos.dividirDatosEnEntrenamientoTest(datos_predecir_mejor_predictor_resultado, nombre_atributo_mejor_predictor_resultado)

    return predictores_entrenar_predecir_mejor_predictor_resultado, predictores_validacion_predecir_mejor_predictor_resultado, predictores_prueba_predecir_mejor_predictor_resultado, y_entrenar_predecir_mejor_predictor_resultado, y_validacion_predecir_mejor_predictor_resultado, y_prueba_predecir_mejor_predictor_resultado, predictores_entrenar_sin_escalar_predecir_mejor_predictor_resultado, y_entrenar_sin_escalar_predecir_mejor_predictor_resultado, predictores_entrenamiento_validacion_predecir_mejor_predictor_resultado, y_entrenamiento_validacion_predecir_mejor_predictor_resultado, indices_entrenamiento_validacion_predecir_mejor_predictor_resultado, num_datos_eliminados_predecir_mejor_predictor_resultado


def mostrarGrafoEstrategia(atributo_mejor_predictor_resultado_con_relacion, atributo_mejor_predictor_predictor_resultado_con_relacion):
    with open(nombre_fichero_estructura_grafo_estrategia) as estructura_grafo_estrategia:    
        
        grafo_estrategia_info = estructura_grafo_estrategia.read()
        grafo_estrategia_info = grafo_estrategia_info.replace(patron_mejor_predictor_resultado, '   ' + atributo_mejor_predictor_resultado_con_relacion + '   ')
        grafo_estrategia_info = grafo_estrategia_info.replace(patron_mejor_predictor_predictor_resultado, '   ' + atributo_mejor_predictor_predictor_resultado_con_relacion + '   ')

        # Guardar el archivo .dot
        fichero_info_grafo = open(nombre_archivo_grafo_estrategia, 'w')
        fichero_info_grafo.write(grafo_estrategia_info)
        fichero_info_grafo.close()

        # Convertir el archivo .dot a .png
        render(extension_fichero_dot, extension_fichero_png, nombre_archivo_grafo_estrategia)

        # Mostrar imagen
        st.image(nombre_archivo_grafo_estrategia + '.' + extension_fichero_png)




def main():

    columnas_indicadas = False
    columnas = []


    if not columnas_indicadas:
        with st.expander(texto_indicar_caracteriscas_dataset):

            # Indicación del nombre de los atributos por parte del usuario
            entrada_nombres_columnas = st.text_area('Indique los nombres de los atributos sin prefijos y separados por comas, en el orden en el que se indican abajo. Una vez inducidos, pulse Ctrl + Enter.',
                                                    'nombre_atributo_1, nombre_atributo_2, ...')
            columnas = [columna.strip() for columna in entrada_nombres_columnas.split(',')]


    if len(columnas) == 31:

        with st.expander(texto_seleccionar_dataset):
            # Subir el dataset
            fichero_subido = st.file_uploader('', type = extension_fichero_conjunto_datos, accept_multiple_files = False)


        if fichero_subido is not None:


            # Leer los datos
            datos = leerDatos(fichero_subido)

            columnas_equipo = columnasConEtiquetaDeEquipo(columnas, etiquetas_equipo)
            atributos_categoricos = atributosCategoricas(columnas)

            # Comprobar que el dataset subido contiene las columnas
            if set(columnas_equipo).issubset(datos.columns):

                columnas_indicadas = True

                with st.expander(nombre_seccion_atributos_utilizados):
                    mostrarTitulo(3, nombre_seccion_atributos_utilizados)
                    mostrarTituloJustificado(4, 'A continuación, se muestra una breve descripción de los atributos seleccionados')
                    explicaciones_atributos_mostrar = explicacionesAtributos(columnas)
                    mostrarListaTexto(explicaciones_atributos_mostrar)


                # Hacer todo el proceso de análisis

                # Preprocesamiento y análisis exploratorio
                figuras_distribucion_atributos_originales, figuras_analisis_lado_mapa, figuras_analisis_exploratorio_continentes, figura_comparacion_observaciones_eliminadas, datos_clustering,  datos_predecir_resultado_partida, predictores_entrenar_predecir_victoria, predictores_validacion_predecir_victoria, predictores_prueba_predecir_victoria, y_entrenar_predecir_victoria, y_validacion_predecir_victoria, y_prueba_predecir_victoria, predictores_entrenamiento_validacion_predecir_victoria, y_entrenamiento_validacion_predecir_victoria, indices_entrenamiento_validacion_predecir_victoria, predictores_entrenamiento_sin_escalar_predecir_victoria, y_entrenar_sin_escalar_predecir_victoria, preprocesamiento_limpieza_datos = hacerPreprocesemientoAnalisisExploratorio(datos, columnas_equipo, columnas, atributos_categoricos)

                # Clustering 
                k_seleccionado_slhouette, k_seleccionado_distorsion, figura_metodo_codo_silhouette, figura_metodo_codo_distorsion, figuras_clustering_silhouette, figuras_clustering_distorsion, precision_clasificacion_clusters_silhouette, precision_clasificacion_clusters_distorsion = hacerClustering(datos_clustering, nombre_atributo_resultado = columnas[3], atributos_categoricos = atributos_categoricos)
                
                # Clasificación para predecir el resultado de la partida: ganar o perder
                figura_importancia_atributos_seleccionados_regresion_logistica_resultado_partida, figura_mejores_combinaciones_atributos_seleccionados_regresion_logistica_resultado_partida, figura_comparacion_modelos_regresion_logistica_resultado_partida, figura_mejor_umbral_regresion_logistica_resultado_partida, figura_grafica_dos_mejores_predictores_individuales_regresion_logistica_resultado_partida, alfa_arbol_decision_clasificacion_resultado_partida, figura_importancia_atributos_arbol_decision_clasificacion_resultado_partida, figura_grafo_arbol_decision_clasificacion_resultado_partida, figura_mejor_alfa_arbol_decision_clasificacion_resultado_partida, figura_importancia_atributos_red_neuronal_clasificacion_resultado_partida, figura_comparacion_modelos_clasificacion_resultado_partida, figura_atributo_mayor_importancia_clasificacion_resultado_partida, atributo_mejor_predictor_resultado, mejor_umbral_regresion_logistica_resultado_partida, nombre_mejor_modelo_red_neuronal_clasificacion_resultado_partida, precision_atributos_seleccionados_arbol_decision_clasificacion_resultado_partida, nombre_archivo_figura_red_neuronal_clasificacion_resultado_partida, nombre_modelo_mayor_puntuacion_clasificacion_resultado_partida = modelosClasificacion(predictores_entrenar_predecir_victoria, predictores_validacion_predecir_victoria, predictores_prueba_predecir_victoria, y_entrenar_predecir_victoria, y_validacion_predecir_victoria, y_prueba_predecir_victoria, predictores_entrenamiento_validacion_predecir_victoria, y_entrenamiento_validacion_predecir_victoria, indices_entrenamiento_validacion_predecir_victoria, predictores_entrenamiento_sin_escalar_predecir_victoria, y_entrenar_sin_escalar_predecir_victoria, 'result')

                predictores_entrenar_predecir_mejor_predictor_resultado, predictores_validacion_predecir_mejor_predictor_resultado, predictores_prueba_predecir_mejor_predictor_resultado, y_entrenar_predecir_mejor_predictor_resultado, y_validacion_predecir_mejor_predictor_resultado, y_prueba_predecir_mejor_predictor_resultado, predictores_entrenar_sin_escalar_predecir_mejor_predictor_resultado, y_entrenar_sin_escalar_predecir_mejor_predictor_resultado, predictores_entrenamiento_validacion_predecir_mejor_predictor_resultado, y_entrenamiento_validacion_predecir_mejor_predictor_resultado, indices_entrenamiento_validacion_predecir_mejor_predictor_resultado, num_datos_eliminados_predecir_mejor_predictor_resultado = datosPredecirMejorPredictorPartida(datos_predecir_resultado_partida, atributo_mejor_predictor_resultado, nombre_atributo_inhibidores = columnas[22], nombre_atributo_barones = columnas[17], nombre_atributo_dragones_ancianos = columnas[13], nombre_atributo_resultado = columnas[3], preprocesamiento_limpieza_datos = preprocesamiento_limpieza_datos)
                
                # Modelos para predecir el atributo más importante para determinar el resultado de la partida
                if atributo_mejor_predictor_resultado in atributos_categoricos:
                    atributo_mejor_predictor_resultado_es_categorico = True
                    figura_importancia_atributos_seleccionados_regresion_logistica_mejor_predictor_resultado, figura_mejores_combinaciones_atributos_seleccionados_regresion_logistica_mejor_predictor_resultado, figura_comparacion_modelos_regresion_logistica_mejor_predictor_resultado, figura_mejor_umbral_regresion_logistica_mejor_predictor_resultado, figura_grafica_dos_mejores_predictores_individuales_regresion_logistica_mejor_predictor_resultado, alfa_arbol_decision_clasificacion_mejor_predictor_resultado, figura_importancia_atributos_arbol_decision_clasificacion_mejor_predictor_resultado, figura_grafo_arbol_decision_clasificacion_mejor_predictor_resultado, figura_mejor_alfa_arbol_decision_clasificacion_mejor_predictor_resultado, figura_importancia_atributos_red_neuronal_clasificacion_mejor_predictor_resultado, figura_comparacion_modelos_clasificacion_mejor_predictor_resultado, figura_atributo_mayor_importancia_clasificacion_mejor_predictor_resultado, atributo_mejor_predictor_predictor_resultado, mejor_umbral_regresion_logistica_mejor_predictor_resultado, nombre_mejor_modelo_red_neuronal_mejor_predictor_resultado, precision_atributos_seleccionados_arbol_decision_clasificacion_mejor_predictor_resultado, nombre_archivo_figura_red_neuronal_clasificacion_mejor_predictor_resultado, nombre_modelo_mayor_puntuacion_predecir_mejor_predictor_resultado = modelosClasificacion(predictores_entrenar_predecir_mejor_predictor_resultado, predictores_validacion_predecir_mejor_predictor_resultado, predictores_prueba_predecir_mejor_predictor_resultado, y_entrenar_predecir_mejor_predictor_resultado, y_validacion_predecir_mejor_predictor_resultado, y_prueba_predecir_mejor_predictor_resultado, predictores_entrenamiento_validacion_predecir_mejor_predictor_resultado, y_entrenamiento_validacion_predecir_mejor_predictor_resultado, indices_entrenamiento_validacion_predecir_mejor_predictor_resultado, predictores_entrenar_sin_escalar_predecir_mejor_predictor_resultado, y_entrenar_sin_escalar_predecir_mejor_predictor_resultado, atributo_mejor_predictor_resultado)
            
                else:
                    atributo_mejor_predictor_resultado_es_categorico = False
                    figura_importancia_atributos_seleccionados_regresion_lineal_mejor_predictor_resultado, figura_mejores_combinaciones_atributos_seleccionados_regresion_lineal_mejor_predictor_resultado, figura_comparacion_modelos_regresion_lineal_mejor_predictor_resultado, alfa_arbol_decision_regresion_mejor_predictor_resultado, figura_importancia_atributos_arbol_decision_regresion_mejor_predictor_resultado, figura_grafo_arbol_decision_regresion_mejor_predictor_resultado, figura_mejor_alfa_arbol_decision_regresion_mejor_predictor_resultado, figura_importancia_atributos_red_neuronal_regresion_mejor_predictor_resultado, figura_comparacion_modelos_regresion_mejor_predictor_resultado, figura_atributo_mayor_importancia_mejor_predictor_resultado , atributo_mejor_predictor_predictor_resultado, nombre_mejor_modelo_red_neuronal_mejor_predictor_resultado, precision_atributos_seleccionados_arbol_decision_regresion_mejor_predictor_resultado, nombre_archivo_figura_red_neuronal_regresion_mejor_predictor_resultado, nombre_modelo_mayor_puntuacion_predecir_mejor_predictor_resultado = modelosRegresion(predictores_entrenar_predecir_mejor_predictor_resultado, predictores_validacion_predecir_mejor_predictor_resultado, predictores_prueba_predecir_mejor_predictor_resultado, y_entrenar_predecir_mejor_predictor_resultado, y_validacion_predecir_mejor_predictor_resultado, y_prueba_predecir_mejor_predictor_resultado, predictores_entrenamiento_validacion_predecir_mejor_predictor_resultado, y_entrenamiento_validacion_predecir_mejor_predictor_resultado, indices_entrenamiento_validacion_predecir_mejor_predictor_resultado, predictores_entrenar_sin_escalar_predecir_mejor_predictor_resultado, y_entrenar_sin_escalar_predecir_mejor_predictor_resultado, atributo_mejor_predictor_resultado)


                # Selector de secciones
                with st.sidebar:
                    secciones_seleccionadas = st.multiselect(
                                            texto_seleccionar_seciones,
                                            [nombre_seccion_analisis_exploratorio_datos, nombre_seccion_analisis_lado_mapa, nombre_seccion_analisis_continentes, nombre_seccion_clustering, nombre_seccion_predecir_resultado_partida, nombre_seccion_predecir_mejor_predictor_resultado_partida])
                aumentarAnchuraBarraLateral()


                # Mostrar la información en función de las secciones seleccionadas

                # Mostrar información sobre el procesamiento del dataset
                with st.expander(nombre_seccion_procesado_dataset):
                    mostrarTitulo(2, nombre_seccion_procesado_dataset)
                    mostrarTitulo(3, 'Observaciones eliminadas respecto al total')
                    mostrarTituloJustificado(4, 'Nótese que la información de la partida se ha separado por equipos, por lo que si se quiere saber la información por partida en general habría que dividir entre dos.')
                    st.plotly_chart(figura_comparacion_observaciones_eliminadas)

                # Mostrar el análisis exploratorio de datos
                if nombre_seccion_analisis_exploratorio_datos in secciones_seleccionadas:
                    with st.expander(nombre_seccion_analisis_exploratorio_datos):
                        mostrarTitulo(2, nombre_seccion_analisis_exploratorio_datos)
                        mostrarTitulo(3, 'Distribución de los valores de los atributos originales')
                        mostrarVariasFiguras(figuras_distribucion_atributos_originales)

                # Mostrar el análisis por continentes
                if nombre_seccion_analisis_continentes in secciones_seleccionadas:
                    with st.expander(nombre_seccion_analisis_continentes):
                        mostrarTitulo(2, nombre_seccion_analisis_continentes)
                        mostrarTituloJustificado(3, 'La información de los continentes se ha encontrado mediante web scrapping en las webs: ')
                        columna_izquierda, columna_central, columna_derecha = st.columns(3)
                        st.markdown('##', unsafe_allow_html = True)
                        with columna_central:
                            st.write('- [Lol Fandom Wiki](' + enlace_web_lol_fandom_wiki + ')')
                            st.write('- [Torneos de LoL (Wikipedia)](' + enlace_web_wikipedia_torneos_lol + ')')
                        titulos_figuras_analisis_continentes = ['Número de partidas de cada continente', 'Cantidad de oro ganado en los 10 primeros minutos en función del continente', 'Cantidad de asesinatos a campeones enemigos en los 10 primeros minutos en función del continente', 'Cantidad de monstruos grandes (dragones, barones Nashor y heraldos) en función del continente']
                        mostrarFigurasAnalisisExploratorio(figuras_analisis_exploratorio_continentes, titulos_figuras_analisis_continentes)

                # Mostrar el ánalisis según el lado del mapa
                if nombre_seccion_analisis_lado_mapa in secciones_seleccionadas:
                    with st.expander(nombre_seccion_analisis_lado_mapa):
                        mostrarTitulo(2, nombre_seccion_analisis_lado_mapa)
                        titulo_figuras_analisis_lado_mapa = ['Porcentaje de victorias general en función del lado del mapa', 'Porcentaje de victorias por cuartil de duración (lado azul)', 'Porcentaje de victorias por cuartil de duración (lado rojo)']
                        mostrarFigurasAnalisisExploratorio(figuras_analisis_lado_mapa, titulo_figuras_analisis_lado_mapa)

                # Mostrar el clustering 
                if nombre_seccion_clustering in secciones_seleccionadas:
                    with st.expander(nombre_seccion_clustering):
                        mostrarTitulo(2, nombre_seccion_clustering)
                        mostrarTitulo(3, 'Método para realizar el clustering: KMedoides')
                        mostrarFigurasClustering(k_seleccionado_slhouette, figuras_clustering_silhouette, figura_metodo_codo_silhouette, precision_clasificacion_clusters_silhouette, 'Silhouette')
                        mostrarFigurasClustering(k_seleccionado_distorsion, figuras_clustering_distorsion, figura_metodo_codo_distorsion, precision_clasificacion_clusters_distorsion, 'la distorsión')

                # Mostrar la clasificación para predecir el resultado de la partida
                if nombre_seccion_predecir_resultado_partida in secciones_seleccionadas:
                    with st.expander(nombre_seccion_predecir_resultado_partida):
                        mostrarTitulo(2, nombre_seccion_predecir_resultado_partida)
                        mostrarTitulo(3, 'Para cada modelo, se seleccionan automáticamente 6 atributos')
                        mostrarFigurasClasificacion('result', figura_importancia_atributos_seleccionados_regresion_logistica_resultado_partida, figura_mejores_combinaciones_atributos_seleccionados_regresion_logistica_resultado_partida, figura_comparacion_modelos_regresion_logistica_resultado_partida, figura_mejor_umbral_regresion_logistica_resultado_partida, figura_grafica_dos_mejores_predictores_individuales_regresion_logistica_resultado_partida, figura_importancia_atributos_arbol_decision_clasificacion_resultado_partida, figura_grafo_arbol_decision_clasificacion_resultado_partida, alfa_arbol_decision_clasificacion_resultado_partida, figura_mejor_alfa_arbol_decision_clasificacion_resultado_partida, figura_importancia_atributos_red_neuronal_clasificacion_resultado_partida, figura_comparacion_modelos_clasificacion_resultado_partida, figura_atributo_mayor_importancia_clasificacion_resultado_partida, mejor_umbral_regresion_logistica_resultado_partida, atributo_mejor_predictor_resultado, nombre_mejor_modelo_red_neuronal_clasificacion_resultado_partida, precision_atributos_seleccionados_arbol_decision_clasificacion_resultado_partida, nombre_archivo_figura_red_neuronal_clasificacion_resultado_partida, nombre_modelo_mayor_puntuacion_clasificacion_resultado_partida)

                # Mostrar los modelos para predecir el atributo más importante para determinar el resultado de la partida
                if nombre_seccion_predecir_mejor_predictor_resultado_partida in secciones_seleccionadas:
                    with st.expander(nombre_seccion_predecir_mejor_predictor_resultado_partida):
                        mostrarTitulo(2, nombre_seccion_predecir_mejor_predictor_resultado_partida + ': ' + atributo_mejor_predictor_resultado)
                        if atributo_mejor_predictor_resultado_es_categorico:
                            mostrarFigurasClasificacion(atributo_mejor_predictor_resultado, figura_importancia_atributos_seleccionados_regresion_logistica_mejor_predictor_resultado, figura_mejores_combinaciones_atributos_seleccionados_regresion_logistica_mejor_predictor_resultado, figura_comparacion_modelos_regresion_logistica_mejor_predictor_resultado, figura_mejor_umbral_regresion_logistica_mejor_predictor_resultado, figura_grafica_dos_mejores_predictores_individuales_regresion_logistica_mejor_predictor_resultado, figura_importancia_atributos_arbol_decision_clasificacion_mejor_predictor_resultado, figura_grafo_arbol_decision_clasificacion_mejor_predictor_resultado, alfa_arbol_decision_clasificacion_mejor_predictor_resultado, figura_mejor_alfa_arbol_decision_clasificacion_mejor_predictor_resultado, figura_importancia_atributos_red_neuronal_clasificacion_mejor_predictor_resultado, figura_comparacion_modelos_clasificacion_mejor_predictor_resultado, figura_atributo_mayor_importancia_clasificacion_mejor_predictor_resultado, mejor_umbral_regresion_logistica_mejor_predictor_resultado, atributo_mejor_predictor_predictor_resultado, nombre_mejor_modelo_red_neuronal_mejor_predictor_resultado, precision_atributos_seleccionados_arbol_decision_clasificacion_mejor_predictor_resultado, nombre_archivo_figura_red_neuronal_clasificacion_mejor_predictor_resultado, nombre_modelo_mayor_puntuacion_predecir_mejor_predictor_resultado, nombre_modelo_mayor_puntuacion_predecir_mejor_predictor_resultado)
                                        
                        else:
                            mostrarFigurasRegresion(atributo_mejor_predictor_resultado, figura_importancia_atributos_seleccionados_regresion_lineal_mejor_predictor_resultado, figura_mejores_combinaciones_atributos_seleccionados_regresion_lineal_mejor_predictor_resultado, figura_comparacion_modelos_regresion_lineal_mejor_predictor_resultado, figura_importancia_atributos_arbol_decision_regresion_mejor_predictor_resultado, figura_grafo_arbol_decision_regresion_mejor_predictor_resultado, alfa_arbol_decision_regresion_mejor_predictor_resultado, figura_mejor_alfa_arbol_decision_regresion_mejor_predictor_resultado, figura_importancia_atributos_red_neuronal_regresion_mejor_predictor_resultado, figura_comparacion_modelos_regresion_mejor_predictor_resultado, figura_atributo_mayor_importancia_mejor_predictor_resultado, atributo_mejor_predictor_predictor_resultado, nombre_mejor_modelo_red_neuronal_mejor_predictor_resultado, precision_atributos_seleccionados_arbol_decision_regresion_mejor_predictor_resultado, nombre_archivo_figura_red_neuronal_regresion_mejor_predictor_resultado, nombre_modelo_mayor_puntuacion_predecir_mejor_predictor_resultado)
                 
                        mostrarTitulo(3, 'Estrategia sugerida')
                        atributo_mejor_predictor_resultado_con_relacion = tipoRelacion(datos_predecir_resultado_partida, atributo_mejor_predictor_resultado, 'result') + ' ' + atributo_mejor_predictor_resultado  
                        atributo_mejor_predictor_predictor_resultado_con_relacion = tipoRelacion(datos_predecir_resultado_partida, atributo_mejor_predictor_predictor_resultado, atributo_mejor_predictor_resultado) + ' ' + atributo_mejor_predictor_predictor_resultado
                        mostrarGrafoEstrategia(atributo_mejor_predictor_resultado_con_relacion, atributo_mejor_predictor_predictor_resultado_con_relacion)           
   


    else:
        mostrarTextoJustificado('No ha indicado las columnas correctamente. Por favor, indica el nombre de las columnas que se describen en la parte inferior.')
        mostrarTextoJustificado('El conjunto de datos debe estar en formato CSV y almacenar la información de la partida en cada fila. Debe contener los siguientes atributos en el orden en el que se indican, por cada uno de los dos equipos de esa partida, estando en primer lugar todos los atributos referentes al equipo azul y después los del equipo rojo. Para indicar que los atributos se refieren al equipo azul, debe escribirse el prefijo t1_ delante del atributo, siendo t2_ el prefijo para el equipo rojo.')
        mostrarListaTexto(explicacionesAtributosPrimeraLetraMayuscula())
        



if __name__ == '__main__':
    main()
