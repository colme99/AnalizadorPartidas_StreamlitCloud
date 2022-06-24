


import pandas as pd
import numpy as np

# Gráficos
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots



class AnalisisExploratorio:



    POSX_TITULO_CENTRADO = 0.5
    POSY_TITULO_CENTRADO = 0.9
    UMBRAL_PORCENTAJE_VICTORIA_DERROTA = 50
    GROSOR_LINEA_PUNTOS = 2
    SEPARACION_BARRAS = 0.2
    ANCHURA_GRAFICA = 700
    ALTURA_GRAFICA = 500
    ALTURA_GRAFICA_DISTRIBUCION_VALORES = 400
    VALOR_JITTER = 1
    UMBRAL_Q1 = 0.25
    UMBRAL_Q2 = 0.5
    UMBRAL_Q3 = 0.75
    colores_continentes = ['lightcoral', 'lightgreen', 'lightskyblue', 'mediumpurple', 'gold']
    colores_lados_mapa = ['cornflowerblue', 'lightcoral']
    color_negro = 'black'
    nombres_lados_mapa = ['Lado azul', 'Lado rojo']
    etiquetas_lados_mapa = ['azul', 'rojo']
    etiquetas_atributos_monstruos_grandes = ['Dragones', 'Barones Nashor', 'Heraldos']
    etiquetas_atributos_dominio10min = ['Oro 10 min', 'Asesinatos 10 min']
    nombre_color = 'colors'
    nombre_num_partidas = 'matches'
    modo_barras_en_grupo = 'group'
    modo_linea_puntos = 'dash'
    color_linea_puntos = 'black'
    tipo_array_datos = 'data'
    modo_incluir_atipicos = 'outliers'
    modo_histograma_porcentaje = 'percent'
    modo_grafica_puntos = 'markers'
    titulo_porcentaje_victorias_lado_mapa = 'Porcentaje de victorias general según el lado del mapa'
    etiqueta_lado_mapa = 'Lado del mapa'
    etiqueta_porcentaje_victorias = 'Porcentaje de victoria'
    etiqueta_x = 'x'
    etiqueta_y = 'y'
    etiqueta_num_partidas = 'Nº Partidas'
    etiqueta_continentes = 'Continentes'
    etiqueta_histograma = "Histograma (%)"
    etiqueta_reduccion_atipicos = 'Reducción de valores atípicos'
    etiqueta_distribucion_original = 'Original'
    etiqueta_duracion_partida = 'Duración de partida (minutos)'
    etiqueta_partidas_normales = 'Partidas normales'
    etiqueta_partidas_valor_atipico = 'Partidas identificadas como valor atípico'
    etiquetas_componentes = ['Componente 1', 'Componente 2', 'Componente 1 + Componente 2']
    titulo_grafica_oro10min = 'Oro en 10 min por continentes'
    titulo_grafica_asesinatos10min = 'Asesinatos en 10 min por continentes'
    titulo_grafica_monstruos_grandes = 'Monstruos grandes por continentes'
    titulo_grafica_num_partidas = 'Datos de cada continente'
    titulo_grafica_porcentaje_victorias_cuartil_tiempo_inicio = 'Porcentaje de victorias por cuartil de duración de partida (lado '
    titulo_grafica_porcentaje_victorias_cuartil_tiempo_fin = ')'
    titulo_grafica_distribucion_valores_inicio = 'Distribución de valores de '
    titulo_grafica_comparacion_distribucion_comparacion_atipicos_inicio = 'Reducción de valores atípicos en '
    titulo_grafica_visualizar_valores_atipicos = 'Visualización de valores atípicos (reducción de dimensionalidad con PCA)'
    separador = '_'
    etiqueta_desviacion_estandar = 'std'



    def __init__(self, nombre_atributo_oro_10min, nombre_atributo_asesinatos_10min, nombre_atributo_dragones, nombre_atributo_barones, nombre_atributo_heraldos, nombre_atributo_lado_mapa, nombre_atributo_resultado_partida, nombre_atributo_duracion_partida, nombre_atributo_continente, nombre_atributo_equipo):
        self.nombres_atributos_dominio10min = [nombre_atributo_oro_10min, nombre_atributo_asesinatos_10min]
        self.nombres_atributos_monstruos_grandes = [nombre_atributo_dragones, nombre_atributo_barones, nombre_atributo_heraldos]
        self.nombre_atributo_lado_mapa = nombre_atributo_lado_mapa
        self.nombre_atributo_resultado_partida = nombre_atributo_resultado_partida
        self.nombre_atributo_duracion_partida = nombre_atributo_duracion_partida
        self.nombre_atributo_continente = nombre_atributo_continente
        self.nombre_atributo_equipo = nombre_atributo_equipo



    def figuraDistribucionAtributos(self, datos):
        num_columnas = datos.shape[1]

        figuras_distribucion_atributos = []

        for i in range(num_columnas):

            figura = make_subplots(rows = 1, cols = 2)
            figura.add_trace(go.Histogram(x = datos.iloc[:,i], marker_color = self.colores_lados_mapa[0], 
                                        histnorm = self.modo_histograma_porcentaje, name = self.etiqueta_histograma), row = 1, col = 1)
            figura.add_trace(go.Box(x = datos.iloc[:,i], boxpoints = self.modo_incluir_atipicos, name = '', 
                                    jitter = self.VALOR_JITTER, marker_color = self.colores_lados_mapa[1], boxmean = True), row = 1, col = 2)

            figura.update_layout(height = self.ALTURA_GRAFICA_DISTRIBUCION_VALORES, width = self.ANCHURA_GRAFICA, title = self.titulo_grafica_distribucion_valores_inicio + datos.columns[i],
                                title_x = self.POSX_TITULO_CENTRADO, title_y = self.POSY_TITULO_CENTRADO, bargap = self.SEPARACION_BARRAS, showlegend = False)

            figuras_distribucion_atributos.append(figura)

        return figuras_distribucion_atributos


    def calcularPorcentajeVictoriasLadosMapa(self, datos):    
        partidas_lado_azul = datos[datos[self.nombre_atributo_lado_mapa] == 0]
        porcentaje_victorias_lado_azul = partidas_lado_azul[partidas_lado_azul[self.nombre_atributo_resultado_partida] == 1].shape[0] / partidas_lado_azul.shape[0] * 100
        porcentaje_victorias_lado_rojo = 100 - porcentaje_victorias_lado_azul
        return [porcentaje_victorias_lado_azul, porcentaje_victorias_lado_rojo]


    def graficaPorcentajeVictoriasLadosMapa(self, porcentajes_victoria_lados_mapa, nombres_lados_mapa):    
        figura = px.bar(x = nombres_lados_mapa, y = porcentajes_victoria_lados_mapa,
                        title = self.titulo_porcentaje_victorias_lado_mapa,
                        labels = {
                                    self.etiqueta_x: self.etiqueta_lado_mapa,
                                    self.etiqueta_y: self.etiqueta_porcentaje_victorias
                                })
        figura.add_hline(y = self.UMBRAL_PORCENTAJE_VICTORIA_DERROTA, line_width = self.GROSOR_LINEA_PUNTOS, line_dash = self.modo_linea_puntos, line_color = self.color_linea_puntos)
        figura.update_layout(title_x = self.POSX_TITULO_CENTRADO, title_y = self.POSY_TITULO_CENTRADO)
        figura.update_traces(showlegend = False, marker_color = self.colores_lados_mapa)
        return figura


    def nombresCuartilesTiempo(slef, umbral_1, umbral_2, umbral_3):
        return ['< ' + str(round(umbral_1/60)) + "'", str(round(umbral_1/60)) + "' - " + str(round(umbral_2/60)) + "'", 
                        str(round(umbral_2/60)) + "' - " + str(round(umbral_3/60)) + "'", '> ' + str(round(umbral_3/60)) + "'"]


    def cuartilesLongitudPartida(self, datos):    
        umbral_1 = datos[self.nombre_atributo_duracion_partida].quantile(self.UMBRAL_Q1)
        umbral_2 = datos[self.nombre_atributo_duracion_partida].quantile(self.UMBRAL_Q2)
        umbral_3 = datos[self.nombre_atributo_duracion_partida].quantile(self.UMBRAL_Q3)

        nombres_cuartiles = self.nombresCuartilesTiempo(umbral_1, umbral_2, umbral_3)
                
        datos_longitud_q1 = datos[datos[self.nombre_atributo_duracion_partida] <= umbral_1]
        datos_longitud_q2 = datos[(datos[self.nombre_atributo_duracion_partida] > umbral_1) & (datos[self.nombre_atributo_duracion_partida] <= umbral_2)]
        datos_longitud_q3 = datos[(datos[self.nombre_atributo_duracion_partida] > umbral_2) & (datos[self.nombre_atributo_duracion_partida] <= umbral_3)]
        datos_longitud_q4 = datos[datos[self.nombre_atributo_duracion_partida] > umbral_3]
        
        return datos_longitud_q1, datos_longitud_q2, datos_longitud_q3, datos_longitud_q4, nombres_cuartiles

    
    def datosLado(self, datos, lado):
        return datos[datos[self.nombre_atributo_lado_mapa] == lado]


    def porcentajeVictoriaLado(self, datos):
        return datos[datos[self.nombre_atributo_resultado_partida] == 1].shape[0] / datos.shape[0] * 100


    def porcentajesVictoriaLados(self, datos_q1, datos_q2, datos_q3, datos_q4, lado):
        datos_q1_lado = self.datosLado(datos_q1, lado)
        datos_q2_lado = self.datosLado(datos_q2, lado)
        datos_q3_lado = self.datosLado(datos_q3, lado)
        datos_q4_lado = self.datosLado(datos_q4, lado)
        
        return [self.porcentajeVictoriaLado(datos_q1_lado), self.porcentajeVictoriaLado(datos_q2_lado), 
                self.porcentajeVictoriaLado(datos_q3_lado), self.porcentajeVictoriaLado(datos_q4_lado)]
    

    def graficaPorcentajeVictoriasPorCuartilTiempo(self, nombres_cuartiles, porcentajes_victorias, nombre_lado, color_lado):
        figura = px.bar(x = nombres_cuartiles, y = porcentajes_victorias,
                        title = self.titulo_grafica_porcentaje_victorias_cuartil_tiempo_inicio + nombre_lado + self.titulo_grafica_porcentaje_victorias_cuartil_tiempo_fin,
                        labels = {
                                    self.etiqueta_x: self.etiqueta_duracion_partida,
                                    self.etiqueta_y: self.etiqueta_porcentaje_victorias
                                })
        figura.add_hline(y = self.UMBRAL_PORCENTAJE_VICTORIA_DERROTA, line_width = self.GROSOR_LINEA_PUNTOS, line_dash = self.modo_linea_puntos, line_color = self.color_linea_puntos)
        figura.update_layout(title_x = self.POSX_TITULO_CENTRADO, title_y = self.POSY_TITULO_CENTRADO)
        figura.update_traces(showlegend = False, marker_color = color_lado)
        return figura
        
    
    def graficaPorcentajeVictoriasCuartilTiempoSegunLadoMapa(self, datos):
        datos_longitud_q1, datos_longitud_q2, datos_longitud_q3, datos_longitud_q4, nombres_cuartiles = self.cuartilesLongitudPartida(datos)
        porcentaje_victorias_lado_azul = self.porcentajesVictoriaLados(datos_longitud_q1, datos_longitud_q2, datos_longitud_q3, datos_longitud_q4, 0)
        porcentaje_victorias_lado_rojo = self.porcentajesVictoriaLados(datos_longitud_q1, datos_longitud_q2, datos_longitud_q3, datos_longitud_q4, 1)
        figura_porcentaje_victorias_cuartil_tiempo_lado_azul = self.graficaPorcentajeVictoriasPorCuartilTiempo(nombres_cuartiles, porcentaje_victorias_lado_azul, self.etiquetas_lados_mapa[0], self.colores_lados_mapa[0])
        figura_porcentaje_victorias_cuartil_tiempo_lado_rojo = self.graficaPorcentajeVictoriasPorCuartilTiempo(nombres_cuartiles, porcentaje_victorias_lado_rojo, self.etiquetas_lados_mapa[1], self.colores_lados_mapa[1])
        return figura_porcentaje_victorias_cuartil_tiempo_lado_azul, figura_porcentaje_victorias_cuartil_tiempo_lado_rojo
    

    def graficaBarrasAgrupadas(self, datos, nombres_columnas, etiquetas_columnas, titulo):
        nombres_barras = datos[self.nombre_atributo_continente]
        figura = go.Figure()

        for i in range(len(nombres_columnas)):
            figura.add_trace(go.Bar(name = etiquetas_columnas[i], x = nombres_barras, y = datos[nombres_columnas[i]], 
                                    error_y = dict(type = self.tipo_array_datos, array = datos[nombres_columnas[i] + self.separador + self.etiqueta_desviacion_estandar])
                                    )
                            )

        figura.update_layout(title = titulo, title_x = self.POSX_TITULO_CENTRADO, title_y = self.POSY_TITULO_CENTRADO, 
                            barmode = self.modo_barras_en_grupo, autosize = False, width = self.ANCHURA_GRAFICA, height = self.ALTURA_GRAFICA)

        figura.update_traces(error_y_thickness = 0.5, error_y_color = self.color_negro)
        return figura


    def mediaPorContinenteOrdenado(self, datos, columnas):
        media_por_continente = datos.groupby(self.nombre_atributo_continente).mean()[columnas]
        for columna in columnas:
            media_por_continente[columna + self.separador + self.etiqueta_desviacion_estandar] = datos.groupby(self.nombre_atributo_continente).std()[columna]
        media_por_continente[self.nombre_color] = self.colores_continentes
        media_por_continente = media_por_continente.sort_values(columnas, ascending = False)
        media_por_continente[self.nombre_atributo_continente] = media_por_continente.index
        return media_por_continente.reset_index(drop = True)


    def graficaCantidadPartidas(self, datos):
        figura = px.bar(datos, x = self.nombre_atributo_continente, y = self.nombre_num_partidas, title = self.titulo_grafica_num_partidas,
                        labels = {
                            self.nombre_atributo_continente: self.etiqueta_continentes,
                            self.nombre_num_partidas: self.etiqueta_num_partidas
                        })

        figura.update_traces(showlegend = False, marker_color = datos[self.nombre_color])
        figura.update_layout(title_x = self.POSX_TITULO_CENTRADO, title_y = self.POSY_TITULO_CENTRADO)
        return figura


    def graficaContinentes(self, datos, nombre_y, etiqueta_y, titulo):
        figura = px.scatter(datos, x = self.nombre_atributo_continente, y = nombre_y, title = titulo, error_y = nombre_y + self.separador + self.etiqueta_desviacion_estandar,
                            labels = {
                                self.nombre_atributo_continente: self.etiqueta_continentes,
                                nombre_y: etiqueta_y
                            })
        figura.update_traces(showlegend = False, marker_color = datos[self.nombre_color],
                            error_y_thickness = 0.5, error_y_color = self.color_negro,
                            marker = dict(size = 12,
                                            line = dict(width = 1, color = self.color_negro)
                                         )
                            )

        figura.update_layout(title_x = self.POSX_TITULO_CENTRADO, title_y = self.POSY_TITULO_CENTRADO)
        return figura


    def partidasPorContinentes(self, datos):
        datos_por_continente = datos.groupby(self.nombre_atributo_continente).count()

        # Se podría haber utilizado cualquier otra columna, pues el número (count) es el mismo
        partidos_por_continente = pd.DataFrame(np.transpose([datos_por_continente.index, datos_por_continente[self.nombre_atributo_resultado_partida], self.colores_continentes]), 
                                                columns = [self.nombre_atributo_continente, self.nombre_num_partidas, self.nombre_color])
        
        partidos_por_continente = partidos_por_continente.sort_values(self.nombre_num_partidas, ascending = False)
        partidos_por_continente.reset_index(drop = True, inplace = True)
        return partidos_por_continente


    def graficaDominio10MinPorContinentes(self, datos):
        dominio_10min_por_continente = self.mediaPorContinenteOrdenado(datos, self.nombres_atributos_dominio10min)
        figura_oro_10min = self.graficaContinentes(dominio_10min_por_continente, self.nombres_atributos_dominio10min[0], self.etiquetas_atributos_dominio10min[0], self.titulo_grafica_oro10min)
        figura_asesinatos_10min = self.graficaContinentes(dominio_10min_por_continente, self.nombres_atributos_dominio10min[1], self.etiquetas_atributos_dominio10min[1], self.titulo_grafica_asesinatos10min)
        return figura_oro_10min, figura_asesinatos_10min
    

    def graficaMonstruosGrandesPorContinentes(self, datos):
        monstruos_grandes_por_continente = self.mediaPorContinenteOrdenado(datos, self.nombres_atributos_monstruos_grandes)
        figura_monstruos_grandes = self.graficaBarrasAgrupadas(monstruos_grandes_por_continente, self.nombres_atributos_monstruos_grandes, self.etiquetas_atributos_monstruos_grandes, self.titulo_grafica_monstruos_grandes)
        return figura_monstruos_grandes


    def figurasAnalisisExploratorioContinentes(self, datos):
        partidos_por_continente = self.partidasPorContinentes(datos)
        figura_partidas_continente = self.graficaCantidadPartidas(partidos_por_continente)
        figura_oro_10min, figura_asesinatos_10min = self.graficaDominio10MinPorContinentes(datos)
        figura_monstruos_grandes_continentes = self.graficaMonstruosGrandesPorContinentes(datos)
        figuras_analisis_exploratorio_continentes = [figura_partidas_continente, figura_oro_10min, figura_asesinatos_10min, figura_monstruos_grandes_continentes]
        return figuras_analisis_exploratorio_continentes


    def figurasAnalisisLadoMapa(self, datos):
        porcentaje_victorias_lados_mapa = self.calcularPorcentajeVictoriasLadosMapa(datos)
        figura_porcentaje_victorias_general_lado_mapa = self.graficaPorcentajeVictoriasLadosMapa(porcentaje_victorias_lados_mapa, self.nombres_lados_mapa)
        figura_porcentaje_victorias_cuartil_tiempo_lado_azul, figura_porcentaje_victorias_cuartil_tiempo_lado_rojo = self.graficaPorcentajeVictoriasCuartilTiempoSegunLadoMapa(datos)
        figuras_analisis_exploratorio_lado_mapa = [figura_porcentaje_victorias_general_lado_mapa, figura_porcentaje_victorias_cuartil_tiempo_lado_azul, figura_porcentaje_victorias_cuartil_tiempo_lado_rojo]
        return figuras_analisis_exploratorio_lado_mapa