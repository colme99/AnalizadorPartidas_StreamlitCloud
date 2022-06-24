


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Clustering
from sklearn_extra.cluster import KMedoids
from yellowbrick.cluster import KElbowVisualizer
import gower

# Gráficos
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

# Clasificador para predecir el cluster al que pertenece cada partida
from sklearn.ensemble import RandomForestClassifier

# Importancia por permutación de atributos
from ImportanciaPermutacion import ImportanciaPermutacion



class Clustering:



    ESTADO_ALEATORIO = 33
    RANGO_VALORES_K = (2,8)
    POSX_TITULO_CENTRADO = 0.5
    POSY_TITULO_CENTRADO = 0.9
    PROPORCION_TEST_ENTRENAMIENTO = 0.2
    UMBRAL_PORCENTAJE_VICTORIA_DERROTA = 50
    GROSOR_LINEA_PUNTOS = 2
    ANCHURA_GRAFICO = 700
    ALTURA_GRAFICO = 500
    SEPARACION_BARRAS = 0.2
    modo_metrica_precalculada = 'precomputed'
    metodo_pam = 'pam'
    metrica_silhouette = 'silhouette'
    metrica_distorsion = 'distortion'
    nombre_atributo_grupos = 'cluster'
    nombre_media_importancia_por_permutacion = 'importances_mean'
    nombre_desviacion_importancia_por_permutacion = 'importances_std'
    nombre_media = 'mean'
    nombre_desviacion = 'std'
    nombre_valor = 'value'
    etiqueta_importancia = 'Importancia'
    etiqueta_cluster = 'Cluster'
    etiqueta_clusters = 'Clusters'
    etiqueta_atributos = 'Atributos'
    etiqueta_color = 'color'
    etiqueta_porcentaje = 'Porcentaje'
    etiqueta_num_clusters = 'Nº de clusters'
    etiqueta_x = 'x'
    etiqueta_y = 'y'
    modo_linea_puntos = 'dash'
    color_linea_puntos = 'black'
    titulo_grafica_comparar_clusters_final = ' en cada cluster'
    titulo_grafica_cantidad_partidas = 'Cantidad de partidas'
    titulo_grafica_porcentaje_victorias = 'Porcentaje de victorias'
    titulo_grafica_importancia_atributos_inicio = 'Atributos en los que más difieren los clusters (clustering con '
    titulo_grafica_importancia_atributos_final = ' clusters)'
    titulo_grafica_media_mejores_atributos_inicio = 'Media de '
    titulo_grafica_moda_mejores_atributos_inicio = 'Moda de '
    titulo_grafica_media_moda_mejores_atributos_final = ' en los clusters'
    titulo_grafica_histograma_mejores_atributos_inicio = 'Histograma de '
    titulo_grafica_mejores_atributos_combinados = ' respecto a '
    titulo_grafica_metodo_codo_inicio = 'Método del codo con métrica '
    titulo_grafica_metodo_codo_fin = ' para el clustering con KMedoides'
    secuencia_colores = px.colors.qualitative.Set2
    color_negro = 'black'



    def __init__(self, datos, nombre_atributo_resultado, atributos_categoricos):
        self.grupos = []
        self.datos_con_grupos = []
        self.datos = datos
        self.atributos_categoricos = atributos_categoricos
        self.nombre_atributo_resultado = nombre_atributo_resultado

        # Calcular la matriz de distancias
        self.matriz_distancias = gower.gower_matrix(self.datos)



    def visualizarMetodoCodo(self, clustering, metodo_metrica):
        figura = plt.figure()
        if metodo_metrica == self.metrica_silhouette:
            nombre_metrica = metodo_metrica.capitalize()
        else:
            nombre_metrica = self.metrica_distorsion
        visualizador_medoto_codo = KElbowVisualizer(clustering, k = self.RANGO_VALORES_K, metric = metodo_metrica, timings = False)
        visualizador_medoto_codo.fit(self.matriz_distancias)   

        plt.suptitle(self.titulo_grafica_metodo_codo_inicio + nombre_metrica + self.titulo_grafica_metodo_codo_fin, fontsize = 14)
        plt.ylabel(nombre_metrica.capitalize(), fontsize = 12)
        plt.xlabel(self.etiqueta_num_clusters, fontsize = 12)

        return visualizador_medoto_codo.elbow_value_, figura
        

    def valoresSeleccionadosK(self):
        clustering = KMedoids(random_state = self.ESTADO_ALEATORIO, metric = self.modo_metrica_precalculada, method = self.metodo_pam)
        clustering.fit(self.matriz_distancias)  
        mejor_k_silhouette, figura_metodo_codo_silhouette = self.visualizarMetodoCodo(clustering, self.metrica_silhouette)
        mejor_k_distorsion, figura_metodo_codo_distorsion = self.visualizarMetodoCodo(clustering, self.metrica_distorsion)
        return mejor_k_silhouette, mejor_k_distorsion, figura_metodo_codo_silhouette, figura_metodo_codo_distorsion


    def datosGrupo(self, datos, num_grupo):
        return datos[datos[self.nombre_atributo_grupos] == num_grupo]


    def listaGrupos(self, N_grupos):
        grupos = []
        for i in range(N_grupos):
            grupos.append(self.datosGrupo(self.datos_con_grupos, i))
        return grupos


    def nombresClusters(self):
        self.nombres_clusters = []
        self.mapeo_nombres_clusters = {}
        for i in range(len(self.grupos)):
            self.nombres_clusters.append(self.etiqueta_cluster + ' ' + str(i+1))
            self.mapeo_nombres_clusters[str(i)] = self.nombres_clusters[i]
    

    def hacerClustering(self, valor_k):
        self.valor_k = valor_k
        clustering = KMedoids(n_clusters = valor_k, random_state = self.ESTADO_ALEATORIO, metric = self.modo_metrica_precalculada, method = self.metodo_pam).fit(self.matriz_distancias)
        self.datos_con_grupos = self.datos.copy()
        self.datos_con_grupos[self.nombre_atributo_grupos] = clustering.labels_
        self.grupos = self.listaGrupos(valor_k)
        self.nombresClusters()


    def cantidadPartidasPorGrupo(self, grupos):
        cantidad_partidas = []
        for grupo in grupos:
            cantidad_partidas.append(grupo.shape[0])
        return cantidad_partidas


    def compararClusters(self, datos, nombres_clusters, etiqueta_y, es_porcentaje_victorias):    
        figura = px.bar(x = nombres_clusters, y = datos, title = etiqueta_y + self.titulo_grafica_comparar_clusters_final,
                    color = nombres_clusters, color_discrete_sequence = self.secuencia_colores, 
                    labels = {
                                    self.etiqueta_x: self.etiqueta_clusters,
                                    self.etiqueta_y: etiqueta_y
                                })
        if es_porcentaje_victorias:
            figura.add_hline(y = self.UMBRAL_PORCENTAJE_VICTORIA_DERROTA, line_width = self.GROSOR_LINEA_PUNTOS, line_dash = self.modo_linea_puntos, line_color = self.color_linea_puntos)
        figura.update_layout(title_x = self.POSX_TITULO_CENTRADO, title_y = self.POSY_TITULO_CENTRADO)
        figura.update_traces(showlegend = False)

        colores = []
        for datos_barra in figura.data:
            colores.append(datos_barra.marker.color)
        return colores, figura


    def graficaCantidadPartidasPorCluster(self):
        cantidad_partidas_grupos = self.cantidadPartidasPorGrupo(self.grupos)
        self.colores_clusters, figura_cantidad_partidas_cluster = self.compararClusters(cantidad_partidas_grupos, self.nombres_clusters, self.titulo_grafica_cantidad_partidas, False)
        return figura_cantidad_partidas_cluster
    

    def calcularPorcentajeVictoria(self, datos):
        return datos[datos[self.nombre_atributo_resultado] == 1].shape[0] / datos.shape[0] * 100


    def porcentajesVictoriaGrupos(self, grupos):    
        porcentajes_victoria = []
        for grupo in grupos:
            porcentajes_victoria.append(self.calcularPorcentajeVictoria(grupo))
        return porcentajes_victoria


    def graficaPorcentajeVictoriasPorCluster(self):
        porcentajes_victorias = self.porcentajesVictoriaGrupos(self.grupos)
        colores, figura_porcentaje_victorias_clusters = self.compararClusters(porcentajes_victorias, self.nombres_clusters, self.titulo_grafica_porcentaje_victorias, True)
        return figura_porcentaje_victorias_clusters


    def graficaImportanciaAtributosClusters(self, nombres_atributos, importancia_atributos, etiqueta_y):
        valor_color_atributos = [str(i) for i in range(self.datos.shape[1])]     
        figura = px.bar(y = importancia_atributos[:5], x = nombres_atributos[:5], title = self.titulo_grafica_importancia_atributos_inicio + str(len(self.grupos)) + self.titulo_grafica_importancia_atributos_final,
                        color = valor_color_atributos[:5], color_discrete_sequence = self.secuencia_colores, 
                        labels = {
                                    self.etiqueta_y: etiqueta_y,
                                    self.etiqueta_x: self.etiqueta_atributos
                                })
        figura.update_layout(title_x = self.POSX_TITULO_CENTRADO, title_y = self.POSY_TITULO_CENTRADO)
        figura.update_traces(showlegend = False)
        return figura


    def importanciaAtributos(self, modelo, predictores, y):
        # Importancia por permutación de atributos
        calculo_importancia = ImportanciaPermutacion()
        self.importancia_atributos = calculo_importancia.importanciaAtributos(modelo, predictores, y, predictores.columns)
        titulo = self.titulo_grafica_importancia_atributos_inicio + str(len(self.grupos)) + self.titulo_grafica_importancia_atributos_final
        figura_importancia_atributos = calculo_importancia.graficaImportanciaAtributos(self.importancia_atributos[:5], titulo)
        return figura_importancia_atributos


    def graficaAtributosDiferentesEntreClusters(self):

        # Splits
        datos_con_grupos_predecir_grupo = self.datos_con_grupos.copy()
        datos_con_grupos_predecir_grupo[self.nombre_atributo_grupos] = datos_con_grupos_predecir_grupo[self.nombre_atributo_grupos].astype(str)
        datos_predictores = self.datos_con_grupos.drop(labels = self.nombre_atributo_grupos, axis = 1)
        predictores_entrenar, predictores_prueba, y_entrenar, y_prueba = train_test_split(datos_predictores, datos_con_grupos_predecir_grupo[self.nombre_atributo_grupos], test_size = self.PROPORCION_TEST_ENTRENAMIENTO, random_state = self.ESTADO_ALEATORIO)
       
        # Clasificación para predecir el cluster al que pertenece cada partida (y posteriormente ver la importancia de sus atributos)
        modelo = RandomForestClassifier(n_estimators = 20, random_state = 33, n_jobs = 1)
        modelo.fit(predictores_entrenar, y_entrenar)
        precision = modelo.score(predictores_prueba, y_prueba)

        return self.importanciaAtributos(modelo, predictores_entrenar, y_entrenar), precision


    def graficaMediaMejoresAtributos(self, columna, columna_grupos, columna_grupos_std):
        figura = px.scatter(x = self.nombres_clusters, y = columna_grupos, title = self.titulo_grafica_media_mejores_atributos_inicio + columna + self.titulo_grafica_media_moda_mejores_atributos_final,
                        color = self.nombres_clusters, color_discrete_sequence = self.secuencia_colores,
                        error_y = columna_grupos_std,
                        labels = {
                                    self.etiqueta_x: self.etiqueta_clusters,
                                    self.etiqueta_y: columna
                                })
        
        figura.update_layout(title_x = self.POSX_TITULO_CENTRADO, title_y = self.POSY_TITULO_CENTRADO)
        figura.update(layout_coloraxis_showscale = False)
        figura.update_traces(showlegend = False, error_y_thickness = 0.5, error_y_color = self.color_negro,
                            marker = dict(size = 12, line = dict(width = 1, color = self.color_negro)))

        return figura


    def graficaModaMejoresAtributos(self, columna, columna_grupos):
        figura = px.scatter(x = self.nombres_clusters, y = [columna_grupos], title = self.titulo_grafica_moda_mejores_atributos_inicio + columna + self.titulo_grafica_media_moda_mejores_atributos_final,
                        color = self.nombres_clusters, color_discrete_sequence = self.secuencia_colores,
                        labels = {
                                    self.etiqueta_x: self.etiqueta_clusters,
                                    self.nombre_valor: columna
                                })
        
        figura.update_layout(title_x = self.POSX_TITULO_CENTRADO, title_y = self.POSY_TITULO_CENTRADO)
        figura.update(layout_coloraxis_showscale = False)
        figura.update_traces(showlegend = False, marker = dict(size = 12, line = dict(width = 1, color = self.color_negro)))
        return figura


    def graficaMediaModaMejoresAtributosClusters(self):
        figuras_media_moda_mejores_atributos = []
        for columna in self.importancia_atributos[:5].index:

            columna_grupos = []
            if not columna in self.atributos_categoricos:
                # Mostrar media y desviación típica
                columna_grupos_std = []
                for grupo in self.grupos:
                    columna_grupos.append(grupo[columna].mean())
                    columna_grupos_std.append(grupo[columna].std())
                figura_media_moda_mejores_atributos = self.graficaMediaMejoresAtributos(columna, columna_grupos, columna_grupos_std)
            
            else:
                # Mostrar moda
                for grupo in self.grupos:
                    columna_grupos.append(grupo[columna].mode().iloc[0])

                figura_media_moda_mejores_atributos = self.graficaModaMejoresAtributos(columna, columna_grupos)

            figuras_media_moda_mejores_atributos.append(figura_media_moda_mejores_atributos)

        return figuras_media_moda_mejores_atributos
    

    def graficaHistogramasMejoresAtributosClusters(self):
        figuras_histogramas_mejores_atributos = []
        for columna in self.importancia_atributos[:5].index:
            N_grupos = len(self.grupos)
            figura = make_subplots(rows = 1, cols = N_grupos, shared_xaxes = True, shared_yaxes = True)
            
            for i in range(N_grupos):
                figura.add_trace(
                    go.Histogram(x = self.grupos[i][columna], name = self.nombres_clusters[i], marker_color = self.colores_clusters[i], histnorm = 'percent'),
                    row = 1, col = (i+1),
                )

            figura.update_yaxes(title_text = self.etiqueta_porcentaje, row = 1, col = 1)

            figura.update_layout(title_x = self.POSX_TITULO_CENTRADO, title_y = self.POSY_TITULO_CENTRADO, title_text = self.titulo_grafica_histograma_mejores_atributos_inicio + columna + ' (%)',
                                    autosize = False, width = self.ANCHURA_GRAFICO, height = self.ALTURA_GRAFICO, bargap = self.SEPARACION_BARRAS)

            figuras_histogramas_mejores_atributos.append(figura)

        return figuras_histogramas_mejores_atributos


    def actualizacionesGraficaCombinadaMejoresAtributos(self, figura):
        figura.for_each_trace(lambda traza: traza.update(name = self.mapeo_nombres_clusters[traza.name],
                                legendgroup = self.mapeo_nombres_clusters[traza.name],
                                hovertemplate = traza.hovertemplate.replace(traza.name, self.mapeo_nombres_clusters[traza.name])
                                )
                            )
        figura.update_layout(title_x = self.POSX_TITULO_CENTRADO, title_y = self.POSY_TITULO_CENTRADO)
        return figura


    def graficaCombinadaMejoresAtributosVariableNumerica(self):
        figura = px.scatter(self.datos_con_grupos, x = self.importancia_atributos.index[1], y = self.importancia_atributos.index[0], 
                            title = self.importancia_atributos.index[0] + self.titulo_grafica_mejores_atributos_combinados + self.importancia_atributos.index[1], 
                            color = self.datos_con_grupos[self.nombre_atributo_grupos].apply(str),
                            labels = {
                                self.etiqueta_color: self.etiqueta_clusters
                            })
        return figura


    def graficaCombinadaMejoresAtributosVariableCategorica(self):
        figura = px.violin(self.datos_con_grupos, x = self.importancia_atributos.index[1], y = self.importancia_atributos.index[0], color = self.nombre_atributo_grupos,
                            title = self.importancia_atributos.index[0] + self.titulo_grafica_mejores_atributos_combinados + self.importancia_atributos.index[1], 
                            labels = {
                                self.nombre_atributo_grupos: self.etiqueta_clusters
                            })
        return figura


    def graficaMejoresAtributosCombinados(self):
        if self.importancia_atributos.index[0] in self.atributos_categoricos and self.importancia_atributos.index[1] in self.atributos_categoricos:
            figura = self.graficaCombinadaMejoresAtributosVariableCategorica()
        else:
            figura = self.graficaCombinadaMejoresAtributosVariableNumerica()
        return self.actualizacionesGraficaCombinadaMejoresAtributos(figura)


    def getFigurasClustering(self, valor_k):
        self.hacerClustering(valor_k)
        figura_cantidad_partidas_clusters = self.graficaCantidadPartidasPorCluster()
        figura_porcentaje_victorias_clusters = self.graficaPorcentajeVictoriasPorCluster()
        figura_atributos_diferentes_clusters, precision_clasificacion_clusters = self.graficaAtributosDiferentesEntreClusters()     
        figuras_media_mejores_atributos = self.graficaMediaModaMejoresAtributosClusters()
        figuras_histogramas_mejores_atributos = self.graficaHistogramasMejoresAtributosClusters()
        figura_mejores_atributos_combinados = self.graficaMejoresAtributosCombinados()
        return [figura_cantidad_partidas_clusters, figura_porcentaje_victorias_clusters, figura_atributos_diferentes_clusters, figuras_media_mejores_atributos, figuras_histogramas_mejores_atributos, figura_mejores_atributos_combinados], precision_clasificacion_clusters



