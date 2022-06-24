


import pandas as pd

# Importancia por permutación de atributos
from sklearn.inspection import permutation_importance

# Para el gráfico con la importancia de los atributos
import plotly.express as px



class ImportanciaPermutacion:



    NUM_REPETICIONES = 20
    ESTADO_ALEATORIO = 33
    POSX_TITULO_GRAFICA = 0.5
    POSY_TITULO_GRAFICA = 0.9
    nombre_importancia = 'Importancia'
    nombre_importancia_media = 'importances_mean'
    etiqueta_x = 'x'
    etiqueta_y = 'y'
    etiqueta_atributos = 'Atributos'
    secuencia_colores = px.colors.qualitative.Set2



    def importanciaAtributos(self, modelo, predictores, y, atributos):
        importancia_permutacion = permutation_importance(modelo, predictores[atributos], y, n_repeats = self.NUM_REPETICIONES, random_state = self.ESTADO_ALEATORIO, n_jobs = 1)
        importancia_atributos = pd.DataFrame(importancia_permutacion[self.nombre_importancia_media], columns = [self.nombre_importancia], index = atributos)
        importancia_atributos.sort_values(by = [self.nombre_importancia], ascending = False, inplace = True)
        return importancia_atributos
    

    def graficaImportanciaAtributos(self, importancia_atributos, titulo):
        figura = px.bar(x = importancia_atributos.index, y = importancia_atributos[self.nombre_importancia], 
                        title = titulo,
                        color = importancia_atributos.index, color_discrete_sequence = self.secuencia_colores,
                        labels = {
                                    self.etiqueta_x: self.etiqueta_atributos,
                                    self.etiqueta_y: self.nombre_importancia
                                })
        figura.update_layout(title_x = self.POSX_TITULO_GRAFICA, title_y = self.POSY_TITULO_GRAFICA)
        figura.update_traces(showlegend = False)
        return figura