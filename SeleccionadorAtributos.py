


# Eliminación recursiva de atributos
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SequentialFeatureSelector



class SeleccionadorAtributos:



    NUM_ATRIBUTOS = 6


    def seleccionarAtributos(self, modelo, predictores, y, soporta_RFE):

        if soporta_RFE:
            # Eliminación recursiva de atributos (RFE)
            selector_atributos = RFE(estimator = modelo, n_features_to_select = self.NUM_ATRIBUTOS).fit(predictores, y)
            seleccionados = selector_atributos.get_support()
            atributos_seleccionados = predictores.columns[seleccionados]

        else:
            # Eliminación secuencial (RFE no soportado)
            selector_atributos = SequentialFeatureSelector(estimator = modelo, n_features_to_select = self.NUM_ATRIBUTOS, n_jobs = 1)
            selector_atributos.fit(predictores, y)
            seleccionados = selector_atributos.get_support()
            atributos_seleccionados = predictores.columns[seleccionados]

        return atributos_seleccionados
