

import re

# Manejador del web driver Firefox
from webdriver_manager.firefox import GeckoDriverManager

# Selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service



class WebScrappingContinentes:



    # Direcciones y opciones para Selenium
    opcion_navegador_no_visible = '--headless'
    opcion_usar_carpeta_temporal = '--disable-dev-shm-usage'
    opcion_desactivar_grafica = '--disable-gpu'
    enlace_base_lol_fandom_wiki = 'https://lol.fandom.com/wiki/'
    enlace_base_wikipedia_lol = 'https://en.wikipedia.org/wiki/List_of_League_of_Legends_leagues_and_tournaments'
    xpath_wikipedia_tabla_base_inicio = '//*[@id="mw-content-text"]/div[1]/table['
    xpath_wikipedia_tabla_base_final = "]/tbody"
    tag_fila_tabla = 'tr'
    tag_encabecado_tabla = 'th'
    comienzo_ruta_relativa = './'
    apertura_parentesis = '('
    cierre_parentesis = ')'
    tag_segunda_celda = 'td[1]'
    tag_tercera_celda = 'td[2]'
    expresion_cualquier_caracter = '[^A-Za-z0-9]+'
    separador = '_'
    xpath_lol_fandom_wiki = '//*[@id="infoboxTournament"]/tbody//td[2]/span/span'
    xpath_lol_fandom_wiki_buscar_liga_opcion1 = '//*[@id="infoboxTeam"]/tbody/tr[8]/td[2]/span/span'
    xpath_lol_fandom_wiki_buscar_liga_opcion2 = '//*[@id="infoboxTeam"]/tbody/tr[9]/td[2]/span/span'
    xpath_lol_fandom_wiki_buscar_liga_opcion3 = '//*[@id="infoboxTeam"]/tbody/tr[7]/td[2]/span/span[2]'
    xpath_lol_fandom_wiki_buscar_liga_opcion4 = '//*[@id="infoboxTeam"]/tbody/tr[10]/td[2]/span/span'
    direccion_temporada_partidas = '/2021_Season/Spring_Season'
    contenido_elemento_texto = 'textContent'

    # Mapeo regiones-continentes
    regiones_america_norte = ['North America']    
    regiones_america_sur = ['Brazil', 'Chile', 'Latin America']
    regiones_europa = ['Europe', 'Turkey', 'UnitedKingdomIreland']
    regiones_asia = ['China', 'Japan', 'Korea', 'SouthKorea', 'Vietnam']
    regiones_oceania = ['Oceania', 'PCS']
    continentes_nombres = ['América del Norte', 'América del Sur', 'Europa', 'Asia', 'Oceania']
    nombre_atributo_liga = 'league'
    nombre_atributo_nombre_equipo = 'playerid'
    nombre_atributo_continente = 'continent'



    def __init__(self, datos):
        self.datos = datos
        self.ligas_localizadas = []
        self.ligas_sin_localizar = []

        # Opciones para usar menos memoria
        opciones = Options()
        opciones.add_argument(self.opcion_navegador_no_visible)
        opciones.add_argument(self.opcion_desactivar_grafica)
        opciones.add_argument(self.opcion_usar_carpeta_temporal)

        # Detectar la versión y descargar automáticamente el controlador web en caso de que fuera necesario
        servicio = Service(GeckoDriverManager().install())
        self.webDriver = webdriver.Firefox(options = opciones, service = servicio)



    def buscarLolFandomWiki(self):
        for liga in self.datos[self.nombre_atributo_liga].unique():
            try:
                enlace = self.enlace_base_lol_fandom_wiki + liga + self.direccion_temporada_partidas
                self.webDriver.get(enlace)
                region = self.webDriver.find_element(By.XPATH, self.xpath_lol_fandom_wiki)
                self.ligas_localizadas.append([liga, region.get_attribute(self.contenido_elemento_texto)])

            except NoSuchElementException:
                self.ligas_sin_localizar.append(liga)


    def buscarLigaEnTabla(self, liga, num_tabla, tiene_columnas_intermedia):
            
        xpath_tabla = self.xpath_wikipedia_tabla_base_inicio + str(num_tabla) + self.xpath_wikipedia_tabla_base_final
        cuerpo_tabla = self.webDriver.find_element(By.XPATH, xpath_tabla)
        filas_tabla = cuerpo_tabla.find_elements(By.TAG_NAME, self.tag_fila_tabla)
            
        liga_encontrada = False    
        for fila in filas_tabla:
            nombre_liga = ''
            palabras = None
            elemento_liga = fila.find_element(By.XPATH, self.comienzo_ruta_relativa + self.tag_encabecado_tabla)
            nombre_elemento_liga = elemento_liga.get_attribute(self.contenido_elemento_texto)

            # Si tiene las iniciales entre paréntesis, coger las iniciales
            if '(' in nombre_elemento_liga:
                nombre_liga = nombre_elemento_liga[nombre_elemento_liga.find(self.apertura_parentesis)+1:nombre_elemento_liga.find(self.cierre_parentesis)]
            else:
                # Si no, crear las iniciales
                palabras = nombre_elemento_liga.split()
                for palabra in palabras:
                    nombre_liga += palabra[0]        
                        
            # También se incluye si el nombre de la liga se escribe por la primera palabra de su nombre
            if liga == nombre_liga or (palabras != None and liga == palabras[0]):
                    
                if tiene_columnas_intermedia:
                    region = fila.find_element(By.XPATH, self.comienzo_ruta_relativa + self.tag_tercera_celda)
                else:
                    region = fila.find_element(By.XPATH, self.comienzo_ruta_relativa + self.tag_segunda_celda)
                        
                nombre_region = region.get_attribute(self.contenido_elemento_texto)
                    
                # Eliminar caracteres especiales
                self.ligas_localizadas.append([liga, re.sub(self.expresion_cualquier_caracter, '', nombre_region)])
                self.ligas_sin_localizar.remove(liga)
                liga_encontrada = True
        
        return liga_encontrada


    def buscarLigaEnTablas(self, liga):
        liga_encontrada = self.buscarLigaEnTabla(liga, 3, False)
        if not liga_encontrada:
            liga_encontrada = self.buscarLigaEnTabla(liga, 4, False)
            if not liga_encontrada:
                liga_encontrada  = self.buscarLigaEnTabla(liga, 5, False)
                if not liga_encontrada:
                    liga_encontrada = self.buscarLigaEnTabla(liga, 7, False)
                    if not liga_encontrada:
                        self.buscarLigaEnTabla(liga, 8, False)                   


    def buscarWikipediaLol(self):
        self.webDriver.get(self.enlace_base_wikipedia_lol)

        for liga in self.ligas_sin_localizar:
            self.buscarLigaEnTablas(liga)


    def regionPorXpath(self, xpath):
        try:
            region = self.webDriver.find_element(By.XPATH, xpath)
            nombre_region = region.get_attribute(self.contenido_elemento_texto)
            if nombre_region != '':
                return [True, nombre_region]
            else:
                return [False, None]
                    
        except NoSuchElementException:
            return [False, None]


    def buscarRegion(self):
        region_recogida, region = self.regionPorXpath(self.xpath_lol_fandom_wiki_buscar_liga_opcion1)
        if not region_recogida:
            region_recogida, region = self.regionPorXpath(self.xpath_lol_fandom_wiki_buscar_liga_opcion2)
            if not region_recogida:
                region_recogida, region = self.regionPorXpath(self.xpath_lol_fandom_wiki_buscar_liga_opcion3)
                if not region_recogida:
                    region_recogida, region = self.regionPorXpath(self.xpath_lol_fandom_wiki_buscar_liga_opcion4)

        return region_recogida, region


    def buscarLigaPorEquipo(self):
        equipos_en_ligas = []
        for liga in self.ligas_sin_localizar:  
            equipos_ligas = self.datos[self.datos[self.nombre_atributo_liga] == liga][self.nombre_atributo_nombre_equipo].values
            equipos_en_ligas.append([equipos_ligas, liga])

        for equipos in equipos_en_ligas:

            region_recogida = False
            i = 0
            while not region_recogida and i < len(equipos[0]):
                equipo_ruta = equipos[0][i].replace(' ', self.separador)
                enlace = self.enlace_base_lol_fandom_wiki + equipo_ruta
                self.webDriver.get(enlace)     
                region_recogida, region = self.buscarRegion()
                i += 1
                        
            if region_recogida:
                self.ligas_localizadas.append([equipos[1], region])
                self.ligas_sin_localizar.remove(equipos[1])


    def mapearContinentesDesdeLiga(self):
        mapeo_continentes = {}

        for liga in self.ligas_localizadas:
            if liga[1] in self.regiones_america_norte:
                mapeo_continentes[liga[0]] = self.continentes_nombres[0]
            elif liga[1] in self.regiones_america_sur:
                mapeo_continentes[liga[0]] = self.continentes_nombres[1]
            elif liga[1] in self.regiones_europa:
                mapeo_continentes[liga[0]] = self.continentes_nombres[2]        
            elif liga[1] in self.regiones_asia:
                mapeo_continentes[liga[0]] = self.continentes_nombres[3]
            else:
                mapeo_continentes[liga[0]] = self.continentes_nombres[4]
            
        self.datos[self.nombre_atributo_continente] = self.datos[self.nombre_atributo_liga].map(mapeo_continentes)


    def hayLigasSinLocalizar(self):
        return len(self.ligas_sin_localizar) > 0


    def crearAtributoContinentes(self):
        num_observaciones_originales = self.datos.shape[0]

        # Buscar a qué región pertenecen las ligas en la web de LoL de Fandom Wiki
        self.buscarLolFandomWiki()

        # Si han quedado ligas sin localizar se procede de forma similar con la web del LoL de Wikipedia
        if self.hayLigasSinLocalizar():
            self.buscarWikipediaLol()

            # Si han quedado ligas de las que no se sabe la región, se va a buscar la liga a través de uno de los equipos que participan en la misma
            if self.hayLigasSinLocalizar():
                self.buscarLigaPorEquipo()

                # Si no aún así no se ha encontrado, eliminar las filas que tengan esas ligas
                if self.hayLigasSinLocalizar():
                    for liga in self.ligas_sin_localizar:
                        datos_liga = self.datos[self.datos[self.nombre_atributo_liga] == liga]
                        self.datos.drop(datos_liga.index, inplace = True)
                    self.datos.reset_index(drop = True)

        self.mapearContinentesDesdeLiga()

        num_observaciones_no_encontradas = num_observaciones_originales - self.datos.shape[0]

        return self.datos, num_observaciones_no_encontradas



