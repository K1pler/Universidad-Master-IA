# CRA1 — Introducción: Apuntes

**Fuente:** `CRA1-Introduccion.pdf`  
**Asignatura:** Conocimiento y Razonamiento Aproximado  
**Autores:** Mari Carmen Garrido Carrera, Rodrigo Martínez Béjar, Mercedes Valdés Vela  
**Centro:** DIIC — Universidad de Murcia  
**Máster:** Inteligencia Artificial (2026/2027)  
**Diapositivas:** 23

---

## Visión general del PDF

Presentación del **Tema 1** de la asignatura. Es un **mapa conceptual** (no un tema técnico denso): sitúa qué estudia CRA, cómo se organiza el temario y con qué asignaturas del grado (GII) y del máster (MIA) conecta.

### Estructura del curso (temario CRA)

1. Razonamiento con modelos ontológicos
2. Razonamiento aproximado basado en lógica difusa
3. Aprendizaje de modelos difusos desde datos
4. Razonamiento aproximado con incertidumbre
5. Aprendizaje de modelos con incertidumbre desde datos
6. Conexión con asignaturas de GII
7. Puntos de conexión con asignaturas de MIA

### Eje de la asignatura

Cómo **representar** conocimiento imperfecto y **razonar / aprender** con él, en dos vías principales:

1. **Semántica estructurada** → ontologías  
2. **Imperfección** → difuso (imprecisión) + incertidumbre (probabilidad / evidencia)

---

## Mapa de bloques conceptuales

| # | Bloque | Diapositivas |
|---|--------|--------------|
| 1 | Representación del conocimiento | 4–6 |
| 2 | Ontologías y Web Semántica | 7–8 |
| 3 | Reglas e información imperfecta | 10–11 |
| 4 | Incertidumbre vs imprecisión (y formalismos) | 12–13 |
| 5 | Soft Computing y aprendizaje de modelos | 15–16 |
| 6 | Conexiones con GII y MIA | 18, 20–23 |

> Las diapositivas **1–3, 9, 14, 17 y 19** son portada o índices; no aportan concepto nuevo.

---

## Bloque 1 — Representación del conocimiento

**Diapositivas: 4, 5 y 6**

### Diapositiva 4 — Por qué importa

Un sistema inteligente solo resuelve bien su tarea si los **datos, la información y el conocimiento** están bien representados.

La **Representación del Conocimiento (RC)** es el área de la IA que estudia cómo **codificar** el conocimiento humano para usarlo computacionalmente (inferir, decidir, explicar).

**Idea clave:** sin una buena RC, el resto (algoritmos, sensores, ML) trabaja sobre una base pobre.

### Diapositiva 5 — Esquemas de representación

Un **esquema** fija cómo se escribe y qué significa lo escrito:

- **Sintaxis:** símbolos y reglas para formar estructuras válidas (como la gramática de un lenguaje).
- **Semántica:** qué significa cada estructura en el dominio real (a qué se refiere en el mundo).

**Ejemplo mental:** en lógica, `P ∧ Q` es sintaxis válida; la semántica dice cuándo esa fórmula es verdadera respecto a hechos del dominio.

### Diapositiva 6 — Propiedades deseables

Un buen esquema debería:

1. Ser **suficientemente expresivo** (todo lo necesario del dominio).
2. Permitir **inferencia** eficiente (sacar conocimiento nuevo a partir del existente).
3. Facilitar **añadir conocimiento** nuevo.
4. Ser **claro, natural y modular**.

En la práctica, la asignatura elige esquemas concretos (ontologías, reglas, difuso…) que intentan equilibrar esas cuatro propiedades.

---

## Bloque 2 — Ontologías y Web Semántica

**Diapositivas: 7 y 8**

### Diapositiva 7 — Del esquema abstracto a las ontologías

Hay muchos esquemas de RC: marcos (*frames*), redes semánticas, sistemas de reglas, ontologías, etc.

Hoy el más usado es el basado en **ontologías** y **grafos de conocimiento**, con tecnologías como:

- **RDF** (*Resource Description Framework*): describe recursos y relaciones (sujeto–predicado–objeto).
- **OWL** (*Web Ontology Language*): define clases, propiedades, restricciones e inferencias más ricas encima de RDF.

Eso sostiene la **Web Semántica**:

- No basta con páginas “legibles por humanos”.
- Se anotan datos con **metadatos semánticos** y **vocabularios compartidos** (ontologías).
- Objetivo: que las máquinas **entiendan e interpreten** la información, no solo la indexen como texto.
- Resultado práctico: búsquedas más potentes, servicios inteligentes y sistemas que **deducen** conocimiento nuevo a partir de los datos enlazados.

**En una frase:** las ontologías dan **significado estructurado** a los datos para razonar con ellos.

### Diapositiva 8 — Protégé

**Protégé** es la herramienta estándar para **crear, editar y explorar ontologías** (clases, propiedades, instancias, razonadores).

En CRA suele ser el entorno práctico del bloque ontológico: modelar un dominio y comprobar qué se puede inferir automáticamente.

---

## Bloque 3 — Reglas e información imperfecta

**Diapositivas: 10 y 11**

### Diapositiva 10 — Reglas de producción

Esquema clásico de representación:

**SI** antecedente **ENTONCES** consecuente *(IF–THEN)*

- **Antecedente:** hecho, premisa o situación que se da.
- **Consecuente:** hecho derivado, conclusión o acción.

**Ejemplos del PDF:**

- SI es un animal con garras → es carnívoro  
- SI el semáforo está en rojo → parar  

**Problema:** a veces los elementos de la regla **no se conocen con exactitud**. Entonces las reglas incorporan mecanismos para tratar esa **información imperfecta**. Ahí entra el razonamiento aproximado de la asignatura.

### Diapositiva 11 — Tipos de imperfección

La información puede fallar de varias formas:

| Tipo | Idea |
|------|------|
| **Incompleta** | Faltan datos necesarios |
| **Contradictoria** | Hay afirmaciones incompatibles |
| **Redundante** | Hay información repetida o sobrante |
| **Vaga / imprecisa** | Lenguaje natural (“alto”, “joven”, “cerca”) |
| **Poco fiable (incertidumbre)** | Sensores imperfectos o juicios subjetivos |

**Punto clave:** “imperfecta” **no es un solo problema**. Vaguedad e incertidumbre se tratan distinto (bloque 4).

---

## Bloque 4 — Incertidumbre vs imprecisión (y formalismos)

**Diapositivas: 12 y 13**

### Diapositiva 12 — Dos problemas distintos

Están relacionados, pero **no son lo mismo**:

| | **Incertidumbre** | **Imprecisión** |
|---|-------------------|-----------------|
| Qué es | Duda del observador sobre si una afirmación es **verdadera** | La afirmación no es solo V/F; tiene **grado** de verdad |
| Ejemplo | Radar da lat/long **precisa**, pero puede estar **mal** | “Es una persona **joven**” (frontera gradual) |
| Formalismos (asignatura) | Modelos probabilísticos; Dempster–Shafer / teoría de la evidencia → **Tema 5** | **Lógica difusa** → **Tema 3** |

**Regla mental:**

- **Incertidumbre** → “¿me fío de este dato?”
- **Imprecisión** → “¿hasta qué punto encaja esta etiqueta?”

### Diapositiva 13 — Lógica difusa en acción

Muestra un **sistema de reglas difusas** para calcular la **propina** (ejemplo clásico):

- Entradas lingüísticas (p. ej. calidad del servicio/comida: malo, regular, bueno).
- Reglas del tipo SI… ENTONCES…, pero con conjuntos **difusos**.
- Salida: propina baja/media/alta, también gradual.

Sirve para ver que el razonamiento aproximado con difuso permite decidir con etiquetas del lenguaje natural, sin forzar umbrales rígidos.

---

## Bloque 5 — Soft Computing y aprendizaje de modelos

**Diapositivas: 15 y 16**

### Diapositiva 15 — De dónde viene la imperfección y qué es Soft Computing

**Orígenes típicos de información imperfecta:**

1. **Experto humano**  
   No conoce el mundo de forma absoluta; además el lenguaje natural es vago (“alto”, “rápido”, “cerca”).

2. **Sensores**  
   Ruido, fallos, condiciones ambientales (p. ej. meteorología) → datos imprecisos o poco fiables.

**Soft Computing** = conjunto de técnicas pensadas para:

- Manejar información imperfecta.
- Emular el razonamiento humano en situaciones ambiguas o complejas.
- Tolerar imprecisión, incertidumbre y ruido.
- Combinar **conocimiento experto** con **aprendizaje desde datos**.

**Técnicas citadas:** redes neuronales, algoritmos genéticos, clustering, e **híbridos** (neuro-difuso, árboles de decisión difusos, clustering difuso, …).

**Enlace con CRA:** no solo *representar* conocimiento imperfecto, sino también *aprender* modelos difusos o con incertidumbre a partir de datos (bloques 3–5 del temario).

### Diapositiva 16 — Red neuro-difusa

Ilustra un sistema **neuro-difuso**: mezcla

- **parte difusa** → reglas e interpretabilidad lingüística, y  
- **parte neuronal** → ajuste/aprendizaje de parámetros desde datos.

**Idea:** el modelo sigue siendo (más) interpretable que una red “caja negra” pura, pero puede entrenarse.

---

## Bloque 6 — Conexiones con GII y MIA

**Diapositivas: 18, 20, 21, 22 y 23**

### Diapositiva 18 — Enlace con el Grado (GII)

CRA **reutiliza**, **desarrolla** y **aplica** contenidos del grado:

| Rol | Contenidos / asignaturas |
|-----|--------------------------|
| Se **usan** | Lógica proposicional (conectivas, tablas de verdad, consecuencia); lógica de predicados (cuantificadores) |
| Se **desarrollan** | Representación del conocimiento; sistemas basados en reglas; gestión de la incertidumbre |
| Se **aplican** | Árboles de decisión, clustering, redes neuronales |
| Asignaturas GII-UMU | **FLI** (1º), **SI** (3º), **DESIN** (4º Computación), **AC** (4º Computación) |

**Mensaje:** CRA no parte de cero; apoya y profundiza lo visto en el grado hacia conocimiento imperfecto y ontologías.

### Diapositiva 20 — MIA: Machine Learning y Computación bio-inspirada

- **Machine Learning:** extraer conocimiento útil / modelos inteligibles desde datos. Soft Computing comparte ese objetivo y fases similares.
- **Computación bio-inspirada:** algoritmos evolutivos (a menudo multiobjetivo) para generar/optimizar reglas o modelos difusos **interpretables**.

### Diapositiva 21 — Extensiones de ML (refuerzo + difuso)

El **aprendizaje por refuerzo** puede combinarse con sistemas de inferencia difusa (aprender políticas/reglas con etiquetas lingüísticas; acelerar DRL con reglas difusas). La diapositiva ilustra el esquema clásico de RL.

### Diapositiva 22 — Visión Artificial

- **Ontologías:** dan semántica al reconocimiento (no solo “qué hay”, sino **contexto y relaciones** entre objetos).
- **Modelos difusos:** útiles cuando las fronteras no son nítidas (color, iluminación, detección/segmentación con reglas difusas).

### Diapositiva 23 — ML explicable e IA y sociedad

- CRA (ontologías, reglas difusas) se presenta como **antítesis de la caja negra**: modelos más interpretables y alineados con la **explicabilidad**.
- En **IA y sociedad**, la explicabilidad es un principio de diseño; representación del conocimiento + razonamiento aproximado van en esa línea.

---

## Resumen final

| # | Bloque | Diapos |
|---|--------|--------|
| 1 | Representación del conocimiento | 4–6 |
| 2 | Ontologías y Web Semántica | 7–8 |
| 3 | Reglas e información imperfecta | 10–11 |
| 4 | Incertidumbre vs imprecisión | 12–13 |
| 5 | Soft Computing y aprendizaje | 15–16 |
| 6 | Conexiones GII / MIA | 18, 20–23 |

**Siguiente paso natural de estudio:** material de `Tema_1` (fundamentos ontológicos).
