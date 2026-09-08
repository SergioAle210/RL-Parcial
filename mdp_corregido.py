# mdp_corregido.py
# CC3104 - Aprendizaje por Refuerzo | Grupo 1 - Entregable 1.3
#
# MDP corregido: nueva representacion de estado y funcion de transicion mejorada.
#
# Estado corregido:
#   (inventario, dias_vencimiento, demanda_nivel, unidades_en_transito, tendencia_demanda)
# Acciones: [0, 10, 20, 30, 40, 50]

import numpy as np

# ---------------------------------------------------------------------------
# ESPACIO DE ESTADOS
# ---------------------------------------------------------------------------

# inventario: multiplos de 10 en [0, 100] => 11 valores (originalmente 10)
NIVELES_INVENTARIO = list(range(0, 110, 10))

# dias_vencimiento: sin cambio => {1, 7, 14, 30, 60} => 5 valores
DIAS_VENCIMIENTO = [1, 7, 14, 30, 60]

# demanda_nivel: sin cambio => 4 categorias
NIVELES_DEMANDA = ['bajo', 'medio', 'alto', 'critico']

# unidades_en_transito [NUEVO]: pedido realizado el dia anterior, aun en camino.
# Justificacion: el MDP original viola la propiedad de Markov porque el
# inventario efectivo del siguiente paso depende no solo del estado actual
# sino de si hay un pedido pendiente de llegada. Con el mismo estado
# (inv=20, days=14, demand='medio') y accion=0, el inventario siguiente es
# distinto si ayer se pidio 30 unidades (llegan hoy) o si no se pidio nada.
# Formalmente: P(s'|s,a) != P(s'|historia, a) => la propiedad de Markov falla.
# Agregar este campo restaura la propiedad: el estado captura toda la info
# necesaria para predecir la transicion siguiente.
# Valores: {0, 10, 20, 30, 40, 50} => 6 valores
UNIDADES_TRANSITO = [0, 10, 20, 30, 40, 50]

# tendencia_demanda [NUEVO]: direccion del cambio de demanda en el corto plazo.
# Justificacion: la funcion de transicion original congela la demanda
# (new_demand = demand_level), ignorando estacionalidad, picos epidemicos y
# caidas de temporada. Con la tendencia en el estado, la transicion puede
# cambiar el nivel de demanda de forma probabilistica y el agente puede
# anticiparlo y ajustar sus pedidos antes de que ocurra el cambio.
# Valores: {'bajando', 'estable', 'subiendo'} => 3 valores
TENDENCIAS_DEMANDA = ['bajando', 'estable', 'subiendo']

# Conteo total de estados
TOTAL_ESTADOS_ORIGINAL  = 10 * 5 * 4           # 200
TOTAL_ESTADOS_CORREGIDO = 11 * 5 * 4 * 6 * 3   # 3960

# Demanda diaria en unidades por nivel de categoria
DEMAND_MAP = {'bajo': 5, 'medio': 15, 'alto': 25, 'critico': 40}


# ---------------------------------------------------------------------------
# MATRICES DE PROBABILIDADES DE TRANSICION
# ---------------------------------------------------------------------------

# Probabilidad de cambio de nivel de demanda condicionada a la tendencia actual.
# Modela la dinamica real del mercado farmaceutico: una tendencia 'subiendo'
# hace mas probable pasar a niveles de demanda mas altos, y viceversa.
# Esto captura estacionalidad (gripe invernal), picos epidemicos, y caidas
# post-temporada que el modelo original con demanda estatica no puede representar.
#
# PROB_DEMANDA[tendencia][nivel_actual] = {nivel_siguiente: probabilidad}
PROB_DEMANDA = {
    'bajando': {
        'bajo':    {'bajo': 0.80, 'medio': 0.15, 'alto': 0.04, 'critico': 0.01},
        'medio':   {'bajo': 0.40, 'medio': 0.45, 'alto': 0.12, 'critico': 0.03},
        'alto':    {'bajo': 0.10, 'medio': 0.35, 'alto': 0.45, 'critico': 0.10},
        'critico': {'bajo': 0.05, 'medio': 0.20, 'alto': 0.35, 'critico': 0.40},
    },
    'estable': {
        'bajo':    {'bajo': 0.70, 'medio': 0.20, 'alto': 0.08, 'critico': 0.02},
        'medio':   {'bajo': 0.15, 'medio': 0.65, 'alto': 0.15, 'critico': 0.05},
        'alto':    {'bajo': 0.05, 'medio': 0.20, 'alto': 0.60, 'critico': 0.15},
        'critico': {'bajo': 0.02, 'medio': 0.08, 'alto': 0.25, 'critico': 0.65},
    },
    'subiendo': {
        'bajo':    {'bajo': 0.50, 'medio': 0.30, 'alto': 0.15, 'critico': 0.05},
        'medio':   {'bajo': 0.05, 'medio': 0.45, 'alto': 0.35, 'critico': 0.15},
        'alto':    {'bajo': 0.02, 'medio': 0.10, 'alto': 0.48, 'critico': 0.40},
        'critico': {'bajo': 0.01, 'medio': 0.04, 'alto': 0.20, 'critico': 0.75},
    },
}

# Probabilidad de cambio de tendencia (cadena de Markov sobre tendencia).
# La tendencia es persistente (probabilidad alta de mantenerse) pero puede
# cambiar gradualmente, reflejando que los picos o caidas no son instantaneos.
# PROB_TENDENCIA[tendencia_actual] = {tendencia_siguiente: probabilidad}
PROB_TENDENCIA = {
    'bajando':  {'bajando': 0.60, 'estable': 0.35, 'subiendo': 0.05},
    'estable':  {'bajando': 0.20, 'estable': 0.60, 'subiendo': 0.20},
    'subiendo': {'bajando': 0.05, 'estable': 0.35, 'subiendo': 0.60},
}


# ---------------------------------------------------------------------------
# FUNCIONES AUXILIARES
# ---------------------------------------------------------------------------

def discretizar_inventario(inv: float) -> int:
    """Redondea inventario al multiplo de 10 mas cercano, acotado en [0, 100]."""
    return int(min(100, max(0, round(inv / 10.0)))) * 10


def discretizar_dias(dias: int) -> int:
    """Mapea dias al valor mas cercano en {1, 7, 14, 30, 60}."""
    return min(DIAS_VENCIMIENTO, key=lambda x: abs(x - dias))


# ---------------------------------------------------------------------------
# FUNCION DE TRANSICION CORREGIDA
# ---------------------------------------------------------------------------

def transition(state: tuple, action: int):
    """
    Funcion de transicion estocastica del MDP corregido.

    Parametros
    ----------
    state : tuple
        (inventario, dias_vencimiento, demanda_nivel,
         unidades_en_transito, tendencia_demanda)
    action : int
        Unidades a pedir, perteneciente a {0, 10, 20, 30, 40, 50}.

    Retorna
    -------
    next_state : tuple
        Estado siguiente con la misma estructura de 5 variables.
    demanda_no_atendida : int
        Unidades no satisfechas en este paso (stockout explicioto).
        El original las ocultaba con max(0,...); aqui se devuelven
        para que la funcion de recompensa (Grupo 2) pueda penalizarlas.
    """
    inventario, dias_vencimiento, demanda_nivel, en_transito, tendencia = state

    # Los pedidos del dia anterior llegan al inicio del dia de hoy.
    # Esto es el nucleo de la correccion de Markov: el inventario efectivo
    # depende del estado completo (que ahora incluye en_transito).
    inventario_efectivo = min(100, inventario + en_transito)

    # Servir la demanda del dia y registrar stockout explicitamente
    demanda_hoy = DEMAND_MAP[demanda_nivel]
    demanda_no_atendida = max(0, demanda_hoy - inventario_efectivo)
    new_inventory = discretizar_inventario(max(0, inventario_efectivo - demanda_hoy))

    # Decrementar dias hasta vencimiento y discretizar al valor mas cercano
    new_days = discretizar_dias(max(1, dias_vencimiento - 1))

    # Muestrear nivel de demanda siguiente segun la tendencia actual.
    # A diferencia del original (demanda congelada), aqui puede cambiar.
    probs_d = PROB_DEMANDA[tendencia][demanda_nivel]
    new_demand = np.random.choice(list(probs_d.keys()), p=list(probs_d.values()))

    # Muestrear tendencia siguiente
    probs_t = PROB_TENDENCIA[tendencia]
    new_tendencia = np.random.choice(list(probs_t.keys()), p=list(probs_t.values()))

    # El pedido de hoy (action) quedara en transito para manana
    new_transito = action

    next_state = (new_inventory, new_days, new_demand, new_transito, new_tendencia)
    return next_state, int(demanda_no_atendida)


def transition_determinista(state: tuple, action: int):
    """
    Version determinista de la transicion: usa el valor mas probable en cada paso.
    Util para depuracion y para entornos que no soportan estocasticidad directa.
    Misma API que transition(); compatible con Grupos 3, 4 y 6.
    """
    inventario, dias_vencimiento, demanda_nivel, en_transito, tendencia = state

    inventario_efectivo = min(100, inventario + en_transito)
    demanda_hoy = DEMAND_MAP[demanda_nivel]
    demanda_no_atendida = max(0, demanda_hoy - inventario_efectivo)
    new_inventory = discretizar_inventario(max(0, inventario_efectivo - demanda_hoy))
    new_days = discretizar_dias(max(1, dias_vencimiento - 1))

    probs_d = PROB_DEMANDA[tendencia][demanda_nivel]
    new_demand = max(probs_d, key=probs_d.get)

    probs_t = PROB_TENDENCIA[tendencia]
    new_tendencia = max(probs_t, key=probs_t.get)

    next_state = (new_inventory, new_days, new_demand, action, new_tendencia)
    return next_state, int(demanda_no_atendida)

