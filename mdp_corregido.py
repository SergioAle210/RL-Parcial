# MDP de gestion de inventario farmaceutico.

import numpy as np


DIAS_VENCIMIENTO = [1, 7, 14, 30, 60]
NIVELES_DEMANDA = ['bajo', 'medio', 'alto', 'crítico']
DEMAND_MAP = {'bajo': 5, 'medio': 15, 'alto': 25, 'crítico': 40}

# El pedido pendiente forma parte del estado para preservar la propiedad de Markov.
UNIDADES_TRANSITO = [0, 10, 20, 30, 40, 50]

# La tendencia condiciona la evolucion probabilistica de la demanda.
TENDENCIAS_DEMANDA = ['bajando', 'estable', 'subiendo']

# El calculo del espacio de estados usa 10 niveles declarados de inventario;
# la grilla de 0 a 100 en pasos de 10 contiene 11 valores.
NIVELES_INVENTARIO_DECLARADOS = 10

# Probabilidades ilustrativas, condicionadas al nivel y la tendencia actuales;
# no estan calibradas con datos reales.
PROB_DEMANDA = {
    'bajando': {
        'bajo':    {'bajo': 0.80, 'medio': 0.15, 'alto': 0.04, 'crítico': 0.01},
        'medio':   {'bajo': 0.40, 'medio': 0.45, 'alto': 0.12, 'crítico': 0.03},
        'alto':    {'bajo': 0.10, 'medio': 0.35, 'alto': 0.45, 'crítico': 0.10},
        'crítico': {'bajo': 0.05, 'medio': 0.20, 'alto': 0.35, 'crítico': 0.40},
    },
    'estable': {
        'bajo':    {'bajo': 0.70, 'medio': 0.20, 'alto': 0.08, 'crítico': 0.02},
        'medio':   {'bajo': 0.15, 'medio': 0.65, 'alto': 0.15, 'crítico': 0.05},
        'alto':    {'bajo': 0.05, 'medio': 0.20, 'alto': 0.60, 'crítico': 0.15},
        'crítico': {'bajo': 0.02, 'medio': 0.08, 'alto': 0.25, 'crítico': 0.65},
    },
    'subiendo': {
        'bajo':    {'bajo': 0.50, 'medio': 0.30, 'alto': 0.15, 'crítico': 0.05},
        'medio':   {'bajo': 0.05, 'medio': 0.45, 'alto': 0.35, 'crítico': 0.15},
        'alto':    {'bajo': 0.02, 'medio': 0.10, 'alto': 0.48, 'crítico': 0.40},
        'crítico': {'bajo': 0.01, 'medio': 0.04, 'alto': 0.20, 'crítico': 0.75},
    },
}

# La tendencia favorece mantenerse y cambia de forma gradual.
PROB_TENDENCIA = {
    'bajando':  {'bajando': 0.60, 'estable': 0.35, 'subiendo': 0.05},
    'estable':  {'bajando': 0.20, 'estable': 0.60, 'subiendo': 0.20},
    'subiendo': {'bajando': 0.05, 'estable': 0.35, 'subiendo': 0.60},
}


def discretizar_inventario(inv):
    """Redondea el inventario al multiplo de 10 mas cercano."""
    return int(min(100, max(0, round(inv / 10.0)))) * 10


def discretizar_dias(dias):
    """Mapea los dias restantes al valor mas cercano de la grilla."""
    return min(DIAS_VENCIMIENTO, key=lambda x: abs(x - dias))


def transition(state, action):
    """Transicion estocastica del MDP.

    state = (inventario, dias_vencimiento, demanda_nivel, unidades_en_transito, tendencia)
    action: unidades a pedir, en {0, 10, 20, 30, 40, 50}.

    Retorna (estado_siguiente, demanda_no_atendida), con el faltante en unidades.
    """
    inventario, dias_vencimiento, demanda_nivel, en_transito, tendencia = state

    # El pedido anterior llega antes de atender la demanda del dia.
    inventario_efectivo = min(100, inventario + en_transito)

    demanda_hoy = DEMAND_MAP[demanda_nivel]
    demanda_no_atendida = max(0, demanda_hoy - inventario_efectivo)
    new_inventory = discretizar_inventario(max(0, inventario_efectivo - demanda_hoy))
    new_days = discretizar_dias(max(1, dias_vencimiento - 1))

    probs_d = PROB_DEMANDA[tendencia][demanda_nivel]
    new_demand = str(np.random.choice(list(probs_d.keys()), p=list(probs_d.values())))

    probs_t = PROB_TENDENCIA[tendencia]
    new_tendencia = str(np.random.choice(list(probs_t.keys()), p=list(probs_t.values())))

    # El pedido de hoy queda en transito para manana.
    new_transito = action

    next_state = (new_inventory, new_days, new_demand, new_transito, new_tendencia)
    return next_state, int(demanda_no_atendida)


def transition_determinista(state, action):
    """Usa el resultado mas probable de cada distribucion para una transicion determinista."""
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


def estado_a_original(state):
    """Proyecta el estado de cinco variables al formato de tres variables."""
    inventario, dias_vencimiento, demanda_nivel, _, _ = state
    return (inventario, dias_vencimiento, demanda_nivel)


def estado_desde_original(state, en_transito=0, tendencia='estable'):
    """Amplia el estado con el pedido en transito y la tendencia de demanda."""
    inventario, dias_vencimiento, demanda_nivel = state
    return (inventario, dias_vencimiento, demanda_nivel, en_transito, tendencia)


def tabla_impacto():
    """Imprime la comparacion del espacio de estados y devuelve los totales."""
    acciones = 6
    dias = len(DIAS_VENCIMIENTO)
    demandas = len(NIVELES_DEMANDA)
    transitos = len(UNIDADES_TRANSITO)
    tendencias = len(TENDENCIAS_DEMANDA)

    total_original = NIVELES_INVENTARIO_DECLARADOS * dias * demandas
    total_corregido = NIVELES_INVENTARIO_DECLARADOS * dias * demandas * transitos * tendencias

    print(f'{"Dimension":38} {"Original":>10} {"Corregido":>12}')
    print(f'{"Niveles de inventario":38} {NIVELES_INVENTARIO_DECLARADOS:>10} {NIVELES_INVENTARIO_DECLARADOS:>12}')
    print(f'{"Dias hasta vencimiento":38} {dias:>10} {dias:>12}')
    print(f'{"Niveles de demanda":38} {demandas:>10} {demandas:>12}')
    print(f'{"Unidades en transito (nuevo)":38} {"---":>10} {transitos:>12}')
    print(f'{"Tendencia de demanda (nuevo)":38} {"---":>10} {tendencias:>12}')
    print()
    print(f'{"Total de estados":38} {total_original:>10} {total_corregido:>12}')
    print(f'{"Pares (estado, accion)":38} {total_original*acciones:>10} {total_corregido*acciones:>12}')
    print(f'{"Entradas de la tabla Q":38} {total_original*acciones:>10} {total_corregido*acciones:>12}')
    print()
    print('Factor de expansion:', total_corregido / total_original)

    return {
        'total_original': total_original,
        'total_corregido': total_corregido,
        'factor_expansion': total_corregido / total_original,
    }


if __name__ == '__main__':
    print('--- Tests de compatibilidad y transicion ---')
    print()

    s0 = (20, 14, 'medio', 0, 'estable')
    assert len(s0) == 5

    assert estado_a_original(s0) == (20, 14, 'medio')
    assert estado_desde_original((20, 14, 'medio')) == (20, 14, 'medio', 0, 'estable')

    s_next, stockout = transition_determinista(s0, action=10)
    assert len(s_next) == 5
    assert isinstance(stockout, int)

    # Un pedido en transito deja mas inventario que no tenerlo.
    sn_con, _ = transition_determinista((10, 14, 'medio', 30, 'estable'), 0)
    sn_sin, _ = transition_determinista((10, 14, 'medio', 0, 'estable'), 0)
    assert sn_con[0] > sn_sin[0]

    # Verifica que la demanda pueda variar entre transiciones.
    np.random.seed(0)
    demandas_observadas = {
        transition((20, 14, 'crítico', 0, 'subiendo'), 0)[0][2]
        for _ in range(100)
    }
    assert len(demandas_observadas) > 1

    print('Tests de transicion y compatibilidad: OK')
    print()

    tabla_impacto()
