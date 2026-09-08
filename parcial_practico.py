def transition(state, action):
    """Calcula el estado siguiente de forma determinista; la acción son unidades y la categoría de demanda permanece fija."""
    inventory, days_to_expiry, demand_level = state
    demand_map = {'bajo': 5, 'medio': 15, 'alto': 25, 'crítico': 40}
    daily_demand = demand_map[demand_level]
    new_inventory = min(100, max(0, inventory + action - daily_demand))
    new_days = max(1, days_to_expiry - 1)
    new_demand = demand_level
    return (new_inventory, new_days, new_demand)


resultados_produccion = {
    'stockouts_por_semana': 23,
    'productos_vencidos_por_semana': 41,
    'costo_almacenamiento_semanal': 8400,
    'costo_objetivo_semanal': 3200,
    'satisfaccion_cliente': 0.61,
}


estado_compartido = (20, 14, 'medio')
accion_compartida = 0
inventario_con_pedido_previo = min(100, max(0, 20 + 30 - 15))
inventario_sin_pedido_previo = min(100, max(0, 20 + 0 - 15))
print('Ejemplo hipotético: mismo estado y acción, distintas historias')
print(f'Historia con entrega pendiente: inventario siguiente = {inventario_con_pedido_previo}')
print(f'Historia sin entrega pendiente: inventario siguiente = {inventario_sin_pedido_previo}')
print(f'Transición original: {transition(estado_compartido, accion_compartida)}')
assert inventario_con_pedido_previo == 35
assert inventario_sin_pedido_previo == 5


ventana_a = [25, 10, 10, 15, 15, 15, 15]
ventana_b = [10, 10, 15, 15, 15, 15, 25]
demanda_manana = 15
media_actual_a = sum(ventana_a) / 7
media_actual_b = sum(ventana_b) / 7
media_siguiente_a = (sum(ventana_a[1:]) + demanda_manana) / 7
media_siguiente_b = (sum(ventana_b[1:]) + demanda_manana) / 7
print(f'Medias actuales: {media_actual_a:.2f} y {media_actual_b:.2f}')
print(f'Medias siguientes: {media_siguiente_a:.2f} y {media_siguiente_b:.2f}')
assert media_actual_a == media_actual_b == 15
assert media_siguiente_a != media_siguiente_b


estado = (60, 14, 'bajo')
trayectoria_original = [estado]
for _ in range(4):
    estado = transition(estado, 0)
    trayectoria_original.append(estado)

print('Transición original con acción de cero unidades')
print('Día | Inventario | Días al vencimiento | Demanda')
for dia, (inventario, vencimiento, nivel) in enumerate(trayectoria_original):
    print(f'{dia} | {inventario} | {vencimiento} | {nivel}')
assert all(s[2] == 'bajo' for s in trayectoria_original)


def balance_diagnostico(inventario_inicial, acciones, demandas):
    """Compara demandas hipotéticas en el balance del MDP y registra unidades no atendidas que el recorte a cero oculta."""
    inventario = inventario_inicial
    filas = []
    for dia, (accion, demanda) in enumerate(zip(acciones, demandas), start=1):
        disponible = inventario + accion
        faltante = max(0, demanda - disponible)
        inventario = min(100, max(0, disponible - demanda))
        filas.append((dia, demanda, inventario, faltante))
    return filas


casos = [
    ('Demanda fija', 60, [0] * 4, [5, 5, 5, 5]),
    ('Pico hipotético', 60, [0] * 4, [5, 40, 40, 5]),
    ('Rotación fija', 50, [10] * 4, [5, 5, 5, 5]),
    ('Caída hipotética', 50, [10] * 4, [0, 0, 0, 0]),
]
balances = {}
print('Escenario | Demanda por día | Inventarios al cierre | Unidades no atendidas')
for nombre, inicial, acciones, demandas in casos:
    filas = balance_diagnostico(inicial, acciones, demandas)
    balances[nombre] = filas
    cierres = [fila[2] for fila in filas]
    no_atendidas = sum(fila[3] for fila in filas)
    print(f'{nombre} | {demandas} | {cierres} | {no_atendidas}')
assert [fila[2] for fila in balances['Demanda fija']] == [s[0] for s in trayectoria_original[1:]]
assert sum(fila[3] for fila in balances['Pico hipotético']) == 30
assert balances['Rotación fija'][-1][2] == 70
assert balances['Caída hipotética'][-1][2] == 90


razon_costo = (
    resultados_produccion['costo_almacenamiento_semanal']
    / resultados_produccion['costo_objetivo_semanal']
)
print(f'Costo de almacenamiento / objetivo: {razon_costo:.3f}')
print('Verificamos los ejemplos de los entregables 1.1 y 1.2.')


# ---------------------------------------------------------------------------
# Entregable 1.3: MDP corregido
# ---------------------------------------------------------------------------
import numpy as np

DIAS_VENCIMIENTO = [1, 7, 14, 30, 60]
NIVELES_DEMANDA = ['bajo', 'medio', 'alto', 'crítico']
UNIDADES_TRANSITO = [0, 10, 20, 30, 40, 50]
TENDENCIAS_DEMANDA = ['bajando', 'estable', 'subiendo']
DEMAND_MAP = {'bajo': 5, 'medio': 15, 'alto': 25, 'crítico': 40}

# Probabilidad de que la demanda pase a cada nivel, condicionada a la tendencia
# actual y al nivel actual. Es un supuesto ilustrativo, no calibrado con datos
# reales, que hace que la demanda suba con mas probabilidad cuando la tendencia
# es 'subiendo', y baje con mas probabilidad cuando es 'bajando'.
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

# Probabilidad de que la tendencia misma cambie. Es persistente (alta
# probabilidad de mantenerse) pero puede transicionar de forma gradual.
PROB_TENDENCIA = {
    'bajando':  {'bajando': 0.60, 'estable': 0.35, 'subiendo': 0.05},
    'estable':  {'bajando': 0.20, 'estable': 0.60, 'subiendo': 0.20},
    'subiendo': {'bajando': 0.05, 'estable': 0.35, 'subiendo': 0.60},
}


def discretizar_inventario(inv):
    """Redondea el inventario al multiplo de 10 mas cercano, acotado en [0, 100].

    El propio codigo original ya asume una grilla de multiplos de 10 en su
    comentario de cabecera, pero su funcion transition() nunca redondea a esa
    grilla (con demanda 'bajo'=5 el inventario cae en 55, 45, valores que no
    son multiplos de 10, como vimos en el entregable 1.2). Para poder usar una
    tabla Q finita como la que espera el Grupo 3, discretizamos explicitamente.
    """
    return int(min(100, max(0, round(inv / 10.0)))) * 10


def discretizar_dias(dias):
    """Mapea los dias restantes al valor mas cercano dentro de la grilla declarada."""
    return min(DIAS_VENCIMIENTO, key=lambda x: abs(x - dias))


def transition_corregida(state, action, rng=None):
    """Transicion del MDP corregido.

    state = (inventario, dias_vencimiento, demanda_nivel, unidades_en_transito, tendencia)
    action: unidades pedidas hoy, en {0, 10, 20, 30, 40, 50}
    rng: generador de numpy para muestrear la demanda y la tendencia siguientes;
         si es None usamos el valor mas probable de cada distribucion (version
         determinista, util para depurar y para comparar contra el original).

    Retorna (estado_siguiente, demanda_no_atendida).
    """
    inventario, dias_vencimiento, demanda_nivel, en_transito, tendencia = state

    # El pedido de ayer llega hoy antes de servir la demanda. Este es el
    # cambio que restaura la propiedad de Markov (entregable 1.1).
    inventario_efectivo = min(100, inventario + en_transito)

    demanda_hoy = DEMAND_MAP[demanda_nivel]
    demanda_no_atendida = max(0, demanda_hoy - inventario_efectivo)
    new_inventory = discretizar_inventario(max(0, inventario_efectivo - demanda_hoy))
    new_days = discretizar_dias(max(1, dias_vencimiento - 1))

    probs_d = PROB_DEMANDA[tendencia][demanda_nivel]
    probs_t = PROB_TENDENCIA[tendencia]
    if rng is None:
        new_demand = max(probs_d, key=probs_d.get)
        new_tendencia = max(probs_t, key=probs_t.get)
    else:
        # str(...) porque rng.choice sobre una lista de str devuelve un
        # escalar numpy (np.str_), y preferimos mantener el estado como
        # tipos nativos de Python en toda la tupla.
        new_demand = str(rng.choice(list(probs_d.keys()), p=list(probs_d.values())))
        new_tendencia = str(rng.choice(list(probs_t.keys()), p=list(probs_t.values())))

    # El pedido de hoy queda en transito para manana.
    new_transito = action

    next_state = (new_inventory, new_days, new_demand, new_transito, new_tendencia)
    return next_state, int(demanda_no_atendida)


def estado_a_original(state):
    """Proyecta el estado corregido (5 variables) al formato original (3 variables),
    para que el codigo de los Grupos 3, 4 y 6 siga funcionando sin cambios."""
    inventario, dias_vencimiento, demanda_nivel, _, _ = state
    return (inventario, dias_vencimiento, demanda_nivel)


def estado_desde_original(state, en_transito=0, tendencia='estable'):
    """Expande un estado original (3 variables) al formato corregido (5 variables),
    con valores neutros por defecto cuando no hay informacion adicional."""
    inventario, dias_vencimiento, demanda_nivel = state
    return (inventario, dias_vencimiento, demanda_nivel, en_transito, tendencia)


estado_con_pedido = (20, 14, 'medio', 30, 'estable')
estado_sin_pedido = (20, 14, 'medio', 0, 'estable')

siguiente_con_pedido, faltante_con_pedido = transition_corregida(estado_con_pedido, 0)
siguiente_sin_pedido, faltante_sin_pedido = transition_corregida(estado_sin_pedido, 0)

print('Estado aumentado con entrega pendiente:', estado_con_pedido)
print('Inventario siguiente:', siguiente_con_pedido[0])
print()
print('Estado aumentado sin entrega pendiente:', estado_sin_pedido)
print('Inventario siguiente:', siguiente_sin_pedido[0])
print()
print('En el entregable 1.1, ambas historias compartian el mismo estado observado (20, 14, medio)')
print('y producian inventarios distintos (35 y 5) sin que el estado por si solo lo explicara.')
print('Ahora cada estado aumentado predice su propio resultado de forma determinista.')

repeticiones = [transition_corregida(estado_con_pedido, 0)[0][0] for _ in range(5)]
assert len(set(repeticiones)) == 1
assert siguiente_con_pedido[0] != siguiente_sin_pedido[0]


rng = np.random.default_rng(0)
estado = (50, 30, 'bajo', 0, 'subiendo')
niveles_observados = []
for _ in range(30):
    estado, _ = transition_corregida(estado, 20, rng=rng)
    niveles_observados.append(estado[2])

print('Niveles de demanda en 30 pasos consecutivos, partiendo de bajo con tendencia subiendo:')
print(niveles_observados)
print()
print('Niveles distintos observados:', sorted(set(niveles_observados)))
assert len(set(niveles_observados)) > 1


estado_corregido_ejemplo = (40, 14, 'alto', 20, 'estable')
proyectado = estado_a_original(estado_corregido_ejemplo)
expandido = estado_desde_original((40, 14, 'alto'))

print('Estado corregido:', estado_corregido_ejemplo)
print('Proyectado al formato original de tres variables:', proyectado)
print()
print('Estado original recibido de otro grupo:', (40, 14, 'alto'))
print('Expandido al formato corregido con valores neutros por defecto:', expandido)

assert proyectado == (40, 14, 'alto')
assert expandido == (40, 14, 'alto', 0, 'estable')


# ---------------------------------------------------------------------------
# Entregable 1.4: impacto sobre el espacio de estados
# ---------------------------------------------------------------------------
niveles_inventario_declarados = 10
niveles_inventario_reales = len(list(range(0, 101, 10)))
print('Niveles de inventario que declara el comentario original:', niveles_inventario_declarados)
print('Niveles de inventario que enumera realmente ese mismo comentario:', niveles_inventario_reales)
print()

acciones = 6
dias = len(DIAS_VENCIMIENTO)
demandas = len(NIVELES_DEMANDA)
transitos = len(UNIDADES_TRANSITO)
tendencias = len(TENDENCIAS_DEMANDA)

total_original = niveles_inventario_declarados * dias * demandas
total_corregido = niveles_inventario_declarados * dias * demandas * transitos * tendencias
factor = total_corregido / total_original

total_original_11 = niveles_inventario_reales * dias * demandas
total_corregido_11 = niveles_inventario_reales * dias * demandas * transitos * tendencias
factor_11 = total_corregido_11 / total_original_11

print(f'{"Dimension":38} {"Original":>10} {"Corregido":>12}')
print(f'{"Niveles de inventario":38} {niveles_inventario_declarados:>10} {niveles_inventario_declarados:>12}')
print(f'{"Dias hasta vencimiento":38} {dias:>10} {dias:>12}')
print(f'{"Niveles de demanda":38} {demandas:>10} {demandas:>12}')
print(f'{"Unidades en transito (nuevo)":38} {"---":>10} {transitos:>12}')
print(f'{"Tendencia de demanda (nuevo)":38} {"---":>10} {tendencias:>12}')
print()
print(f'{"Total de estados":38} {total_original:>10} {total_corregido:>12}')
print(f'{"Pares (estado, accion)":38} {total_original*acciones:>10} {total_corregido*acciones:>12}')
print(f'{"Entradas de la tabla Q":38} {total_original*acciones:>10} {total_corregido*acciones:>12}')
print()
print('Factor de expansion usando 10 niveles de inventario:', factor)
print('Factor de expansion usando 11 niveles de inventario:', factor_11)


episodios_originales = 1000
pasos_por_episodio_supuesto = 30  # supuesto ilustrativo, el env no esta disponible

visitas_promedio_original = (episodios_originales * pasos_por_episodio_supuesto) / (total_original * acciones)
episodios_necesarios_corregido = visitas_promedio_original * (total_corregido * acciones) / pasos_por_episodio_supuesto

print('Visitas promedio por par (estado, accion) en el original con 1000 episodios:', round(visitas_promedio_original, 2))
print('Episodios necesarios en el corregido para mantener esa misma cobertura promedio:', round(episodios_necesarios_corregido))
