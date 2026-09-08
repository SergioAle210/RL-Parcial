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
