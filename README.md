# RL-Parcial

Auditamos la representación del estado y la función de transición de un sistema de inventario farmacéutico como parte del **grupo 1**.

## Trabajo realizado

- **1.1:** analizamos la propiedad de Markov y mostramos cómo la información omitida puede afectar la predicción del estado siguiente.
- **1.2:** estudiamos el supuesto de demanda constante y comparamos escenarios de picos y caídas de demanda con ejemplos ejecutables.
- **1.3:** corregimos el MDP agregando unidades en tránsito (restaura la propiedad de Markov) y tendencia de demanda (deja de estar congelada), verificado contra los contraejemplos de 1.1 y 1.2.
- **1.4:** medimos el impacto del MDP corregido sobre el espacio de estados (200 a 3600 estados, 18 veces más) y sus implicaciones para el algoritmo de aprendizaje.
- Relacionamos los hallazgos con las métricas de producción, verificamos los cálculos e incluimos referencias de apoyo.

## Archivos

- [parcial_practico.ipynb](parcial_practico.ipynb): reunimos el análisis completo (1.1 a 1.4), el código y sus resultados en un cuaderno Jupyter.
- [parcial_practico.py](parcial_practico.py): reunimos únicamente el código y docstrings breves, espejo del notebook.
- [mdp_corregido.py](mdp_corregido.py): el MDP corregido (1.3) y la tabla de impacto (1.4) como módulo independiente, listo para compartir con los Grupos 3, 4 y 6 durante la sesión presencial.

Están pendientes las cuatro preguntas de integración y la reflexión grupal, que se resuelven durante y después de la sesión presencial con los resultados de los demás grupos.
