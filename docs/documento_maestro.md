# WhaleTracker: Documento Maestro de Ingeniería y Estrategia

## 1. Introducción y Tesis de Inversión
**WhaleTracker** es una plataforma de análisis de asimetría informativa que explota las "goteras del capitalismo". La tesis central es que el sistema financiero no es perfectamente eficiente debido a la existencia de "insiders" (CEOs y políticos) cuyas acciones personales preceden a movimientos estructurales del mercado.

### 1.1 El Problema: El Ruido del Mercado
La mayoría de los pequeños inversores pierden dinero siguiendo noticias ya descontadas o ruidos mediáticos.

### 1.2 La Solución: El Rastro del Dinero Real
Identificar transacciones donde el que tiene la información pone su propio capital en juego. No buscamos predecir el futuro, sino observar quién está apostando por él desde dentro.

---

## 2. El Core de Verificación: Protocolo de Backtesting Estricto
Como se especificó en la concepción del proyecto, antes de cualquier inversión real de 150€, el sistema debe superar un **Motor de Verificación** riguroso que elimine sesgos y valide la latencia legal.

### 2.1 Especificaciones Técnicas del Test #0 (Ground Zero)
Para que la comprobación sea profesional y no una "caja negra" engañosa, el algoritmo de test debe seguir estas reglas innegociables (Extraído de la conversación técnica):

1. **Timestamp de Transparencia (Regla Anti-Trampas)**:
   - El error más común es usar la fecha en que el insider compró. El sistema **tiene prohibido** usar ese dato para la ejecución.
   - **Ejecución Real**: La simulación solo puede "comprar" **2 días hábiles después** de que el informe sea público en la SEC (Form 4) o el Senado (PDI). Esto asegura que el beneficio mostrado sea 100% replicable por el usuario.

2. **Criterio de Supervivencia Único (Survivorship Bias)**:
   - El test no se limita a las empresas que existen hoy. Debe alimentarse de bases de datos históricas (SEC EDGAR desde 2004) que incluyan empresas que **quebraron o desaparecieron**. Si un insider compró algo que se fue a cero, el test debe reflejar la pérdida total de los 150€.

3. **La Senda del Acelerador (Matemática de los 150€)**:
   - **Aporte Mensual**: Sumar 150€ virtuales cada mes a una bolsa de efectivo.
   - **Lógica de Acumulación**: Si en un mes el score es < 85, el dinero se mantiene en 'cash'.
   - **Mirror Exit (Salida Espejo)**: El test busca en los datos históricos cuándo ese mismo insider vendió. En esa fecha exacta, el test vende la posición.

### 2.2 El "Audit Trail" (La Caja Negra de Trazabilidad)
Cada alerta generada por el backtest debe generar un log de auditoría con los siguientes campos obligatorios para garantizar transparencia total:
- **Evento**: Acción detectada (ej. Compra del Senador X en Empresa Y).
- **Fecha de Informe**: Día en que la información llegó al dominio público.
- **Fecha de Ejecución**: Día en que el algoritmo habría comprado (Informe + 2 días).
- **Razón del Score**: Justificación técnica del motor (ej. Sector Estratégico + Miembro de Comité + Small Cap).

### 2.3 Métricas de Éxito de la Comprobación
El resultado final del test comparativo (2014-2024) debe entregar:
- **Alertas enviadas vs. Alertas rentables**: (Ej: 12 enviadas, 8 rentables - 66%).
- **Rendimiento Algoritmo vs. S&P 500**: Comparativa directa de crecimiento del capital bajo las mismas condiciones de latencia.
- **Métrica de Antifragilidad**: Capacidad del algoritmo para evitar caídas sistémicas (ej. 2022) mediante la detección de ventas masivas previas de insiders.

---

## 3. Marco Filosófico: El Método Taleb
Implementación técnica de los principios de **Nassim Nicholas Taleb**.

### 3.1 Antifragilidad y Convexidad Positiva
Buscamos apuestas donde la pérdida máxima es conocida (el capital aportado) pero el potencial de ganancia es ilimitado (Small Caps con multiplicadores x5, x8).

### 3.2 Vía Negativa (Eliminar lo Frágil)
El algoritmo descarta automáticamente empresas con alta deuda, sectores hiper-regulados o acciones con baja liquidez que "atrapan" al inversor.

### 3.3 Skin in the Game (Jugarse el pellejo)
Priorizamos compras masivas con capital personal líquido. Ignoramos bonos u opciones automáticas de compensación salarial.

### 3.4 Estrategia de la Pesa (Barbell Strategy)
90% del capital acumulado se mueve a una "fortaleza segura" (activos indexados), mientras el 10% restante sigue operando en el "acelerador" de alta asimetría.

---

## 4. Análisis de Sectores Estratégicos y Casos Reales
El sistema se especializa en "Fosos de Asimetría" validados por la historia:

| Sector | Driver de Asimetría | Caso Real de Referencia |
| :--- | :--- | :--- |
| **Defensa/Aeroespacial** | Contratos y Presupuestos. | **Mark Green**: Compras en Small Caps de logística/defensa. |
| **Tecnología/Chips** | Subsidios y Geopolítica. | **Nancy Pelosi**: LEAPS en NVDA impulsados por la Ley de Chips. |
| **Suministro Industrial** | Integración Vertical. | **Caso Toyota**: Inversión en proveedores antes de contratos masivos. |
| **Biomedicina** | Ensayos FDA Fase III. | **Clusters de Insiders**: Compras colectivas antes de resultados binarios. |

---

## 5. Arquitectura Técnica (Módulos Lego)

### 5.1 Módulo I: "El Sabueso" (Ingesta y Limpieza)
Connectors para SEC EDGAR y STOCK Act. Limpieza automática de "ruido" de ventas programadas.

### 5.2 Módulo II: "El Filtro Taleb" (Scoring Engine)
Puntuación (0-100) basada en: Autoridad del insider, Factor de Cluster (unanimidad), Tamaño de la apuesta relativo al patrimonio y Potencial de Convexidad (Market Cap).

### 5.3 Módulo III: "El Espejo" (Gestión de Alertas)
Entrada y Salida síncrona con el insider principal. Gestión del "Cubo de los 150€".

### 5.4 Módulo IV: "El Oráculo" (Trazabilidad)
Interfaz de auditoría para navegar por los logs del "Audit Trail" y validar la trazabilidad del algoritmo en cualquier punto del pasado.

---

## 6. Roadmap de Implementación
1. **Fase 1: El Test de la Verdad (Motor de Backtesting)**. Implementar el motor de 10 años con datos de SEC/Senado.
2. **Fase 2: Infraestructura (Supabase/FastAPI)**. Definir tablas de Trazabilidad y Ranking.
3. **Fase 3: Módulos Activos**. Scrapers y Motor de Scoring en tiempo real.
4. **Fase 4: El Chivato**. Notificaciones de entrada/salida y Dashboard de Trazabilidad.
