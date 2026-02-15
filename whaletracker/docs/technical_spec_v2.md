# WhaleTracker Technical Specification v2.0

> **La Santísima Trinidad**: Taleb (filosofía) + López de Prado (metodología) + Mandelbrot (estructura de la realidad)

---

## 1. Visión y Tesis

WhaleTracker explota **asimetría informativa legal**: seguimos el dinero real de insiders, activistas, instituciones y políticos. No predecimos el futuro — nos preparamos para la convexidad.

**Edge**: Mientras otros persiguen momentum y noticias, nosotros seguimos a quienes ponen su dinero donde tienen información estructural.

**Disciplina**: Stop-loss estricto (-15%), diversificación (nunca >10% en single position), exit cuando insider sale.

---

## 2. Pilares Teóricos

### 2.1 Taleb (Estrategia Antifrágil)
- **Convexidad**: Downside limitado (stop-loss), upside ilimitado (x20+)
- **Via Negativa**: Eliminar malas operaciones > buscar buenas
- **Barbell**: 90% capital seguro, 10% en señales de alta convicción
- **Skin in the Game**: Solo seguimos insiders que compran con su propio dinero

### 2.2 López de Prado (Rigor Científico ML)

| Técnica | Criticidad | Propósito |
|---------|-----------|-----------|
| **Triple Barrier Method** | ⚠️ CRÍTICO | Labels realistas (profit/loss/timeout), elimina lookahead bias |
| **Fractional Differencing** | ⚠️ CRÍTICO | Features stationary que mantienen memoria (d óptimo ~0.4) |
| **Sample Weights (Uniqueness)** | 🟡 MUY IMPORTANTE | Previene overfitting por samples solapados |
| **Purged K-Fold CV** | ⚠️ CRÍTICO | Cross-validation sin leakage temporal (purging + embargo) |
| **Meta-Labeling** | 🟡 IMPORTANTE | Bet sizing óptimo (primary model + meta model) |
| **MDA Feature Importance** | 🟢 RECOMENDADO | Feature selection correcto (vs MDI biased) |

#### Triple Barrier Method
En lugar de labeling simple (sube/baja), define 3 barreras por trade:
- **Superior**: Profit target (2× volatilidad)
- **Inferior**: Stop-loss (1× volatilidad)
- **Temporal**: Max holding period (180 días)

Label = primera barrera tocada. Captura asimetría real del trading.

#### Meta-Labeling (Bet Sizing)
Dos modelos en cascada:
1. **Primary**: Detecta oportunidades (insider buy → ¿buena oportunidad?)
2. **Meta**: Decide tamaño de posición (0-100% del capital asignado)

`bet_size = P(primary_acierta) × P(meta_confirma)`

#### Purged K-Fold
- **Purging**: Elimina de train los samples que solapan temporalmente con test
- **Embargo**: Gap adicional post-test para evitar look-ahead
- Si el modelo pasa CPCV → seguridad estadística real (no suerte)

### 2.3 Mandelbrot (Estructura Fractal)

#### Levy Stable Distribution
Los mercados **NO** siguen distribución normal. Fat tails son reales:
- **α = 2**: Normal (Gauss)
- **α < 1.8**: Fat tails significativos
- **α < 1.5**: EXTREME fat tails — eventos x20+ estadísticamente más probables

#### Hurst Exponent (Persistencia)
- **H > 0.5**: Serie con memoria (tendencia persistente)
- **H = 0.5**: Random walk (ruido)
- **H < 0.5**: Mean reversion

Aplicación: Si H > 0.6 después de insider buy → tendencia se auto-alimenta → **SEÑAL FUERTE**

#### Tiempo Fractal (Intrinsic Time)
El mercado no se mueve por minutos, sino por información:
- 1 día aburrido = 1 tick de información
- 1 flash crash = 1000 ticks de información

Implementación: tick cuando `(volumen × volatilidad) > threshold`

#### Multifractal Spectrum
Detectar cambio de régimen: si H(q) varía significativamente → mercado cambia de trending a mean-reverting.

---

## 3. Fuentes de Datos

### 3.1 Señales de Entrada (Triggers)

| Fuente | Tipo | Coste | Señal |
|--------|------|-------|-------|
| **SEC EDGAR Form 4** | Insider purchases | GRATIS | Entry trigger primario |
| **OpenInsider** | Cluster buys | GRATIS | Entry trigger fuerte |
| **SEC EDGAR Form D** | Private offerings (startups) | GRATIS | Startup funding signal |
| **SEC EDGAR 13D/13G** | Activist stakes | GRATIS | Strategic investment |
| **SEC EDGAR 13F** | Institutional holdings | GRATIS | Confirmación (45d lag) |
| **SBIR/STTR Database** | Gov grants a startups | GRATIS | Validación tecnológica |
| **USPTO Patents** | Patentes tech | GRATIS | Innovación real + IP |
| **OTC Markets** | Pre-NASDAQ startups | GRATIS | Universo de micro-caps |
| **USAspending.gov** | Contratos gobierno | GRATIS | Validación de "gotera" |
| **Crunchbase** | VC funding rounds | $29-99/mo | Enriquecimiento (Fase 2+) |

### 3.2 Datos de Mercado

| Fuente | Tipo | Coste | Uso |
|--------|------|-------|-----|
| **Yahoo Finance** | OHLCV diario | GRATIS | Fase 1: desarrollo y backtest |
| **Polygon.io** | Tick-by-tick | $199/mo | Fase 2: datos Mandelbrot fractales |
| **IEX Cloud** | Institucional | $9-499/mo | Balance calidad/precio |
| **Financial Modeling Prep** | Fundamentales | Medio | Históricos + estados financieros |

### 3.3 Microestructura (Fase Avanzada)

| Fuente | Tipo | Coste | Uso |
|--------|------|-------|-----|
| **Interactive Brokers** | Order book Level 2 | Con cuenta | VPIN + whale detection |
| **Alpaca Markets** | WebSocket real-time | GRATIS | Puente a FIX sin complejidad |

### 3.4 Estrategia de Datos por Fase
1. **Fase 1**: SEC EDGAR (gratis) + Yahoo Finance (gratis) → Lógica base + backtest
2. **Fase 2**: + Polygon / IEX → Velocidad + datos fractales
3. **Fase 3**: + IBKR / FIX → Microestructura real

---

## 4. Universo de Tracking

### 4.1 Insiders Corporativos (Form 4)
- Officers (CEO, CFO, CTO)
- Directors
- 10% owners
- Señales: purchases (entry), sales (exit), cluster buys (strong entry)

### 4.2 Activistas e Instituciones (13D/13G, 13F)
- 12 super-investors tracked: Berkshire Hathaway, Bridgewater, Renaissance, Baupost, Pershing Square, etc.
- Strategic investors: empresas comprando en su sector (Toyota → proveedor)

### 4.3 Startups y Penny Stocks (Form D, SBIR, OTC)
**Sweet spot de convexidad extrema**:
- Market cap $10M-$500M
- Float < 50M shares
- Sectores estratégicos: Quantum, AI, Defense Tech, Clean Energy, Biotech, Advanced Materials
- Una ballena comprando $200K = 2% del market cap → movimiento instantáneo

**Criterio de qualidad para startups:**
- SBIR Phase II award ✓ (gobierno validó la tecnología)
- Patentes en tech disruptiva ✓ (IP real)
- VC de primer nivel invirtiendo ✓ (smart money)
- Cotiza en OTC/exchange ✓ (podemos comprar con €150)

**VC/Whale list para startups:**
Andreessen Horowitz, Sequoia Capital, Founders Fund, Lux Capital, In-Q-Tel (CIA venture arm), Breakthrough Energy Ventures (Gates), ARK Invest

### 4.4 Políticos (STOCK Act)
- Senadores y congresistas con committee alignment
- Validación: USAspending.gov para confirmar contratos

---

## 5. Arquitectura del Sistema

```
LAYER 1: DATA INGESTION
├── SEC EDGAR (Form 4, Form D, 13D/13G, 13F, S-1)
├── OpenInsider (Cluster buys, instant data)
├── Senate/Congress (STOCK Act disclosures)
├── SBIR/STTR Database (Gov grants)
├── USPTO Patents (Innovation)
├── OTC Markets (Pre-NASDAQ)
├── Market Data (Yahoo → Polygon → FIX)
└── USAspending.gov (Government contracts)

LAYER 2: FEATURE ENGINEERING (30+ → 55+ features)
├── Insider Behavior (5): win_rate, frequency, consistency, holding, size
├── Transaction (3): is_purchase, filing_delay, value_zscore
├── Timing (4): days_since_crash, earnings_proximity, sector_momentum
├── Company (4): log_market_cap, volatility, short_interest, volume_anomaly
├── Cluster (3): num_buyers, temporal_density, quality
├── Macro (3): vix, yield_curve, dxy
├── Political (3): is_politician, committee_alignment, seniority
├── Whale (5): whale_type, cluster_count, institutional_accumulation,
│              is_strategic, buyer_conviction
├── Mandelbrot (future): hurst_exponent, levy_alpha, fractal_regime
└── Startup (future): sbir_validated, patent_count, vc_tier, float_pct

LAYER 3: ML MODELS
├── Quantum Dense Network (QDN) — Core scoring model
├── Triple Barrier Labeling — Realistic labels
├── Meta-Labeling — Bet sizing
├── Purged K-Fold CV — Validation sin leakage
└── Sample Weights — Anti-overfitting

LAYER 4: ANALYSIS ENGINES
├── Mandelbrot Analyzer (Hurst, Levy, Fractal Time)
├── WhaleConnector (Form 4 + 13D + 13F unified)
├── Startup Screener (SBIR + Patents + OTC)
├── Inference API (score_opportunity + check_exit_signals)
└── Monitor (continuous scanning 6h loop)

LAYER 5: RISK MANAGEMENT
├── Stop-loss: -15% per position
├── Position sizing: nunca >10% en single
├── Exit signals: insider sales monitoring
└── Portfolio: Kelly criterion + diversification
```

---

## 6. Módulos Implementados (Actual)

| Módulo | Estado | Ubicación |
|--------|--------|-----------|
| `QDNConfig` | ✅ Completo | `qdn/config.py` |
| `DenseNetwork` | ✅ Completo | `qdn/dense_network.py` |
| `FeatureEngineer` (30 features) | ✅ Completo | `qdn/features/engineer.py` |
| `SECConnector` (Form 4) | ✅ Completo | `qdn/data/sec_connector.py` |
| `MarketConnector` | ✅ Completo | `qdn/data/market_connector.py` |
| `SenateConnector` | ✅ Completo | `qdn/data/senate_connector.py` |
| `WhaleConnector` | ✅ Completo | `qdn/data/whale_connector.py` |
| `QDNPipeline` | ✅ Completo | `qdn/pipeline.py` |
| `QDNInference` | ✅ Completo | `qdn/inference.py` |
| `QDNMonitor` | ✅ Completo | `qdn/monitor.py` |

## 7. Módulos Por Implementar (Roadmap)

| Módulo | Prioridad | Fase |
|--------|-----------|------|
| `TripleBarrierLabeling` | ⚠️ CRÍTICO | Phase 2 |
| `FractionalDifferentiation` | ⚠️ CRÍTICO | Phase 3 |
| `SampleWeights` | 🟡 IMPORTANTE | Phase 3 |
| `PurgedKFold` | ⚠️ CRÍTICO | Phase 2 |
| `MetaLabeling` | 🟡 IMPORTANTE | Phase 3 |
| `MandelbrotAnalyzer` (Hurst + Levy) | 🟡 IMPORTANTE | Phase 3 |
| `FractalTimeProcessor` | 🟢 NICE-TO-HAVE | Phase 4 |
| `StartupDataSources` (SBIR, USPTO, OTC) | 🟡 IMPORTANTE | Phase 4 |
| `PennyStockHunter` | 🟡 IMPORTANTE | Phase 4 |
| `StrategicTechStartupScreener` | 🟢 RECOMENDADO | Phase 4 |
| `FIXDataConnector` + `VPINCalculator` | 🟢 OPCIONAL | Phase 6 |
| `DataQualityPipeline` | 🟡 IMPORTANTE | Phase 2 |

---

## 8. Risk Management Rules ("5 Leyes Sagradas")

1. **Nunca más del 10% en una sola posición** — Diversificación obligatoria
2. **Stop-loss al -15%** — Proteger capital es no-negociable
3. **Exit inmediato si insider vende** — Si ellos salen, nosotros salimos
4. **Solo comprar purchases, nunca perseguir ventas** — Las ventas son ruido (impuestos, divorcios)
5. **Si no pasa el Triple Barrier Test → no operar** — El algoritmo no se autoengaña

---

## 9. Roadmap de Implementación

### FASE 1: Fundación (Mes 1-2) ← **EN PROGRESO**
- [x] Core QDN model (DenseNetwork + config)
- [x] Data pipeline (SEC, Senate, Market connectors)
- [x] Feature engineering (30 features)
- [x] WhaleConnector (Form 4 + 13D + 13F)
- [x] Inference API + Monitor
- [ ] Backtest 10 años completo
- [ ] Validar Sortino > 2.0

### FASE 2: ML Avanzado (Mes 3-4)
- [ ] Triple Barrier Labeling
- [ ] Purged K-Fold CV
- [ ] Data Quality Pipeline
- [ ] Adversarial training mejorado
- [ ] Validar Sortino > 2.5

### FASE 3: Mandelbrot + López de Prado (Mes 5)
- [ ] Hurst Exponent calculation
- [ ] Levy distribution fitting
- [ ] Fractional differencing
- [ ] Sample weights (uniqueness)
- [ ] Meta-labeling (bet sizing)
- [ ] Re-train modelos v2

### FASE 4: Startups & Penny Stocks (Mes 6)
- [ ] SBIR/STTR data fetcher
- [ ] USPTO patent scanner
- [ ] OTC Markets screener
- [ ] Startup whale detector
- [ ] Penny stock universe (~500 companies)
- [ ] Backtest startup performance

### FASE 5: API & Backend Producción (Mes 7-8)
- [ ] FastAPI endpoints
- [ ] PostgreSQL schema
- [ ] Redis caching
- [ ] Authentication + rate limiting
- [ ] Monitoring (Prometheus)
- [ ] Cloud deployment

### FASE 6: FIX & Microestructura [Opcional] (Mes 9)
- [ ] FIX connector (IBKR)
- [ ] VPIN calculation
- [ ] Order book analysis
- [ ] A/B test FIX vs no-FIX

### FASE 7-10: Frontend, Beta, Launch (Mes 10-13+)
- [ ] Dashboard (React)
- [ ] Beta testing (10 usuarios)
- [ ] Legal review + compliance
- [ ] Public launch

---

## 10. Referencias

**Libros:**
1. Nassim Nicholas Taleb — *Antifragile*
2. Marcos López de Prado — *Advances in Financial Machine Learning*
3. Benoît Mandelbrot — *The Misbehavior of Markets*

**Papers:**
1. López de Prado — "The 10 Reasons Most Machine Learning Funds Fail"
2. Mandelbrot & Hudson — "A Multifractal Walk Down Wall Street"

**APIs:**
- SEC EDGAR: https://www.sec.gov/edgar/
- SBIR Database: https://www.sbir.gov/
- USPTO PatentsView: https://patentsview.org/
- OTC Markets: https://www.otcmarkets.com/

---

## 11. Glosario

| Término | Definición |
|---------|-----------|
| **Convexidad** | Asimetría donde downside limitado, upside ilimitado |
| **Hurst Exponent** | Medida de persistencia en series temporales (0-1) |
| **Levy Distribution** | Distribución con fat tails para eventos extremos |
| **VPIN** | Volume-Synchronized Probability of Informed Trading |
| **Triple Barrier** | Labeling con 3 barreras (profit, loss, time) |
| **Purged K-Fold** | Cross-validation sin leakage temporal |
| **Meta-Labeling** | Dos modelos (primary + meta) para bet sizing |
| **SBIR** | Small Business Innovation Research (gov grants) |
| **QDN** | Quantum Dense Network |
