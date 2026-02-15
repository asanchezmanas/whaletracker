-- QDN Supabase Schema
-- Core database schema for insider trading intelligence

-- ==============================================================
-- 1. INSIDER TRANSACTIONS (primary data source)
-- ==============================================================
CREATE TABLE IF NOT EXISTS insider_transactions (
    id              UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    ticker          TEXT NOT NULL,
    company_name    TEXT,
    cik             TEXT,                     -- SEC Central Index Key
    insider_name    TEXT NOT NULL,
    insider_title   TEXT,                     -- CEO, CFO, Director, Senator, etc.
    
    -- Transaction details
    transaction_date  DATE NOT NULL,
    filing_date       DATE NOT NULL,
    transaction_code  TEXT NOT NULL,           -- P=Purchase, S=Sale, A=Award, etc.
    shares            BIGINT,
    price             DECIMAL(12, 4),
    value             DECIMAL(14, 2),          -- shares * price
    ownership_after   BIGINT,                  -- Total shares after transaction
    
    -- Source metadata
    source          TEXT NOT NULL,             -- 'sec_form4', 'senate', 'house'
    raw_data        JSONB,                     -- Original raw record for audit
    
    -- Timestamps
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT unique_transaction UNIQUE (ticker, insider_name, transaction_date, transaction_code, shares)
);

CREATE INDEX idx_transactions_ticker ON insider_transactions(ticker);
CREATE INDEX idx_transactions_date ON insider_transactions(transaction_date);
CREATE INDEX idx_transactions_insider ON insider_transactions(insider_name);
CREATE INDEX idx_transactions_source ON insider_transactions(source);
CREATE INDEX idx_transactions_code ON insider_transactions(transaction_code);

-- ==============================================================
-- 2. COMPANIES
-- ==============================================================
CREATE TABLE IF NOT EXISTS companies (
    ticker          TEXT PRIMARY KEY,
    name            TEXT,
    sector          TEXT,
    industry        TEXT,
    market_cap      BIGINT,
    cik             TEXT UNIQUE,
    exchange        TEXT,                      -- NYSE, NASDAQ, etc.
    status          TEXT DEFAULT 'active',     -- active, delisted, merged
    
    -- Updated periodically
    float_pct       DECIMAL(5, 2),             -- Float percentage
    short_interest  DECIMAL(5, 2),             -- Short interest as % of float
    avg_volume_30d  BIGINT,
    
    -- Metadata
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_companies_sector ON companies(sector);
CREATE INDEX idx_companies_cik ON companies(cik);

-- ==============================================================
-- 3. INSIDERS
-- ==============================================================
CREATE TABLE IF NOT EXISTS insiders (
    id              UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    name            TEXT NOT NULL UNIQUE,
    title           TEXT,
    insider_type    TEXT NOT NULL,              -- 'corporate', 'senator', 'representative'
    
    -- Political fields (senators/representatives)
    party           TEXT,
    state           TEXT,
    committees      TEXT[],                    -- Array of committee names
    seniority_years INTEGER,
    
    -- Computed metrics (updated periodically)
    historical_win_rate   DECIMAL(5, 4),       -- % of profitable trades
    avg_return_6m         DECIMAL(8, 4),        -- Average 6-month return
    total_trades          INTEGER DEFAULT 0,
    
    -- Metadata
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_insiders_type ON insiders(insider_type);
CREATE INDEX idx_insiders_party ON insiders(party);

-- ==============================================================
-- 4. FEATURE VECTORS (computed features for each transaction)
-- ==============================================================
CREATE TABLE IF NOT EXISTS feature_vectors (
    id              UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    transaction_id  UUID NOT NULL REFERENCES insider_transactions(id),
    computed_at     TIMESTAMPTZ DEFAULT NOW(),
    model_version   TEXT NOT NULL,
    
    -- Features stored as JSONB for flexibility
    -- (schema evolves as we add features in phases)
    features        JSONB NOT NULL,
    
    -- Metadata
    feature_count   INTEGER,
    
    CONSTRAINT unique_feature_vector UNIQUE (transaction_id, model_version)
);

CREATE INDEX idx_features_transaction ON feature_vectors(transaction_id);
CREATE INDEX idx_features_version ON feature_vectors(model_version);

-- ==============================================================
-- 5. PREDICTIONS
-- ==============================================================
CREATE TABLE IF NOT EXISTS predictions (
    id                UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    transaction_id    UUID NOT NULL REFERENCES insider_transactions(id),
    model_version     TEXT NOT NULL,
    
    -- Model outputs
    convexity_score   DECIMAL(6, 2),           -- [0, 100]
    expected_return   DECIMAL(8, 4),
    downside_risk     DECIMAL(5, 4),            -- [0, 1]
    upside_potential  DECIMAL(8, 4),
    tail_probability  DECIMAL(5, 4),            -- [0, 1]
    
    -- Actual outcome (filled later for validation)
    actual_return_1m  DECIMAL(8, 4),
    actual_return_3m  DECIMAL(8, 4),
    actual_return_6m  DECIMAL(8, 4),
    actual_return_12m DECIMAL(8, 4),
    
    -- Metadata
    created_at        TIMESTAMPTZ DEFAULT NOW(),
    
    CONSTRAINT unique_prediction UNIQUE (transaction_id, model_version)
);

CREATE INDEX idx_predictions_transaction ON predictions(transaction_id);
CREATE INDEX idx_predictions_score ON predictions(convexity_score);
CREATE INDEX idx_predictions_version ON predictions(model_version);

-- ==============================================================
-- 6. MONITORING
-- ==============================================================
CREATE TABLE IF NOT EXISTS drift_reports (
    id                  UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    report_date         DATE NOT NULL,
    model_version       TEXT NOT NULL,
    n_drifted_features  INTEGER,
    is_significant      BOOLEAN DEFAULT false,
    report              JSONB,                 -- Full drift analysis
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS model_registry (
    id              UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    version         TEXT NOT NULL UNIQUE,
    status          TEXT DEFAULT 'staging',    -- staging, production, retired
    metrics         JSONB,                     -- Training & validation metrics
    config          JSONB,                     -- Model config snapshot
    artifact_path   TEXT,                      -- Path to saved model weights
    trained_at      TIMESTAMPTZ,
    promoted_at     TIMESTAMPTZ,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ==============================================================
-- 7. AUDIT LOG (for compliance)
-- ==============================================================
CREATE TABLE IF NOT EXISTS audit_log (
    id              UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    event_type      TEXT NOT NULL,              -- 'prediction', 'trade_signal', 'compliance_check'
    event_data      JSONB NOT NULL,
    user_id         TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_audit_event_type ON audit_log(event_type);
CREATE INDEX idx_audit_created ON audit_log(created_at);

-- ==============================================================
-- Row Level Security (Supabase)
-- ==============================================================
ALTER TABLE insider_transactions ENABLE ROW LEVEL SECURITY;
ALTER TABLE predictions ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit_log ENABLE ROW LEVEL SECURITY;
