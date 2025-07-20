-- Enable extensions for text processing and similarity
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS fuzzystrmatch;

-- Source systems information
CREATE TABLE source_systems (
    id SERIAL PRIMARY KEY,
    system_name TEXT NOT NULL,
    system_description TEXT
);

-- Nodes table - stores all entities (source records AND clusters)
CREATE TABLE nodes (
    id SERIAL PRIMARY KEY,
    type TEXT NOT NULL,                     -- 'RECORD', 'CLUSTER'
    source_system_id INTEGER REFERENCES source_systems(id) NULL, -- Only for RECORDs
    external_id TEXT,                       -- Only for RECORDs
    raw_data JSONB,                         -- Original data (for RECORDs)
    normalized_data JSONB,                  -- Normalized data for comparison (for RECORDs)
    golden_data JSONB,                      -- Best composite record (for CLUSTERs)
    confidence_score FLOAT,                 -- For CLUSTER nodes
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    -- Ensure uniqueness for source system records
    CONSTRAINT unique_source_record UNIQUE (source_system_id, external_id)
        DEFERRABLE INITIALLY DEFERRED, -- Only applies when both fields are not null
    -- Ensure external_id is present for RECORDs with a source_system_id
    CONSTRAINT valid_record_requires_external_id 
        CHECK (type != 'RECORD' OR source_system_id IS NULL OR external_id IS NOT NULL)
);

-- Create GIN indexes for JSONB and text search
CREATE INDEX nodes_raw_data_gin_idx ON nodes USING GIN (raw_data);
CREATE INDEX nodes_normalized_data_gin_idx ON nodes USING GIN (normalized_data);
CREATE INDEX nodes_type_idx ON nodes (type);

-- Relationships table - all relationships between nodes
CREATE TABLE relationships (
    id SERIAL PRIMARY KEY,
    source INTEGER REFERENCES nodes(id) ON DELETE CASCADE,
    target INTEGER REFERENCES nodes(id) ON DELETE CASCADE,
    type TEXT NOT NULL,                    -- 'SIMILAR_TO', 'BELONGS_TO'
    properties JSONB,                      -- All relationship properties (scores, evidence, etc.)
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    -- Ensure no duplicate undirected relationships for same type
    CONSTRAINT unique_relationship UNIQUE (
        LEAST(source, target), 
        GREATEST(source, target),
        type
    ),
    -- Ensure no self-loops
    CHECK (source <> target)
);

-- Create indexes for relationship lookups
CREATE INDEX relationships_source_idx ON relationships(source);
CREATE INDEX relationships_target_idx ON relationships(target);
CREATE INDEX relationships_type_idx ON relationships(type);
CREATE INDEX relationships_properties_gin_idx ON relationships USING GIN (properties);

-- Create a view for records with their cluster assignments
CREATE VIEW record_clusters AS
SELECT 
    r.id AS record_id,
    r.source_system_id,
    r.external_id,
    r.raw_data,
    r.normalized_data,
    c.id AS cluster_id,
    c.golden_data,
    c.confidence_score
FROM 
    nodes r
JOIN 
    relationships rel ON r.id = rel.source
JOIN 
    nodes c ON rel.target = c.id
WHERE 
    r.type = 'RECORD'
    AND c.type = 'CLUSTER'
    AND rel.type = 'BELONGS_TO';