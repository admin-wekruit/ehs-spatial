CREATE TABLE IF NOT EXISTS projects (
 id uuid PRIMARY KEY, title text NOT NULL, capability_sha256 char(64) NOT NULL UNIQUE,
 request_id uuid NOT NULL, request_sha256 char(64) NOT NULL,
 default_branch_id uuid NOT NULL, fork_source_revision_id uuid,
 created_at timestamptz NOT NULL DEFAULT now(), UNIQUE(capability_sha256,request_id)
);
CREATE TABLE IF NOT EXISTS scene_branches (
 id uuid PRIMARY KEY, project_id uuid NOT NULL REFERENCES projects(id), kind text NOT NULL CHECK(kind IN ('reconstruction','planning')),
 title text NOT NULL, source_revision_id uuid, head_revision_id uuid NOT NULL,
 request_id uuid, request_sha256 char(64), created_at timestamptz NOT NULL DEFAULT now(),
 UNIQUE(project_id,id), UNIQUE(project_id,request_id)
);
CREATE TABLE IF NOT EXISTS captures (
 id uuid PRIMARY KEY, project_id uuid NOT NULL REFERENCES projects(id), branch_id uuid NOT NULL,
 base_revision_id uuid NOT NULL, revision_id uuid NOT NULL,
 request_id uuid NOT NULL, request_sha256 char(64) NOT NULL,
 target text NOT NULL CHECK(target IN ('scene','standalone_object')), images jsonb NOT NULL,
 task jsonb NOT NULL, created_at timestamptz NOT NULL DEFAULT now(),
 UNIQUE(project_id,id), UNIQUE(project_id,request_id),
 FOREIGN KEY(project_id,branch_id) REFERENCES scene_branches(project_id,id)
);
CREATE TABLE IF NOT EXISTS scene_revisions (
 id uuid PRIMARY KEY, project_id uuid NOT NULL REFERENCES projects(id), branch_id uuid NOT NULL,
 parent_revision_id uuid, source_revision_id uuid, document jsonb NOT NULL,
 document_sha256 char(64) NOT NULL, label text, created_at timestamptz NOT NULL DEFAULT now(),
 UNIQUE(project_id,id), FOREIGN KEY(project_id,branch_id) REFERENCES scene_branches(project_id,id),
 FOREIGN KEY(project_id,parent_revision_id) REFERENCES scene_revisions(project_id,id),
 FOREIGN KEY(source_revision_id) REFERENCES scene_revisions(id)
);
DO $$ BEGIN
 ALTER TABLE projects ADD CONSTRAINT project_default_branch FOREIGN KEY(id,default_branch_id) REFERENCES scene_branches(project_id,id) DEFERRABLE INITIALLY DEFERRED;
 ALTER TABLE projects ADD CONSTRAINT project_fork_source FOREIGN KEY(fork_source_revision_id) REFERENCES scene_revisions(id);
 ALTER TABLE scene_branches ADD CONSTRAINT branch_head FOREIGN KEY(project_id,head_revision_id) REFERENCES scene_revisions(project_id,id) DEFERRABLE INITIALLY DEFERRED;
 ALTER TABLE scene_branches ADD CONSTRAINT branch_source FOREIGN KEY(project_id,source_revision_id) REFERENCES scene_revisions(project_id,id);
 ALTER TABLE captures ADD CONSTRAINT capture_base FOREIGN KEY(project_id,base_revision_id) REFERENCES scene_revisions(project_id,id);
 ALTER TABLE captures ADD CONSTRAINT capture_revision FOREIGN KEY(project_id,revision_id) REFERENCES scene_revisions(project_id,id) DEFERRABLE INITIALLY DEFERRED;
EXCEPTION WHEN duplicate_object THEN NULL; END $$;
CREATE TABLE IF NOT EXISTS edit_batches (
 id uuid PRIMARY KEY, project_id uuid NOT NULL REFERENCES projects(id), branch_id uuid NOT NULL,
 base_revision_id uuid NOT NULL, revision_id uuid NOT NULL, request_id uuid NOT NULL,
 request_sha256 char(64) NOT NULL, operations jsonb NOT NULL, inverse_operations jsonb NOT NULL,
 undo_of uuid, redo_of uuid, created_at timestamptz NOT NULL DEFAULT now(),
 UNIQUE(project_id,id), UNIQUE(project_id,request_id),
 FOREIGN KEY(project_id,branch_id) REFERENCES scene_branches(project_id,id),
 FOREIGN KEY(project_id,base_revision_id) REFERENCES scene_revisions(project_id,id),
 FOREIGN KEY(project_id,revision_id) REFERENCES scene_revisions(project_id,id),
 FOREIGN KEY(project_id,undo_of) REFERENCES edit_batches(project_id,id),
 FOREIGN KEY(project_id,redo_of) REFERENCES edit_batches(project_id,id)
);
CREATE TABLE IF NOT EXISTS jobs (
 id uuid PRIMARY KEY, project_id uuid NOT NULL REFERENCES projects(id), branch_id uuid NOT NULL,
 base_revision_id uuid NOT NULL, request_id uuid NOT NULL, request_sha256 char(64) NOT NULL,
 kind text NOT NULL, inputs jsonb NOT NULL, config jsonb NOT NULL,
 status text NOT NULL CHECK(status IN ('pending_dispatch','queued','running','succeeded','incomplete','failed','outcome_unknown','cancelled')),
 cancel_requested boolean NOT NULL DEFAULT false, attempt integer NOT NULL DEFAULT 0, attempt_token uuid,
 dispatched_at timestamptz, executor_ref text, result jsonb, late_results jsonb NOT NULL DEFAULT '[]',
 result_revision_id uuid, head_advanced boolean NOT NULL DEFAULT false,
 created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now(),
 UNIQUE(project_id,id), UNIQUE(project_id,request_id),
 FOREIGN KEY(project_id,branch_id) REFERENCES scene_branches(project_id,id),
 FOREIGN KEY(project_id,base_revision_id) REFERENCES scene_revisions(project_id,id),
 FOREIGN KEY(project_id,result_revision_id) REFERENCES scene_revisions(project_id,id)
);
CREATE INDEX IF NOT EXISTS jobs_dispatch_outbox ON jobs(created_at) WHERE status='pending_dispatch';
CREATE TABLE IF NOT EXISTS assets (
 id uuid PRIMARY KEY, project_id uuid NOT NULL REFERENCES projects(id), job_id uuid,
 storage_key text NOT NULL, sha256 char(64) NOT NULL, size_bytes bigint NOT NULL CHECK(size_bytes>=0),
 media_type text NOT NULL, metadata jsonb NOT NULL DEFAULT '{}', created_at timestamptz NOT NULL DEFAULT now(),
 UNIQUE(project_id,id), UNIQUE(project_id,storage_key),
 FOREIGN KEY(project_id,job_id) REFERENCES jobs(project_id,id)
);
CREATE TABLE IF NOT EXISTS model_calls (
 id uuid PRIMARY KEY, project_id uuid NOT NULL REFERENCES projects(id), job_id uuid NOT NULL,
 attempt_token uuid NOT NULL, provider text NOT NULL, model text NOT NULL, request_key text NOT NULL,
 code_sha256 text, model_sha256 text, adapter_sha256 text, input_sha256 text,
 status text NOT NULL CHECK(status IN ('reserved','succeeded','failed','outcome_unknown')),
 estimated_cost numeric NOT NULL CHECK(estimated_cost>=0), actual_cost numeric CHECK(actual_cost>=0), response jsonb,
 created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now(),
 UNIQUE(project_id,id), UNIQUE(job_id,request_key),
 FOREIGN KEY(project_id,job_id) REFERENCES jobs(project_id,id)
);
CREATE TABLE IF NOT EXISTS publications (
 id uuid PRIMARY KEY, project_id uuid NOT NULL REFERENCES projects(id), scene_revision_id uuid NOT NULL,
 request_id uuid NOT NULL, request_sha256 char(64) NOT NULL, title text NOT NULL,
 evaluation_ids jsonb NOT NULL, review_ids jsonb NOT NULL, snapshot jsonb NOT NULL,
 created_at timestamptz NOT NULL DEFAULT now(), UNIQUE(project_id,id), UNIQUE(project_id,request_id),
 FOREIGN KEY(project_id,scene_revision_id) REFERENCES scene_revisions(project_id,id)
);
CREATE TABLE IF NOT EXISTS agent_turns (
 id uuid PRIMARY KEY, project_id uuid NOT NULL REFERENCES projects(id), conversation_id uuid NOT NULL,
 branch_id uuid NOT NULL, base_revision_id uuid NOT NULL, request_id uuid NOT NULL,
 request_sha256 char(64) NOT NULL, request jsonb NOT NULL, response jsonb,
 status text NOT NULL CHECK(status IN ('pending','running','succeeded','failed','outcome_unknown')),
 created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now(),
 UNIQUE(project_id,id), UNIQUE(project_id,request_id),
 FOREIGN KEY(project_id,branch_id) REFERENCES scene_branches(project_id,id),
 FOREIGN KEY(project_id,base_revision_id) REFERENCES scene_revisions(project_id,id)
);
CREATE OR REPLACE FUNCTION panoptes_immutable() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN
 RAISE EXCEPTION 'immutable_platform_record'; END $$;
DO $$ DECLARE table_name text; BEGIN
 FOREACH table_name IN ARRAY ARRAY['scene_revisions','edit_batches','publications','assets','captures'] LOOP
  EXECUTE format('DROP TRIGGER IF EXISTS immutable_record ON %I',table_name);
  EXECUTE format('CREATE TRIGGER immutable_record BEFORE UPDATE OR DELETE ON %I FOR EACH ROW EXECUTE FUNCTION panoptes_immutable()',table_name);
 END LOOP;
END $$;
