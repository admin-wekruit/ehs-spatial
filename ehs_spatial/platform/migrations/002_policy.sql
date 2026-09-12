CREATE TABLE IF NOT EXISTS policy_sources (
 id uuid PRIMARY KEY, project_id uuid NOT NULL REFERENCES projects(id),
 content_sha256 char(64) NOT NULL, document jsonb NOT NULL,
 created_at timestamptz NOT NULL DEFAULT now(), UNIQUE(project_id,id)
);
CREATE TABLE IF NOT EXISTS policies (
 id uuid PRIMARY KEY, project_id uuid NOT NULL REFERENCES projects(id), title text NOT NULL,
 request_id uuid NOT NULL, request_sha256 char(64) NOT NULL,
 active_revision_id uuid, draft_revision_id uuid,
 activation_requests jsonb NOT NULL DEFAULT '{}',
 created_at timestamptz NOT NULL DEFAULT now(), UNIQUE(project_id,id),UNIQUE(project_id,request_id)
);
CREATE TABLE IF NOT EXISTS policy_revisions (
 id uuid PRIMARY KEY, project_id uuid NOT NULL, policy_id uuid NOT NULL,
 source_id uuid NOT NULL, parent_revision_id uuid,
 request_id uuid NOT NULL, request_sha256 char(64) NOT NULL,
 document jsonb NOT NULL, document_sha256 char(64) NOT NULL,
 created_at timestamptz NOT NULL DEFAULT now(),UNIQUE(project_id,id),UNIQUE(policy_id,id),UNIQUE(project_id,request_id),
 FOREIGN KEY(project_id,policy_id) REFERENCES policies(project_id,id),
 FOREIGN KEY(project_id,source_id) REFERENCES policy_sources(project_id,id),
 FOREIGN KEY(policy_id,parent_revision_id) REFERENCES policy_revisions(policy_id,id)
);
CREATE TABLE IF NOT EXISTS policy_evaluations (
 id uuid PRIMARY KEY, project_id uuid NOT NULL, scene_revision_id uuid NOT NULL,
 request_id uuid NOT NULL, request_sha256 char(64) NOT NULL, context text NOT NULL CHECK(context IN ('observed','planning')),
 document jsonb NOT NULL, created_at timestamptz NOT NULL DEFAULT now(), UNIQUE(project_id,id),UNIQUE(project_id,request_id),
 FOREIGN KEY(project_id,scene_revision_id) REFERENCES scene_revisions(project_id,id)
);
CREATE TABLE IF NOT EXISTS policy_reviews (
 id uuid PRIMARY KEY, project_id uuid NOT NULL,evaluation_id uuid NOT NULL,finding_id uuid NOT NULL,
 request_id uuid NOT NULL,request_sha256 char(64) NOT NULL,document jsonb NOT NULL,
 created_at timestamptz NOT NULL DEFAULT now(),UNIQUE(project_id,id),UNIQUE(project_id,request_id),
 FOREIGN KEY(project_id,evaluation_id) REFERENCES policy_evaluations(project_id,id)
);
CREATE TABLE IF NOT EXISTS policy_evidence_requests (
 id uuid PRIMARY KEY,project_id uuid NOT NULL,evaluation_id uuid NOT NULL,finding_id uuid NOT NULL,
 request_id uuid NOT NULL,request_sha256 char(64) NOT NULL,document jsonb NOT NULL,
 created_at timestamptz NOT NULL DEFAULT now(),UNIQUE(project_id,id),UNIQUE(project_id,request_id),
 FOREIGN KEY(project_id,evaluation_id) REFERENCES policy_evaluations(project_id,id)
);
CREATE OR REPLACE FUNCTION policy_immutable_record() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN RAISE EXCEPTION 'immutable_policy_record'; END;
$$;
DO $$ DECLARE t text; BEGIN
 FOREACH t IN ARRAY ARRAY['policy_sources','policy_revisions','policy_evaluations','policy_reviews','policy_evidence_requests'] LOOP
  IF NOT EXISTS(SELECT 1 FROM pg_trigger WHERE tgname=t||'_immutable' AND tgrelid=t::regclass) THEN
   EXECUTE format('CREATE TRIGGER %I BEFORE UPDATE OR DELETE ON %I FOR EACH ROW EXECUTE FUNCTION policy_immutable_record()',t||'_immutable',t);
  END IF;
 END LOOP;
END $$;
