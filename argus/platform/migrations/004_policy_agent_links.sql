ALTER TABLE policy_revisions ADD COLUMN IF NOT EXISTS agent_turn_id uuid;
DO $$ BEGIN
 ALTER TABLE policy_revisions ADD CONSTRAINT policy_agent_ownership FOREIGN KEY(project_id,agent_turn_id) REFERENCES agent_turns(project_id,id);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;
CREATE UNIQUE INDEX IF NOT EXISTS policy_agent_applied_once ON policy_revisions(agent_turn_id) WHERE agent_turn_id IS NOT NULL;
