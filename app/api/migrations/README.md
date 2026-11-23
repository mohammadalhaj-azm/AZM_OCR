# Database migrations (PostgreSQL job store)

Each file is one migration, named `NNN_short_name.sql` and numbered from `001` with no gaps. At startup the service
applies the missing migrations in order, in one transaction, under an advisory lock (so replicas that start
together don't race), and records each one in `ocr_schema_migrations`.

Rules:
- **Never edit a migration that has been deployed.** Add a new numbered file instead.
- **Keep every migration backward compatible** with the release before it: add tables and nullable columns (or
  columns with a default), and drop things only in a later release once no running replica uses them. During a
  rolling deploy, older replicas keep running against the newer schema.
- Document content never goes into a plain column. It belongs in the sealed `bytea` columns (see `001_init.sql`).
