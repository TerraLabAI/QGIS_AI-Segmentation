# Per-User API Key Authentication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single shared `PRO_API_KEY` in `.env` with per-user API keys stored in Supabase, validated by the Azure SAM3 server at session start, and entered by users in the QGIS plugin settings.

**Architecture:** Each paying user has a unique key (`tl_pro_XXXXXX`) stored in `subscriptions.activation_key` (raw) and hashed in `api_keys.key_hash` (SHA-256). The QGIS plugin stores the key in QSettings. On `/set_image`, the Azure server hashes the received key and queries Supabase to verify it is active — if yes, the session proceeds; if no, 401. Stripe webhooks maintain key lifecycle (generate on subscribe, revoke on cancel).

**Tech Stack:** Python 3.9+ (plugin), FastAPI + httpx (Azure server), Supabase REST API (validation), Supabase Edge Functions / Deno (Stripe webhook), QgsSettings (plugin key storage), SHA-256 (key hashing).

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `server/sam3/supabase_auth.py` | **CREATE** | Hash key + validate against Supabase |
| `server/sam3/main.py` | **MODIFY** | Replace static key check with Supabase validation at `/set_image` |
| `server/sam3/requirements.txt` | **MODIFY** | Add `httpx` |
| `src/core/activation_manager.py` | **MODIFY** | Add `get/set_pro_api_key()` using QSettings |
| `src/ui/ai_segmentation_plugin.py` | **MODIFY** | Read key from QSettings instead of `.env` |
| `src/ui/ai_segmentation_pro_dockwidget.py` | **MODIFY** | Read key from QSettings + add API key input UI |
| `i18n/fr.ts`, `i18n/pt_BR.ts`, `i18n/es.ts` | **MODIFY** | Add translations for new UI strings |
| Supabase migration SQL | **APPLY** | Populate `api_keys` from existing `subscriptions.activation_key` |
| Supabase Edge Function (Stripe webhook) | **MODIFY** | Generate/revoke `api_keys` on subscription events |
| Azure Container App | **CONFIGURE** | Add `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` secrets |

---

## Context: Existing Data

- **2 existing Pro users** have `activation_key` = `tl_pro_XXXXXX` in the `subscriptions` table (clear text, 39 chars).
- The `api_keys` table currently has 0 rows — it was created but never populated.
- The migration (Task 1) will hash these existing keys and insert them into `api_keys`, so existing users need no action.
- New users going forward will have their key generated automatically by the Stripe webhook (Task 5).

---

## Task 1: Supabase Migration — Populate `api_keys` from Existing Subscriptions

**Files:**
- Apply migration via Supabase MCP: no file to create, use `apply_migration`

**Context:** The 2 existing Pro users have `activation_key` in `subscriptions`. We need their SHA-256 hashes in `api_keys` so the Azure server can validate them. We use `encode(sha256(...), 'hex')` which is Postgres's built-in SHA-256 (pgcrypto extension).

- [ ] **Step 1: Check pgcrypto is available**

  Run via Supabase MCP `execute_sql` on project `yrwmbtljenvgahhetudv`:
  ```sql
  SELECT encode(sha256('test'::bytea), 'hex');
  ```
  Expected: returns a 64-char hex string. If error, run `CREATE EXTENSION IF NOT EXISTS pgcrypto;` first.

- [ ] **Step 2: Preview what the migration will insert**

  Run via `execute_sql`:
  ```sql
  SELECT
    s.contact_id,
    encode(sha256(s.activation_key::bytea), 'hex') AS key_hash,
    true AS active,
    now() AS created_at
  FROM subscriptions s
  WHERE s.activation_key IS NOT NULL
    AND s.product_id = 'ai-segmentation'
    AND s.status = 'active'
    AND NOT EXISTS (
      SELECT 1 FROM api_keys ak WHERE ak.contact_id = s.contact_id
    );
  ```
  Expected: 2 rows with valid 64-char hashes.

- [ ] **Step 3: Apply the migration**

  Use Supabase MCP `apply_migration` (NOT `execute_sql`) with name `populate_api_keys_from_subscriptions`:
  ```sql
  INSERT INTO api_keys (contact_id, key_hash, active, created_at)
  SELECT
    s.contact_id,
    encode(sha256(s.activation_key::bytea), 'hex') AS key_hash,
    true AS active,
    now() AS created_at
  FROM subscriptions s
  WHERE s.activation_key IS NOT NULL
    AND s.product_id = 'ai-segmentation'
    AND s.status = 'active'
    AND NOT EXISTS (
      SELECT 1 FROM api_keys ak WHERE ak.contact_id = s.contact_id
    );
  ```

- [ ] **Step 4: Verify migration**

  Run via `execute_sql`:
  ```sql
  SELECT ak.active, ak.created_at, LENGTH(ak.key_hash) AS hash_len,
         c.plan, s.activation_key IS NOT NULL AS has_key
  FROM api_keys ak
  JOIN contacts c ON c.id = ak.contact_id
  JOIN subscriptions s ON s.contact_id = ak.contact_id;
  ```
  Expected: 2 rows, `active=true`, `hash_len=64`, `plan=pro`.

- [ ] **Step 5: Commit**

  ```bash
  git add -A
  git commit -m "chore: document Supabase migration for per-user api_keys"
  ```

---

## Task 2: Azure Server — Per-User Key Validation via Supabase

**Files:**
- Create: `server/sam3/supabase_auth.py`
- Modify: `server/sam3/main.py`
- Modify: `server/sam3/requirements.txt`

**Context:** Currently `main.py` has a global `API_KEY` env var and a `check_api_key()` function called on all 3 endpoints. We replace this with a Supabase lookup at `/set_image` only. `/predict` and `/reset` rely on `session_id` (UUID4, unguessable) — no re-validation needed.

The server will need 2 new env vars set in Azure (Task 6): `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY`. In dev, if these are unset, validation is skipped (so local dev still works).

- [ ] **Step 1: Add `httpx` to requirements**

  In `server/sam3/requirements.txt`, add after `numpy`:
  ```
  httpx>=0.27.0
  ```

- [ ] **Step 2: Create `server/sam3/supabase_auth.py`**

  ```python
  """Supabase-based per-user API key validation for the SAM3 server."""
  import hashlib
  import os

  import httpx
  from fastapi import HTTPException

  SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
  SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")


  def hash_api_key(raw_key: str) -> str:
      return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


  def validate_api_key(raw_key: str) -> None:
      """Validate key against Supabase api_keys table.

      Raises HTTPException 401 if invalid/revoked, 503 if Supabase unreachable.
      If SUPABASE_URL is not configured, validation is skipped (dev mode).
      """
      if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
          return  # dev mode: no Supabase configured

      if not raw_key:
          raise HTTPException(status_code=401, detail="API key required")

      key_hash = hash_api_key(raw_key)

      try:
          resp = httpx.get(
              f"{SUPABASE_URL}/rest/v1/api_keys",
              params={
                  "key_hash": f"eq.{key_hash}",
                  "active": "eq.true",
                  "select": "id",
                  "limit": "1",
              },
              headers={
                  "apikey": SUPABASE_SERVICE_ROLE_KEY,
                  "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
              },
              timeout=5.0,
          )
      except httpx.RequestError:
          raise HTTPException(status_code=503, detail="Auth service unavailable")

      if resp.status_code != 200 or not resp.json():
          raise HTTPException(status_code=401, detail="Invalid or expired API key")
  ```

- [ ] **Step 3: Update `main.py` — remove static key, add Supabase validation**

  Remove these lines near the top of `main.py`:
  ```python
  API_KEY = os.environ.get("API_KEY", "")
  ```

  Remove the entire `check_api_key` function (lines ~73-75):
  ```python
  def check_api_key(x_api_key: Optional[str] = Header(None)):
      if API_KEY and x_api_key != API_KEY:
          raise HTTPException(status_code=401, detail="Invalid API key")
  ```

  Add import at the top of `main.py` (after existing imports):
  ```python
  from supabase_auth import validate_api_key
  ```

  In the `/set_image` endpoint, replace `check_api_key(x_api_key)` with:
  ```python
  validate_api_key(x_api_key or "")
  ```

  In the `/predict` endpoint, remove the `x_api_key` parameter and `check_api_key(x_api_key)` call entirely:
  ```python
  # Before:
  def predict(req: PredictRequest, x_api_key: Optional[str] = Header(None)):
      check_api_key(x_api_key)

  # After:
  def predict(req: PredictRequest):
  ```

  In the `/reset` endpoint, remove the `x_api_key` parameter and `check_api_key(x_api_key)` call entirely:
  ```python
  # Before:
  def reset(session_id: str, x_api_key: Optional[str] = Header(None)):
      check_api_key(x_api_key)

  # After:
  def reset(session_id: str):
  ```

- [ ] **Step 4: Write test for `supabase_auth.py`**

  Create `server/sam3/tests/test_supabase_auth.py`:
  ```python
  import hashlib
  from unittest.mock import patch, MagicMock
  import pytest
  from fastapi import HTTPException

  # Set env vars before import
  import os
  os.environ["SUPABASE_URL"] = "https://test.supabase.co"
  os.environ["SUPABASE_SERVICE_ROLE_KEY"] = "test-key"

  from supabase_auth import hash_api_key, validate_api_key


  def test_hash_api_key():
      result = hash_api_key("tl_pro_abc123")
      assert len(result) == 64
      assert result == hashlib.sha256("tl_pro_abc123".encode()).hexdigest()


  def test_validate_valid_key():
      mock_resp = MagicMock()
      mock_resp.status_code = 200
      mock_resp.json.return_value = [{"id": "some-uuid"}]
      with patch("supabase_auth.httpx.get", return_value=mock_resp):
          validate_api_key("tl_pro_valid")  # must not raise


  def test_validate_invalid_key():
      mock_resp = MagicMock()
      mock_resp.status_code = 200
      mock_resp.json.return_value = []  # no matching row
      with patch("supabase_auth.httpx.get", return_value=mock_resp):
          with pytest.raises(HTTPException) as exc_info:
              validate_api_key("tl_pro_bad")
          assert exc_info.value.status_code == 401


  def test_validate_empty_key():
      with pytest.raises(HTTPException) as exc_info:
          validate_api_key("")
      assert exc_info.value.status_code == 401


  def test_validate_supabase_unreachable():
      import httpx
      with patch("supabase_auth.httpx.get", side_effect=httpx.RequestError("timeout")):
          with pytest.raises(HTTPException) as exc_info:
              validate_api_key("tl_pro_any")
          assert exc_info.value.status_code == 503
  ```

- [ ] **Step 5: Run the tests**

  ```bash
  cd "server/sam3"
  pip install httpx pytest fastapi
  python -m pytest tests/test_supabase_auth.py -v
  ```
  Expected: all 5 tests PASS.

- [ ] **Step 6: Commit**

  ```bash
  git add server/sam3/supabase_auth.py server/sam3/main.py \
          server/sam3/requirements.txt server/sam3/tests/
  git commit -m "feat: validate per-user API keys via Supabase at session start"
  ```

---

## Task 3: Plugin — Store API Key in QSettings

**Files:**
- Modify: `src/core/activation_manager.py`
- Modify: `src/ui/ai_segmentation_plugin.py`
- Modify: `src/ui/ai_segmentation_pro_dockwidget.py`

**Context:** Both `plugin.py` and `pro_dockwidget.py` currently read `PRO_API_KEY` from a `.env` file. We replace both with a call to `get_pro_api_key()` from `activation_manager.py`. The key will be stored under `AISegmentation/pro_api_key` in QSettings (persists between QGIS sessions, OS-native storage).

- [ ] **Step 1: Add `get/set_pro_api_key` to `activation_manager.py`**

  In `src/core/activation_manager.py`, add after the existing `ACTIVATION_KEY` constant:
  ```python
  PRO_API_KEY_SETTING = f"{SETTINGS_PREFIX}/pro_api_key"
  ```

  Add these two functions after `deactivate_plugin()`:
  ```python
  def get_pro_api_key() -> str:
      """Return the stored PRO API key, or empty string if not set."""
      settings = QgsSettings()
      return settings.value(PRO_API_KEY_SETTING, "", type=str)


  def set_pro_api_key(key: str) -> None:
      """Store the PRO API key in QSettings."""
      settings = QgsSettings()
      settings.setValue(PRO_API_KEY_SETTING, key.strip())
  ```

- [ ] **Step 2: Update `ai_segmentation_plugin.py`**

  Find the import line (around line 1336 context):
  ```python
  from ..core.activation_manager import ...
  ```
  Add `get_pro_api_key` to that import (or add a new import line if needed).

  Replace the entire `.env` reading block (lines ~1336-1344) in the `_start_pro_segmentation` method:
  ```python
  # REMOVE:
  env_path = pathlib.Path(__file__).parent.parent.parent / ".env"
  api_key = ""
  if env_path.exists():
      with open(env_path, encoding="utf-8") as f:
          for line in f:
              line = line.strip()
              if line.startswith("PRO_API_KEY="):
                  api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                  break

  # REPLACE WITH:
  api_key = get_pro_api_key()
  ```

  Update the 401 error message (lines ~1358-1362) to remove the `.env` reference:
  ```python
  # REMOVE:
  tr(
      "Invalid PRO API key.\n\n"
      "Check the value of PRO_API_KEY in:\n"
      "{}"
  ).format(str(env_path))

  # REPLACE WITH:
  tr(
      "Invalid PRO API key.\n\n"
      "Go to the PRO settings to update your API key."
  )
  ```

  Update the second 401 error message (line ~1409-1412) the same way.

  Remove the `import pathlib` if it is no longer used elsewhere in the file (search first).

- [ ] **Step 3: Update `ai_segmentation_pro_dockwidget.py`**

  Add `get_pro_api_key` to the activation_manager import (line 23):
  ```python
  from ..core.activation_manager import is_plugin_activated, get_pro_api_key
  ```

  Replace the `.env` reading block in `_on_start_pro_clicked` (lines ~472-480):
  ```python
  # REMOVE:
  env_path = pathlib.Path(__file__).parent.parent.parent / ".env"
  api_key = ""
  if env_path.exists():
      with open(env_path, encoding="utf-8") as f:
          for line in f:
              line = line.strip()
              if line.startswith("PRO_API_KEY="):
                  api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                  break

  # REPLACE WITH:
  api_key = get_pro_api_key()
  ```

  Update the error message (lines ~483-492):
  ```python
  # REMOVE old message referencing .env file:
  tr(
      "PRO API key is not configured.\n\n"
      "Create the file .env at the root of the plugin directory\n"
      "with the content:\n"
      "PRO_API_KEY=your_key_here"
  )

  # REPLACE WITH:
  tr(
      "PRO API key is not configured.\n\n"
      "Enter your API key in the PRO settings panel."
  )
  ```

  Remove `import pathlib` from the dockwidget if no longer used.

- [ ] **Step 4: Commit**

  ```bash
  git add src/core/activation_manager.py \
          src/ui/ai_segmentation_plugin.py \
          src/ui/ai_segmentation_pro_dockwidget.py
  git commit -m "feat: store PRO API key in QSettings instead of .env file"
  ```

---

## Task 4: Plugin UI — API Key Input Field in PRO Settings

**Files:**
- Modify: `src/ui/ai_segmentation_pro_dockwidget.py`
- Modify: `i18n/fr.ts`, `i18n/pt_BR.ts`, `i18n/es.ts`

**Context:** Users need a way to enter their API key in the plugin. Add a small settings section at the bottom of the PRO dock widget with: a label, a password-style QLineEdit (so the key is not visible by default), and a "Save" button. On save, call `set_pro_api_key()`.

**New UI strings to add to all 3 .ts files:**
- `"API Key"` (label)
- `"Enter your tl_pro_... key"` (placeholder)
- `"Save"` (button)
- `"API key saved."` (confirmation)
- `"Invalid PRO API key.\n\nGo to the PRO settings to update your API key."` (error)
- `"PRO API key is not configured.\n\nEnter your API key in the PRO settings panel."` (error)

- [ ] **Step 1: Add API key section to the dockwidget `__init__`**

  In `ai_segmentation_pro_dockwidget.py`, find where the widget layout is constructed (search for `QVBoxLayout` or `addWidget` calls). Add at the end of the layout construction:

  ```python
  from qgis.PyQt.QtWidgets import QLineEdit, QPushButton, QLabel, QHBoxLayout

  # API key section
  api_key_label = QLabel(tr("API Key"))
  self._api_key_edit = QLineEdit()
  self._api_key_edit.setEchoMode(QLineEdit.Password)
  self._api_key_edit.setPlaceholderText(tr("Enter your tl_pro_... key"))
  self._api_key_edit.setText(get_pro_api_key())

  api_key_save_btn = QPushButton(tr("Save"))
  api_key_save_btn.clicked.connect(self._on_save_api_key)

  api_key_row = QHBoxLayout()
  api_key_row.addWidget(self._api_key_edit)
  api_key_row.addWidget(api_key_save_btn)

  # Add to main layout (adjust variable name to match existing layout)
  main_layout.addWidget(api_key_label)
  main_layout.addLayout(api_key_row)
  ```

  Add the save handler method:
  ```python
  def _on_save_api_key(self):
      from qgis.PyQt.QtWidgets import QMessageBox
      from ..core.activation_manager import set_pro_api_key
      key = self._api_key_edit.text().strip()
      set_pro_api_key(key)
      QMessageBox.information(self, tr("API Key"), tr("API key saved."))
  ```

  **Important:** Read the full `__init__` of the dockwidget before making this change to find the correct layout variable name and insertion point.

- [ ] **Step 2: Add i18n strings to all 3 .ts files**

  In `i18n/fr.ts`, inside `<context><name>AISegmentation</name>`, add:
  ```xml
  <message>
      <source>API Key</source>
      <translation>Clé API</translation>
  </message>
  <message>
      <source>Enter your tl_pro_... key</source>
      <translation>Entrez votre clé tl_pro_...</translation>
  </message>
  <message>
      <source>Save</source>
      <translation>Enregistrer</translation>
  </message>
  <message>
      <source>API key saved.</source>
      <translation>Clé API enregistrée.</translation>
  </message>
  <message>
      <source>Invalid PRO API key.

Go to the PRO settings to update your API key.</source>
      <translation>Clé API PRO invalide.

Allez dans les paramètres PRO pour mettre à jour votre clé.</translation>
  </message>
  <message>
      <source>PRO API key is not configured.

Enter your API key in the PRO settings panel.</source>
      <translation>La clé API PRO n'est pas configurée.

Entrez votre clé API dans le panneau des paramètres PRO.</translation>
  </message>
  ```

  Repeat for `i18n/pt_BR.ts` (Portuguese translations) and `i18n/es.ts` (Spanish translations). Use accurate translations for each language.

- [ ] **Step 3: Manual test in QGIS**

  - Open QGIS, load the plugin
  - Open the PRO dock widget
  - Verify the API Key field appears at the bottom
  - Enter a test key, click Save
  - Close and reopen QGIS — verify the key is still there (persisted in QSettings)
  - Enter the real `tl_pro_XXXXXX` key, click Save
  - Click "Start PRO" — verify no "PRO API Key Missing" error

- [ ] **Step 4: Commit**

  ```bash
  git add src/ui/ai_segmentation_pro_dockwidget.py \
          i18n/fr.ts i18n/pt_BR.ts i18n/es.ts
  git commit -m "feat: add API key input field to PRO dock widget"
  ```

---

## Task 5: Stripe Webhook — Auto-Generate and Revoke Keys

**Files:**
- Modify: existing Stripe webhook handler (Supabase Edge Function — find it in your Supabase project)

**Context:** When a user pays via Stripe, the existing webhook already creates a `subscriptions` row with `activation_key`. We need to add two behaviors:
1. **On new subscription**: generate a key, store raw in `subscriptions.activation_key`, store hash in `api_keys`.
2. **On subscription cancelled/expired**: set `api_keys.active = false` for that contact.

Do NOT add email sending logic.

**Finding the webhook:** In your Supabase dashboard, go to Edge Functions. Find the function that handles Stripe `customer.subscription.*` events. It should already write to `subscriptions`.

- [ ] **Step 1: Find the existing Stripe webhook Edge Function**

  In the Supabase dashboard or via Supabase CLI, list Edge Functions:
  ```bash
  supabase functions list
  ```
  Identify the function handling Stripe webhooks (likely `stripe-webhook` or similar).

- [ ] **Step 2: Understand the key generation pattern**

  Keys follow the format `tl_pro_XXXXXX` (39 chars total). The existing 2 users have keys like `tl_pro_6XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` (length 39). Generate new keys consistently:

  In Deno/TypeScript:
  ```typescript
  function generateApiKey(): string {
    const bytes = crypto.getRandomValues(new Uint8Array(24));
    const hex = Array.from(bytes).map(b => b.toString(16).padStart(2, '0')).join('');
    return `tl_pro_${hex}`; // "tl_pro_" (7) + 48 hex chars = 55 chars total
  }

  async function hashApiKey(rawKey: string): Promise<string> {
    const encoder = new TextEncoder();
    const data = encoder.encode(rawKey);
    const hashBuffer = await crypto.subtle.digest('SHA-256', data);
    return Array.from(new Uint8Array(hashBuffer))
      .map(b => b.toString(16).padStart(2, '0'))
      .join('');
  }
  ```

  **Decision on key length:** New keys will be 55 chars (`tl_pro_` + 48 hex). Existing keys are 39 chars. This is intentional — the system validates by SHA-256 hash only, so the raw key length is irrelevant. Do NOT try to match the old length.

- [ ] **Step 3: Add key generation on `customer.subscription.created`**

  In the webhook handler, in the section that handles new/active subscriptions (where it inserts into `subscriptions`), add after the subscription insert:

  ```typescript
  // Generate API key for the new subscriber
  const rawKey = generateApiKey();
  const keyHash = await hashApiKey(rawKey);

  // Update subscription with raw key (for dashboard display)
  await supabaseAdmin
    .from('subscriptions')
    .update({ activation_key: rawKey })
    .eq('id', newSubscription.id);

  // Insert hash into api_keys
  await supabaseAdmin
    .from('api_keys')
    .upsert({
      contact_id: contactId,
      key_hash: keyHash,
      active: true,
    }, { onConflict: 'contact_id' }); // one active key per user
  ```

- [ ] **Step 4: Add key revocation on `customer.subscription.deleted`**

  In the webhook handler, in the section handling cancellation/deletion:
  ```typescript
  // Revoke the API key
  await supabaseAdmin
    .from('api_keys')
    .update({ active: false, revoked_at: new Date().toISOString() })
    .eq('contact_id', contactId);

  // Also update contact plan to free
  await supabaseAdmin
    .from('contacts')
    .update({ plan: 'free' })
    .eq('id', contactId);
  ```

- [ ] **Step 5: Deploy the updated Edge Function**

  ```bash
  supabase functions deploy stripe-webhook
  ```

- [ ] **Step 6: Test with a Stripe test webhook**

  In Stripe dashboard, use "Send test webhook" for `customer.subscription.created`.
  Then verify in Supabase:
  ```sql
  SELECT ak.active, ak.key_hash, s.activation_key
  FROM api_keys ak
  JOIN subscriptions s ON s.contact_id = ak.contact_id
  ORDER BY ak.created_at DESC LIMIT 5;
  ```
  Expected: new row with `active=true`, matching hashes.

- [ ] **Step 7: Commit**

  If the Edge Function source is tracked in your git repo (e.g., `supabase/functions/stripe-webhook/`), commit it:
  ```bash
  git add supabase/functions/stripe-webhook/
  git commit -m "feat: auto-generate and revoke api_keys on Stripe subscription events"
  ```
  If the function is managed only via Supabase dashboard/CLI and not in git, skip this step — deployment in Step 5 is sufficient.

---

## Task 6: Azure — Add Supabase Secrets to Container App

**Context:** The Azure SAM3 server (`sam-api` container app) needs `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` as environment variables. These must be stored as Azure secrets (not plain env vars).

**Values to retrieve:**
- `SUPABASE_URL`: `https://yrwmbtljenvgahhetudv.supabase.co`
- `SUPABASE_SERVICE_ROLE_KEY`: find in Supabase dashboard → Project Settings → API → `service_role` key (secret, starts with `eyJ...`)

- [ ] **Step 1: Add secrets to Azure Container App**

  ```bash
  az containerapp secret set \
    --name sam-api \
    --resource-group terralab-sam \
    --secrets \
      supabase-url="https://yrwmbtljenvgahhetudv.supabase.co" \
      supabase-service-role-key="<YOUR_SERVICE_ROLE_KEY>"
  ```

- [ ] **Step 2: Map secrets to environment variables**

  ```bash
  az containerapp update \
    --name sam-api \
    --resource-group terralab-sam \
    --set-env-vars \
      SUPABASE_URL=secretref:supabase-url \
      SUPABASE_SERVICE_ROLE_KEY=secretref:supabase-service-role-key
  ```

- [ ] **Step 3: Verify the container app restarted with new env vars**

  ```bash
  az containerapp show \
    --name sam-api \
    --resource-group terralab-sam \
    --query "properties.template.containers[0].env"
  ```
  Expected: `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` appear with value `secretref:supabase-url` and `secretref:supabase-service-role-key` respectively. Azure always shows `secretref:` references here, never the raw secret value — this is correct behavior, not an error.

- [ ] **Step 4: Rebuild and redeploy the Docker image** (required to include `httpx`)

  ```bash
  az acr build \
    --registry terralabsamacr \
    --image sam-api:gpu-v2 \
    server/sam3 \
    --file server/sam3/Dockerfile
  ```

  Then update the container app to use the new image:
  ```bash
  az containerapp update \
    --name sam-api \
    --resource-group terralab-sam \
    --image terralabsamacr.azurecr.io/sam-api:gpu-v2
  ```

- [ ] **Step 5: End-to-end test**

  Using the existing `tl_pro_XXXXXX` key from `subscriptions.activation_key`:
  1. Set the key in the QGIS plugin API Key field (Task 4)
  2. Start a PRO segmentation session
  3. Verify warm-up succeeds and a segmentation completes without 401 errors
  4. Temporarily set `api_keys.active = false` in Supabase for that user
  5. Try to start a new PRO session → verify 401 error appears in plugin
  6. Reset `api_keys.active = true`

- [ ] **Step 6: Commit**

  ```bash
  git commit -m "feat: configure Azure SAM3 server with Supabase credentials for key validation"
  ```

---

## Execution Order

Tasks **1, 2, 3** can be worked in parallel (no dependencies between them).
Task **4** depends on Task 3 (needs `get_pro_api_key` from Task 3).
Task **5** is independent (Supabase Edge Function only).
Task **6** depends on Task 2 (needs the new server code deployed).

Recommended order: 1 → 2 → 3 → 4 → 5 → 6

---

## Security Notes

- The `SUPABASE_SERVICE_ROLE_KEY` bypasses Row Level Security. Keep it **only in Azure secrets**, never in code or git.
- The `.env` file with the old `PRO_API_KEY` should be deleted (or at minimum removed from git history) once all users have migrated. Do not do this until the migration is validated end-to-end.
- `api_keys.key_hash` stores SHA-256 with no salt, which is acceptable for API keys (they are long, random, and high-entropy). Do not use this approach for passwords.
