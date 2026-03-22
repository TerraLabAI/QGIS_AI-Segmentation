# Per-User API Key Authentication — Design Document

**Date:** 2026-03-22
**Branch:** LA-new-cloud-features
**Status:** Approved by user

---

## Goal

Replace the single shared `PRO_API_KEY` in `.env` with per-user API keys stored in
Supabase, validated by the Azure SAM3 server at session start, and entered by users
via a copy-paste field in the QGIS plugin settings.

---

## Architecture Overview

```
STRIPE (paiement)
    │ webhook
    ▼
SUPABASE — TerraLab-UserBase
    subscriptions.activation_key  ← clé en clair (pour dashboard)
    api_keys.key_hash             ← hash SHA-256
    api_keys.active               ← true / false
    │ query REST (service_role_key) — appelé par Azure à chaque /set_image
    ▼
AZURE — Container App sam-api
    /set_image  → hash(clé reçue) → vérifie dans Supabase
    /predict    → fait confiance au session_id (UUID4)
    /reset      → idem
    ▲ X-Api-Key: tl_pro_xxxxx
    │
PLUGIN QGIS
    QSettings["AISegmentation/pro_api_key"] ← stockage local
    Dock PRO → champ de saisie + bouton Save
```

---

## Key Lifecycle

### New subscriber
1. User pays → Stripe sends webhook
2. Supabase Edge Function generates key `tl_pro_<48 hex chars>` (55 chars total)
3. Raw key stored in `subscriptions.activation_key` (visible in dashboard)
4. SHA-256 hash inserted into `api_keys` with `active = true`
5. User copies key from dashboard and pastes into QGIS plugin

### Subscription expired
1. Stripe sends `customer.subscription.deleted` webhook
2. Edge Function sets `api_keys.active = false` and `revoked_at = now()`
3. Next QGIS session start → Azure gets 401 → plugin shows "Abonnement expiré"

### Migration of 2 existing Pro users
1. SQL migration computes `sha256(activation_key)` for existing subscription rows
2. Inserts into `api_keys` — no action required from the users
3. Their existing keys (`tl_pro_XXXXXX`, 39 chars) continue to work unchanged

---

## Component Changes

### Supabase
- **Migration SQL**: populate `api_keys` from the 2 existing `subscriptions.activation_key` rows
- **Edge Function (Stripe webhook)**: add key generation on `customer.subscription.created`
  and key revocation on `customer.subscription.deleted`

### Azure SAM3 Server (`server/sam3/`)
- New file `supabase_auth.py`: hash incoming key + query Supabase REST API
- `main.py`: replace static `check_api_key` with `validate_api_key` from `supabase_auth`
  - Validation only on `/set_image` (not on `/predict` or `/reset`)
  - `/predict` and `/reset` trust `session_id` (UUID4, 122 bits of entropy)
- `requirements.txt`: add `httpx>=0.27.0`
- Two new env vars in Azure Container App: `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`

### QGIS Plugin
- `src/core/activation_manager.py`: add `get_pro_api_key()` and `set_pro_api_key()`
  storing under `AISegmentation/pro_api_key` in QSettings
- `src/ui/ai_segmentation_plugin.py`: read key from QSettings instead of `.env`
- `src/ui/ai_segmentation_pro_dockwidget.py`: read key from QSettings + add API key
  input field (QLineEdit in Password mode) + Save button to PRO dock
- `i18n/fr.ts`, `i18n/pt_BR.ts`, `i18n/es.ts`: add translations for new UI strings

---

## Out of Scope

- Email sending of the API key (handled separately outside this feature)
- Dashboard web page to display the key (key is available in `subscriptions.activation_key`)
- GPU/CUDA mentions in any UI

---

## Security Decisions

| Element | Decision |
|---------|----------|
| Key storage in Supabase | SHA-256 hash only in `api_keys` (one-way) |
| Raw key | Only in `subscriptions.activation_key` for dashboard display |
| `SUPABASE_SERVICE_ROLE_KEY` | Azure secret only, never in code or git |
| Key in plugin | QSettings (OS-native storage, not committed to git) |
| Current `.env` file | Delete after end-to-end validation |
| SHA-256 without salt | Acceptable for API keys (high natural entropy of long random keys) |

---

## Database Schema Reference

```
api_keys
  id             uuid PK
  contact_id     uuid FK → contacts.id
  key_hash       text UNIQUE  (SHA-256 hex, 64 chars)
  active         boolean DEFAULT true
  created_at     timestamptz
  revoked_at     timestamptz nullable

subscriptions
  id                     uuid PK
  contact_id             uuid FK → contacts.id
  stripe_subscription_id text
  plan                   text ('pro')
  status                 text ('active' | 'canceled' ...)
  current_period_end     timestamptz
  activation_key         text  (raw key, shown in dashboard)
  product_id             text ('ai-segmentation' | 'ai-canvas')

contacts
  id           uuid PK
  email        text UNIQUE
  plan         text ('free' | 'pro')
  auth_user_id uuid FK → auth.users.id
```

---

## Key Format

- **Existing keys** (2 users): `tl_pro_XXXXXX` — 39 chars, stored in `subscriptions.activation_key`
- **New keys**: `tl_pro_` + 48 hex chars = 55 chars total
- Format difference is intentional — validation is hash-based, raw length is irrelevant
