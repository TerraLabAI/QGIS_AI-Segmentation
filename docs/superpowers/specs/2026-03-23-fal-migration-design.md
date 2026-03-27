# Migration Azure SAM3 -> fal.ai (text-only)

**Date:** 2026-03-23
**Status:** Approved

## Objectif

Remplacer l'infrastructure Azure Container Apps self-hosted (SAM3) par fal.ai serverless pay-per-request. Simplification radicale : text prompt uniquement, suppression du mode interactif par points et du batch tiling.

## Décisions clés

- Endpoint fal.ai : `fal-ai/sam-3/image-rle` ($0.005/requête, retourne RLE natif)
- Proxy auth : Supabase Edge Function (valide clé PRO, détient la clé fal.ai)
- Pas de limite d'inférences pour l'instant
- Batch tiling supprimé : fal.ai gère le text prompt sur l'image complète nativement
- Mode interactif par points supprimé pour cette itération

## Architecture

```
Plugin QGIS
  -> Supabase Edge Function /fal-proxy
     -> valide api_key (table api_keys, SHA256, active=true)
     -> upload image via fal-client -> image_url temporaire
     -> appelle fal-ai/sam-3/image-rle {image_url, prompt, include_scores, include_boxes}
     -> retourne {rle, scores, boxes}
  -> décode RLE -> masque numpy -> polygone QGIS
```

## Composants

### 1. Supabase Edge Function
- Fichier : `supabase/functions/fal-proxy/index.ts`
- Input : `{image_b64: string, prompt: string, api_key: string}`
- Output : `{rle, scores, boxes}` ou erreur HTTP
- Secrets : `FAL_API_KEY` (stocké dans Supabase secrets, jamais dans le code)
- Auth Supabase : `SUPABASE_SERVICE_ROLE_KEY` natif

### 2. Plugin - `src/core/pro_predictor.py`
- Classe `FalSam3Predictor` remplace `CloudSam3Predictor`
- Méthode unique : `predict_text(image_array: np.ndarray, prompt: str) -> dict`
- Encode image en JPEG base64, POST vers edge function, retourne RLE

### 3. Plugin - `src/core/model_config.py`
- Supprime `SAM3_CLOUD_URL`
- Ajoute `FAL_EDGE_FUNCTION_URL`

### 4. Plugin - `src/ui/ai_segmentation_plugin.py`
- Supprime `_run_pro_detection()` (mode interactif points)
- Supprime logique warm-up / retry exponential
- Simplifie `_run_pro_text_detection()` -> `_run_fal_detection()` (une requête, pas de tiling)
- Simplifie `_on_start_pro_segmentation()` (juste vérifier que clé existe)

## Ce qui est supprimé

- `set_image()` et gestion de session_id
- `warm_up()` et `warm_up_with_retry()`
- `predict()` avec point_coords
- Batch tiling adaptatif (2-6 tuiles)
- Dialog retry warm-up
- `server/sam3/` entier (hors scope de ce changement, à archiver séparément)

## Ce qui ne change pas

- Table `api_keys` Supabase (aucune migration)
- Clés PRO utilisateurs (format `tl_pro_...`)
- UI panneau texte dans le dockwidget
- Conversion masque -> polygone QGIS
- SAM1/SAM2 local (inchangé)

## Erreurs gérées

- 401 : clé invalide -> message utilisateur "Clé PRO invalide"
- 503 : fal.ai ou Supabase indisponible -> message "Service temporairement indisponible"
- Timeout (30s) -> message "Délai dépassé, réessayez"
