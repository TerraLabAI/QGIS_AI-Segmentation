# Migration fal.ai (SAM3 text-only) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remplacer l'infrastructure Azure Container Apps SAM3 par fal.ai serverless, avec uniquement le mode text prompt (pas de points, pas de tiling).

**Architecture:** Le plugin encode l'image en JPEG base64 et l'envoie à une Supabase Edge Function (`/fal-proxy`), qui valide la clé PRO, uploade l'image vers fal.ai, appelle `fal-ai/sam-3/image-rle`, et retourne le masque en RLE. Le plugin décode le RLE en numpy mask et utilise le pipeline existant de conversion mask -> polygone QGIS. Le warm-up, les sessions, les points, et le tiling disparaissent.

**Tech Stack:** Python (urllib, base64, numpy), Deno/TypeScript (Supabase Edge Functions), `@fal-ai/client` (npm), fal.ai API (`fal-ai/sam-3/image-rle`)

**Spec:** `docs/superpowers/specs/2026-03-23-fal-migration-design.md`

---

## Fichiers touchés

| Fichier | Action | Rôle |
|---------|--------|------|
| `supabase/functions/fal-proxy/index.ts` | Créer | Edge Function proxy: auth + upload + appel fal.ai |
| `src/core/pro_predictor.py` | Réécrire | `FalSam3Predictor` remplace `CloudSam3Predictor` |
| `src/core/model_config.py` | Modifier ligne 73 | Remplace `SAM3_CLOUD_URL` par `FAL_EDGE_FUNCTION_URL` |
| `src/ui/ai_segmentation_plugin.py` | Modifier | Supprime warmup/retry/points PRO, simplifie start PRO, remplace `_run_pro_text_detection` |

---

## Task 1: Créer la Supabase Edge Function

**Files:**
- Create: `supabase/functions/fal-proxy/index.ts`

### Contexte
La Edge Function est un fichier TypeScript Deno hébergé par Supabase. Elle reçoit `{image_b64, prompt, api_key}`, valide la clé PRO dans la table `api_keys` (SHA256 hash + `active=true`), uploade l'image vers fal.ai, appelle `fal-ai/sam-3/image-rle`, et retourne le résultat.

Elle a accès nativement aux variables `SUPABASE_URL` et `SUPABASE_SERVICE_ROLE_KEY`. La clé fal.ai sera ajoutée comme secret Supabase sous le nom `FAL_API_KEY` (jamais dans le code).

- [ ] **Step 1.1: Créer le répertoire**

```bash
mkdir -p "supabase/functions/fal-proxy"
```

- [ ] **Step 1.2: Écrire l'Edge Function**

Créer `supabase/functions/fal-proxy/index.ts` :

```typescript
import { createClient } from "npm:@supabase/supabase-js@2";
import * as fal from "npm:@fal-ai/client";

const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

async function sha256Hex(text: string): Promise<string> {
  const hashBuffer = await crypto.subtle.digest(
    "SHA-256",
    new TextEncoder().encode(text)
  );
  return Array.from(new Uint8Array(hashBuffer))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

function base64ToBlob(b64: string, mimeType: string): Blob {
  const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  return new Blob([bytes], { type: mimeType });
}

Deno.serve(async (req: Request) => {
  if (req.method === "OPTIONS") {
    return new Response("ok", { headers: CORS_HEADERS });
  }

  let body: { image_b64: string; prompt: string; api_key: string };
  try {
    body = await req.json();
  } catch {
    return new Response(JSON.stringify({ error: "Invalid JSON body" }), {
      status: 400,
      headers: { ...CORS_HEADERS, "Content-Type": "application/json" },
    });
  }

  const { image_b64, prompt, api_key } = body;

  if (!image_b64 || !prompt || !api_key) {
    return new Response(
      JSON.stringify({ error: "Missing required fields: image_b64, prompt, api_key" }),
      { status: 400, headers: { ...CORS_HEADERS, "Content-Type": "application/json" } }
    );
  }

  // Validate API key against Supabase api_keys table
  const supabase = createClient(
    Deno.env.get("SUPABASE_URL")!,
    Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!
  );

  const keyHash = await sha256Hex(api_key);
  const { data, error: dbError } = await supabase
    .from("api_keys")
    .select("id")
    .eq("key_hash", keyHash)
    .eq("active", true)
    .maybeSingle();

  if (dbError || !data) {
    return new Response(JSON.stringify({ error: "Invalid or expired API key" }), {
      status: 401,
      headers: { ...CORS_HEADERS, "Content-Type": "application/json" },
    });
  }

  // Configure fal client with our API key (never exposed to users)
  fal.config({ credentials: Deno.env.get("FAL_API_KEY")! });

  // Upload image to fal.ai storage
  let imageUrl: string;
  try {
    const imageBlob = base64ToBlob(image_b64, "image/jpeg");
    imageUrl = await fal.storage.upload(imageBlob);
  } catch (uploadErr) {
    return new Response(
      JSON.stringify({ error: "Image upload failed: " + String(uploadErr) }),
      { status: 502, headers: { ...CORS_HEADERS, "Content-Type": "application/json" } }
    );
  }

  // Call fal.ai SAM3 RLE endpoint
  let result: unknown;
  try {
    result = await fal.run("fal-ai/sam-3/image-rle", {
      input: {
        image_url: imageUrl,
        prompt: prompt,
        include_scores: true,
        include_boxes: true,
        return_multiple_masks: true,
        max_masks: 10,
      },
    });
  } catch (falErr) {
    return new Response(
      JSON.stringify({ error: "fal.ai inference failed: " + String(falErr) }),
      { status: 502, headers: { ...CORS_HEADERS, "Content-Type": "application/json" } }
    );
  }

  return new Response(JSON.stringify(result), {
    status: 200,
    headers: { ...CORS_HEADERS, "Content-Type": "application/json" },
  });
});
```

- [ ] **Step 1.3: Déployer le secret FAL_API_KEY dans Supabase**

Dans le terminal, avec la Supabase CLI (remplacer `<project-ref>` par le ref du projet, et `<fal-api-key>` par la clé fal.ai disponible dans les secrets du projet) :

```bash
supabase secrets set FAL_API_KEY=<fal-api-key> --project-ref <project-ref>
```

- [ ] **Step 1.4: Déployer la Edge Function**

```bash
supabase functions deploy fal-proxy --project-ref <project-ref>
```

L'URL de la fonction sera : `https://<project-ref>.supabase.co/functions/v1/fal-proxy`

- [ ] **Step 1.5: Tester la Edge Function manuellement (curl)**

Avec une image de test base64 et une clé PRO valide :

```bash
curl -X POST https://<project-ref>.supabase.co/functions/v1/fal-proxy \
  -H "Content-Type: application/json" \
  -d '{"image_b64": "<small-jpeg-b64>", "prompt": "building", "api_key": "<valid-tl-pro-key>"}'
```

Attendu : réponse JSON avec `rle`, `scores`, `boxes`.

- [ ] **Step 1.6: Commit**

```bash
git add supabase/functions/fal-proxy/index.ts
git commit -m "feat: add Supabase Edge Function fal-proxy for SAM3 inference via fal.ai"
```

---

## Task 2: Réécrire `src/core/pro_predictor.py`

**Files:**
- Modify: `src/core/pro_predictor.py` (réécriture complète)

### Contexte
`CloudSam3Predictor` (313 lignes) est remplacé par `FalSam3Predictor`. La nouvelle classe :
- `set_image(image_np)` : stocke l'image numpy localement (no-op upload, compatible avec le flow existant de `_extract_and_encode_crop`)
- `predict_text(prompt)` : encode l'image en JPEG base64, POST vers la Edge Function, retourne `{rle, scores, boxes}`
- Conserve `cleanup()` pour compatibilité avec le reste du plugin

Le RLE retourné par fal.ai est au format COCO : `{"counts": [...], "size": [H, W]}` (Fortran order). La méthode `decode_rle_to_mask` le convertit en numpy bool array `(H, W)`.

- [ ] **Step 2.1: Écrire le fichier**

Remplacer intégralement `src/core/pro_predictor.py` :

```python
"""FalSam3Predictor - calls fal.ai SAM3 via Supabase Edge Function proxy."""
import io
import json
import base64
import urllib.request
import urllib.error
from typing import Optional, Tuple

import numpy as np
from PIL import Image as PILImage

from qgis.core import QgsMessageLog, Qgis

from .model_config import FAL_EDGE_FUNCTION_URL

_TIMEOUT_PREDICT = 60


def decode_rle_to_mask(rle: dict) -> np.ndarray:
    """Decode COCO uncompressed RLE to a boolean numpy mask (H, W).

    rle format: {"counts": [n0, n1, n2, ...], "size": [H, W]}
    Counts alternate between 0-pixels and 1-pixels in Fortran (column-major) order.
    """
    counts = rle["counts"]
    h, w = rle["size"]
    flat = np.zeros(h * w, dtype=np.uint8)
    pos = 0
    for i, count in enumerate(counts):
        if i % 2 == 1:
            flat[pos:pos + count] = 1
        pos += count
    # Reshape in Fortran order then transpose to row-major (H, W)
    return flat.reshape(w, h).T.astype(bool)


class FalSam3Predictor:

    def __init__(self, api_key: str, edge_function_url: str = FAL_EDGE_FUNCTION_URL) -> None:
        self._api_key = api_key
        self._edge_function_url = edge_function_url
        self._last_image_np: Optional[np.ndarray] = None
        self.is_image_set = False
        self.original_size: Optional[Tuple[int, int]] = None

    def set_image(self, image_np: np.ndarray) -> None:
        """Store image for subsequent predict_text calls. No network call."""
        self._last_image_np = image_np
        self.original_size = (image_np.shape[0], image_np.shape[1])
        self.is_image_set = True
        QgsMessageLog.logMessage(
            "FalSam3Predictor: image stored locally ({}x{})".format(
                image_np.shape[1], image_np.shape[0]
            ),
            "AI Segmentation", level=Qgis.MessageLevel.Info
        )

    def predict_text(self, prompt: str) -> dict:
        """Send image + prompt to the Edge Function proxy, return raw fal.ai response.

        Returns dict with keys: rle (list of dicts or single dict), scores, boxes, metadata.
        Raises RuntimeError on HTTP error or timeout.
        """
        if self._last_image_np is None:
            raise RuntimeError("No image set. Call set_image first.")

        pil_img = PILImage.fromarray(self._last_image_np)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=90)
        image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        payload = json.dumps({
            "image_b64": image_b64,
            "prompt": prompt,
            "api_key": self._api_key,
        }).encode("utf-8")

        req = urllib.request.Request(
            self._edge_function_url,
            data=payload,
            method="POST",
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT_PREDICT) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                QgsMessageLog.logMessage(
                    "FalSam3Predictor: prediction OK (masks={})".format(  # noqa: E501
                        len(result.get("rle", [])) if isinstance(result.get("rle"), list) else 1
                    ),
                    "AI Segmentation", level=Qgis.MessageLevel.Info
                )
                return result
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = json.loads(e.read().decode("utf-8")).get("error", "")
            except Exception:
                pass
            if e.code == 401:
                raise RuntimeError("Invalid PRO API key.")
            raise RuntimeError("fal.ai proxy error {}: {}".format(e.code, detail))
        except urllib.error.URLError as e:
            raise RuntimeError("fal.ai proxy unreachable: {}".format(e.reason))

    def reset_image(self) -> None:
        self._last_image_np = None
        self.is_image_set = False
        self.original_size = None

    def cleanup(self) -> None:
        self.reset_image()
```

- [ ] **Step 2.2: Vérifier la syntaxe**

```bash
cd "/Users/lilien/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/QGIS_AI-Segmentation-Pro"
python3 -c "import ast; ast.parse(open('src/core/pro_predictor.py').read()); print('OK')"
```

Attendu : `OK`

- [ ] **Step 2.3: Vérifier flake8**

```bash
python3 -m flake8 src/core/pro_predictor.py --max-line-length=120
```

Attendu : aucune erreur.

- [ ] **Step 2.4: Test unitaire decode_rle_to_mask**

```bash
python3 -c "
import sys
sys.path.insert(0, 'src')
import numpy as np

# Simulate: 2x2 mask where only top-left pixel is 1
# COCO RLE Fortran order: [0, 1, 3] -> 0 zeros, 1 one, 3 zeros
# In 2x2 Fortran order: col0=[px0, px1], col1=[px2, px3]
# counts=[0,1,3]: 0 zeros, 1 one (px0=1), 3 zeros
rle = {'counts': [0, 1, 3], 'size': [2, 2]}

# Manual import without QGIS
import importlib.util, types
# Stub QGIS modules
for mod in ['qgis', 'qgis.core']:
    sys.modules[mod] = types.ModuleType(mod)
sys.modules['qgis.core'].QgsMessageLog = type('L', (), {'logMessage': staticmethod(lambda *a, **kw: None)})()
sys.modules['qgis.core'].Qgis = type('Q', (), {'Info': 0})()

spec = importlib.util.spec_from_file_location('pro_predictor', 'src/core/pro_predictor.py')

# Stub model_config
sys.modules['src'] = types.ModuleType('src')
sys.modules['src.core'] = types.ModuleType('src.core')
mc = types.ModuleType('src.core.model_config')
mc.FAL_EDGE_FUNCTION_URL = 'http://localhost'
sys.modules['src.core.model_config'] = mc

# Import via exec
with open('src/core/pro_predictor.py') as f:
    code = f.read().replace('from .model_config import FAL_EDGE_FUNCTION_URL', '')
    code = 'FAL_EDGE_FUNCTION_URL = \"http://localhost\"\n' + code

ns = {}
exec(code, ns)
decode_rle_to_mask = ns['decode_rle_to_mask']

mask = decode_rle_to_mask(rle)
assert mask.shape == (2, 2), f'Shape {mask.shape} != (2,2)'
assert mask[0, 0] == True, f'px[0,0] should be True, got {mask[0,0]}'
assert mask[0, 1] == False
assert mask[1, 0] == False
assert mask[1, 1] == False
print('decode_rle_to_mask: OK')
"
```

Attendu : `decode_rle_to_mask: OK`

- [ ] **Step 2.5: Commit**

```bash
git add src/core/pro_predictor.py
git commit -m "feat: replace CloudSam3Predictor with FalSam3Predictor (fal.ai text-only)"
```

---

## Task 3: Mettre à jour `model_config.py`

**Files:**
- Modify: `src/core/model_config.py:71-73`

- [ ] **Step 3.1: Remplacer la constante Azure par l'URL Edge Function**

Dans `src/core/model_config.py`, remplacer les lignes 71-73 :

```python
# SAM3 cloud-only
SAM3_MODEL_NAME = "SAM 3"
SAM3_CLOUD_URL = "https://sam3-api.kindrock-9d62e9fa.francecentral.azurecontainerapps.io"
```

par :

```python
# SAM3 cloud via fal.ai proxy
SAM3_MODEL_NAME = "SAM 3"
FAL_EDGE_FUNCTION_URL = "https://<project-ref>.supabase.co/functions/v1/fal-proxy"
```

**Note :** Remplacer `<project-ref>` par le ref réel du projet Supabase (visible dans les settings Supabase > Project Settings > General).

- [ ] **Step 3.2: Vérifier syntaxe**

```bash
python3 -c "import ast; ast.parse(open('src/core/model_config.py').read()); print('OK')"
```

Attendu : `OK`

- [ ] **Step 3.3: Commit**

```bash
git add src/core/model_config.py
git commit -m "feat: replace SAM3_CLOUD_URL (Azure) with FAL_EDGE_FUNCTION_URL (Supabase proxy)"
```

---

## Task 4: Supprimer le warm-up et simplifier `_on_start_pro_segmentation`

**Files:**
- Modify: `src/ui/ai_segmentation_plugin.py`

### Contexte
`_CloudWarmupWorker` (lignes 261-280) et toute la logique de warm-up/retry dans `_on_start_pro_segmentation` (lignes 1325-1444) disparaissent. Le nouveau flow : vérifier que la clé n'est pas vide, créer `FalSam3Predictor`, activer le mode PRO. Plus de thread, plus de dialog d'attente.

- [ ] **Step 4.1: Supprimer la classe `_CloudWarmupWorker`**

Supprimer intégralement les lignes 261-280 dans `src/ui/ai_segmentation_plugin.py` :

```python
class _CloudWarmupWorker(QObject):
    finished = pyqtSignal()
    attempt_started = pyqtSignal(int, int)  # (attempt_num, max_attempts)

    def __init__(self, cloud, use_retry=False):
        super().__init__()
        self.cloud = cloud
        self.result = False
        self.error_type = "none"
        self.use_retry = use_retry

    def run(self):
        def on_attempt(n, m):
            self.attempt_started.emit(n, m)
        if self.use_retry and hasattr(self.cloud, 'warm_up_with_retry'):
            self.result, self.error_type = self.cloud.warm_up_with_retry(attempt_callback=on_attempt)
        else:
            self.result = self.cloud.warm_up()
            self.error_type = "unknown" if not self.result else "none"
        self.finished.emit()
```

- [ ] **Step 4.2: Supprimer `_warmup_thread` et `_warmup_worker` du constructeur**

Dans `AISegmentationPlugin.__init__`, chercher et supprimer les lignes qui initialisent ces attributs (grep pour `_warmup_thread` et `_warmup_worker`).

- [ ] **Step 4.3: Réécrire `_on_start_pro_segmentation`**

Remplacer intégralement la méthode `_on_start_pro_segmentation` (lignes 1325-1477) par :

```python
def _on_start_pro_segmentation(self, layer: QgsRasterLayer):
    """Start PRO (SAM 3) segmentation via fal.ai."""
    from ..core.pro_predictor import FalSam3Predictor
    from ..core.activation_manager import get_pro_api_key
    # Note: ensure_venv_packages_available() is intentionally omitted:
    # fal.ai requires no local venv packages (all inference is remote).

    api_key = get_pro_api_key()
    if not api_key:
        QMessageBox.warning(
            self.iface.mainWindow(),
            tr("AI Segmentation PRO"),
            tr(
                "No PRO API key configured.\n\n"
                "Go to the PRO settings to enter your API key."
            )
        )
        return

    # Clean up previous predictor
    if self.predictor:
        try:
            self.predictor.cleanup()
        except Exception:
            pass
    self.predictor = FalSam3Predictor(api_key=api_key)

    # Validate layer
    if not self._is_layer_valid(layer):
        QgsMessageLog.logMessage(
            "Layer was deleted before PRO segmentation could start",
            "AI Segmentation", level=Qgis.MessageLevel.Warning)
        self.predictor.cleanup()
        self.predictor = None
        return

    try:
        layer_name = layer.name().replace(" ", "_")
        raster_path = os.path.normcase(layer.source())
    except RuntimeError:
        self.predictor.cleanup()
        self.predictor = None
        return

    self._reset_session()
    self._current_layer = layer
    self._current_layer_name = layer_name
    self._is_online_layer = self._is_online_provider(layer)
    self._is_non_georeferenced_mode = (
        not self._is_online_layer
        and not self._is_layer_georeferenced(layer)
    )
    self._current_raster_path = raster_path

    # Set up CRS transforms
    canvas_crs = self.iface.mapCanvas().mapSettings().destinationCrs()
    raster_crs = layer.crs() if layer else None
    self._canvas_to_raster_xform = None
    self._raster_to_canvas_xform = None
    if raster_crs and canvas_crs.isValid() and raster_crs.isValid():
        if canvas_crs != raster_crs:
            self._canvas_to_raster_xform = QgsCoordinateTransform(
                canvas_crs, raster_crs, QgsProject.instance())
            self._raster_to_canvas_xform = QgsCoordinateTransform(
                raster_crs, canvas_crs, QgsProject.instance())

    self._active_mode = 'pro'
    self._activate_segmentation_tool()
    QgsMessageLog.logMessage(
        "PRO mode activated (fal.ai)",
        "AI Segmentation", level=Qgis.MessageLevel.Info
    )
```

- [ ] **Step 4.4: Vérifier flake8**

```bash
python3 -m flake8 src/ui/ai_segmentation_plugin.py --max-line-length=120 --select=F401,F841,E501 | head -20
```

- [ ] **Step 4.5: Commit**

```bash
git add src/ui/ai_segmentation_plugin.py
git commit -m "refactor: remove CloudWarmupWorker and warm-up retry logic, simplify PRO start"
```

---

## Task 5: Réécrire `_run_pro_text_detection` sans tiling

**Files:**
- Modify: `src/ui/ai_segmentation_plugin.py`

### Contexte
`_run_pro_text_detection` (lignes 1690-~1910) fait du tiling adaptatif (2-36 tuiles). Elle est remplacée par `_run_fal_detection` : une seule requête sur le centre du canvas visible. Le pipeline `mask_to_polygons` et l'affichage rubber bands restent identiques.

Le flow :
1. Obtenir le prompt texte
2. Extraire le crop centré sur le canvas via `_extract_and_encode_crop` (appelle `predictor.set_image` en interne -> stocke l'image localement)
3. Appeler `predictor.predict_text(prompt)` -> réponse fal.ai
4. Itérer sur les RLEs retournés, décoder chacun, filtrer par score
5. Convertir en polygones, afficher

- [ ] **Step 5.1: Remplacer `_run_pro_text_detection` par `_run_fal_detection`**

Remplacer intégralement la méthode `_run_pro_text_detection` (ligne 1690 jusqu'à la fin de la méthode ~ligne 1910) par :

```python
def _run_fal_detection(self):
    """Detect objects matching the text prompt on the current canvas view."""
    import numpy as np
    from ..core.pro_predictor import decode_rle_to_mask

    if not self._active_dock or not self.predictor:
        return
    text_prompt = self._active_dock.get_pro_text_prompt()
    if not text_prompt:
        return

    raster_layer = self._current_layer
    if raster_layer is None:
        return

    canvas = self.iface.mapCanvas()
    canvas_extent = canvas.extent()
    canvas_center = canvas_extent.center()

    # Convert canvas center to raster CRS if needed
    canvas_crs = canvas.mapSettings().destinationCrs()
    layer_crs = raster_layer.crs()
    if canvas_crs != layer_crs:
        xform = QgsCoordinateTransform(canvas_crs, layer_crs, QgsProject.instance())
        raster_center = xform.transform(canvas_center)
    else:
        raster_center = canvas_center

    # Extract and store the image (calls predictor.set_image internally)
    if not self._extract_and_encode_crop(raster_center):
        return

    if self._current_crop_info is None:
        return

    crop_bounds = self._current_crop_info['bounds']
    img_shape = self._current_crop_info['img_shape']
    img_height, img_width = img_shape
    minx, miny, maxx, maxy = crop_bounds

    crs_value = None
    try:
        if self._current_layer and self._current_layer.crs().isValid():
            crs_value = self._current_layer.crs().authid()
    except RuntimeError:
        pass

    transform_info = {
        "bbox": (minx, miny, maxx, maxy),
        "img_shape": (img_height, img_width),
        "crs": crs_value,
    }

    score_threshold = self._active_dock.get_score_threshold() if self._active_dock else 0.0

    self.iface.messageBar().pushMessage(
        tr("AI Segmentation"),
        tr("Detecting '{prompt}'...").format(prompt=text_prompt),
        level=Qgis.MessageLevel.Info,
        duration=0
    )
    QApplication.processEvents()

    try:
        result = self.predictor.predict_text(text_prompt)
    except RuntimeError as e:
        self.iface.messageBar().clearWidgets()
        QMessageBox.warning(
            self.iface.mainWindow(),
            tr("Detection Error"),
            str(e)
        )
        return

    self.iface.messageBar().clearWidgets()

    # result["rle"] can be a single dict or a list of dicts
    rle_list = result.get("rle", [])
    if isinstance(rle_list, dict):
        rle_list = [rle_list]
    scores_list = result.get("scores") or []

    all_detections = []
    for i, rle in enumerate(rle_list):
        score = float(scores_list[i]) if i < len(scores_list) else 1.0
        if score < score_threshold:
            continue
        try:
            mask = decode_rle_to_mask(rle)
        except Exception as e:
            QgsMessageLog.logMessage(
                "RLE decode error for mask {}: {}".format(i, e),
                "AI Segmentation", level=Qgis.MessageLevel.Warning
            )
            continue
        all_detections.append((mask, score, transform_info))

    if not all_detections:
        self.iface.messageBar().pushMessage(
            tr("AI Segmentation"),
            tr("No objects found for '{prompt}'.").format(prompt=text_prompt),
            level=Qgis.MessageLevel.Info,
            duration=5
        )
        return

    from ..core.polygon_exporter import mask_to_polygons
    all_detections.sort(key=lambda x: x[1], reverse=True)

    IOU_THRESHOLD = 0.3

    def _iou(g1, g2):
        inter = g1.intersection(g2)
        if inter.isEmpty():
            return 0.0
        union = g1.combine(g2)
        return inter.area() / union.area() if union.area() > 0 else 0.0

    accepted_geoms = []
    batch_count = 0

    for mask, score, ti in all_detections:
        if mask.sum() < 20:
            continue
        polys = mask_to_polygons(mask, ti)
        if not polys:
            continue
        geom = QgsGeometry.unaryUnion(polys)
        if not geom or geom.isEmpty():
            continue
        if any(_iou(geom, ag) > IOU_THRESHOLD for ag in accepted_geoms):
            continue
        accepted_geoms.append(geom)

        rb = QgsRubberBand(self.iface.mapCanvas(), QgsWkbTypes.PolygonGeometry)
        rb.setColor(QColor(0, 200, 100, 120))
        rb.setFillColor(QColor(0, 200, 100, 80))
        rb.setWidth(2)
        display_geom = QgsGeometry(geom)
        self._transform_geometry_to_canvas_crs(display_geom)
        rb.setToGeometry(display_geom, None)

        self._pro_pending_detections.append({
            'mask': mask,
            'score': score,
            'transform_info': ti.copy(),
            'rb': rb,
        })
        batch_count += 1

    self._clear_mask_visualization()
    self.current_mask = None
    self.current_transform_info = None

    if batch_count > 0:
        self._pro_detection_batches.append(batch_count)
        self.iface.messageBar().pushMessage(
            tr("AI Segmentation"),
            tr("{n} object(s) detected. Review and save.").format(n=batch_count),
            level=Qgis.MessageLevel.Success,
            duration=5
        )
```

- [ ] **Step 5.2: Mettre à jour les deux connexions du signal**

Il y a exactement **deux occurrences** à remplacer dans `ai_segmentation_plugin.py` :
- Ligne ~544 : connexion initiale dans `__init__` ou `_connect_signals`
- Ligne ~666 : reconnexion dans le bloc de changement de layer

```bash
grep -n "_run_pro_text_detection" src/ui/ai_segmentation_plugin.py
```

Remplacer chaque occurrence de :
```python
self.pro_dock_widget.pro_detect_requested.connect(self._run_pro_text_detection)
```
par :
```python
self.pro_dock_widget.pro_detect_requested.connect(self._run_fal_detection)
```

- [ ] **Step 5.3: Vérifier flake8**

```bash
python3 -m flake8 src/ui/ai_segmentation_plugin.py --max-line-length=120 --select=F401,F841,E501 | head -20
```

- [ ] **Step 5.4: Commit**

```bash
git add src/ui/ai_segmentation_plugin.py
git commit -m "feat: replace tiled _run_pro_text_detection with single-request _run_fal_detection"
```

---

## Task 6: Supprimer le mode interactif PRO par points

**Files:**
- Modify: `src/ui/ai_segmentation_plugin.py`

### Contexte
`_run_pro_detection` (ligne 1479) est le handler de clic interactif PRO. Avec fal.ai text-only, cette méthode n'a plus lieu d'être. Il faut aussi nettoyer le handler de clic sur la carte qui l'appelle (ligne 3011).

- [ ] **Step 6.1: Supprimer `_run_pro_detection`**

Supprimer intégralement la méthode `_run_pro_detection` (ligne 1479 jusqu'à sa fin).

- [ ] **Step 6.2: Nettoyer le handler de clic**

Autour de la ligne 3011, il y a un appel à `self._run_pro_detection(point)`. Chercher le contexte exact :

```bash
grep -n "_run_pro_detection" src/ui/ai_segmentation_plugin.py
```

Remplacer la branche PRO dans le handler de clic de la carte. Si c'était la seule chose que faisait ce handler en mode PRO, soit supprimer la branche, soit la remplacer par un `pass` ou un message informatif (le clic n'a plus d'effet en mode PRO, c'est le bouton "Detect" qui lance la détection).

- [ ] **Step 6.3: Supprimer `_pro_reference_set` si plus utilisé**

```bash
grep -n "_pro_reference_set" src/ui/ai_segmentation_plugin.py
```

Si les seules occurrences restantes sont dans du code déjà supprimé, supprimer aussi l'initialisation dans `__init__` (ligne ~311).

- [ ] **Step 6.4: Vérifier qu'il n'y a plus de références à `CloudSam3Predictor`**

```bash
grep -n "CloudSam3Predictor\|SAM3_CLOUD_URL\|warm_up\|_CloudWarmupWorker" src/ui/ai_segmentation_plugin.py src/core/pro_predictor.py src/core/model_config.py
```

Attendu : aucune occurrence.

- [ ] **Step 6.5: Flake8 final**

```bash
python3 -m flake8 src/ui/ai_segmentation_plugin.py src/core/pro_predictor.py src/core/model_config.py --max-line-length=120
```

Attendu : aucune erreur.

- [ ] **Step 6.6: Commit**

```bash
git add src/ui/ai_segmentation_plugin.py
git commit -m "refactor: remove PRO interactive point detection mode (text-only with fal.ai)"
```

---

## Task 7: Test end-to-end

### Contexte
Il n'existe pas de test automatisé QGIS possible ici. Les validations sont manuelles dans QGIS.

- [ ] **Step 7.1: Test de démarrage PRO sans clé**

Dans QGIS, ouvrir le panneau PRO, vider le champ clé API, cliquer "Start PRO".
Attendu : message d'erreur "No PRO API key configured."

- [ ] **Step 7.2: Test de démarrage PRO avec clé invalide**

Entrer une clé invalide, démarrer. Cliquer "Detect".
Attendu : message d'erreur "Invalid PRO API key." (renvoyé par la Edge Function 401).

- [ ] **Step 7.3: Test de détection avec clé valide**

Avec une clé PRO valide et une image raster chargée dans QGIS :
1. Démarrer le mode PRO -> aucun dialog d'attente, activation immédiate
2. Entrer le prompt "building" (ou tout objet visible dans l'image)
3. Cliquer "Detect"
4. Attendu : rubber bands verts apparaissent sur les objets détectés dans ~5-15s

- [ ] **Step 7.4: Test de sauvegarde polygone**

Après la détection, cliquer "Save polygon" sur un des rubber bands.
Attendu : le polygone est sauvegardé dans la couche vecteur QGIS comme avant.

- [ ] **Step 7.5: Vérifier les logs QGIS**

Ouvrir View > Panels > Log Messages > "AI Segmentation".
Attendu : logs `FalSam3Predictor: image stored locally` et `FalSam3Predictor: prediction OK`.

- [ ] **Step 7.6: Commit final**

```bash
git add .
git commit -m "feat: migration Azure SAM3 -> fal.ai complete (text-only, no sessions, no tiling)"
```
