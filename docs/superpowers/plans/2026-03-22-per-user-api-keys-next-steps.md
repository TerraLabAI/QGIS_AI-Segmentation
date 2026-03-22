# Per-User API Keys — Actions manuelles restantes

Toute l'implémentation technique est terminée. Il reste 3 actions manuelles à faire.

---

## Étape 1 — Relier Stripe à Supabase

Dans **Stripe Dashboard** → Developers → Webhooks → "Add endpoint"

- **URL** : `https://yrwmbtljenvgahhetudv.supabase.co/functions/v1/stripe-webhook`
- **Événements à cocher** :
  - `customer.subscription.created`
  - `customer.subscription.updated`
  - `customer.subscription.deleted`
  - `invoice.paid`

Après avoir cliqué "Add endpoint", Stripe affiche un **Signing secret** (commence par `whsec_...`). Copie-le — tu en as besoin à l'étape suivante.

---

## Étape 2 — Donner les clés Stripe à Supabase

Dans **Supabase Dashboard** → Edge Functions → `stripe-webhook` → Secrets

Ajoute ces 2 secrets :

| Nom | Valeur |
|-----|--------|
| `STRIPE_WEBHOOK_SECRET` | Le `whsec_...` copié depuis Stripe (étape 1) |
| `STRIPE_SECRET_KEY` | Ta clé secrète Stripe (`sk_live_...` ou `sk_test_...`) |

---

## Étape 3 — Communiquer leur clé aux utilisateurs existants

Les 2 utilisateurs Pro actuels ont déjà une clé générée. Elle est visible dans :

**Supabase Dashboard** → Table Editor → table `subscriptions` → colonne `activation_key`

Tu dois leur envoyer cette clé et leur expliquer :

> "Ouvre QGIS, dans le dock AI Segmentation PRO, fais défiler vers le bas jusqu'au champ **API Key**, colle ta clé et clique **Save**."

---

## Après ces 3 étapes

Le système est entièrement opérationnel :
- Nouveau paiement Stripe → clé générée automatiquement dans Supabase
- Abonnement annulé → clé désactivée automatiquement
- Utilisateur colle sa clé dans QGIS → accès au serveur IA validé en temps réel
